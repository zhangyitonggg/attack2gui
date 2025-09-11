# Please replace the provided qwen2_vl/ folder with the one in transformers/src/transformers/models/ to ensure the experiments can be correctly reproduced.
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import *

from PIL import Image
import torch
import os
import json
import random
from tqdm import tqdm
from transformers.models.qwen2_vl.myImageProcessor import create_preprocess


import numpy as np
import types
import copy
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import argparse
import torch.nn.functional as F

from torchvision.transforms import Resize, InterpolationMode
from qwen_hook import get_hook_logger

from prompts import system_prompt, get_user_prompt

resized_height = 336
resized_width = 336

def one_step(model, tokenizer, original_processor, main_device,
        image, raw_image_tensor, image_tensor, instruction, target_text, mask=None,
        epsilon=16/255, alpha=1/255, max_len=1024, hook_logger=None, attn_lambda=1):
    adv_image = image_tensor.clone().detach()
    adv_image.requires_grad_(True)
    processor = copy.deepcopy(original_processor)
    my__preprocess, my_preprocess = create_preprocess(adv_image)
    processor.image_processor._preprocess = types.MethodType(my__preprocess, processor.image_processor)
    processor.image_processor.preprocess = types.MethodType(my_preprocess, processor.image_processor)

    raw_image_tensor_denorm = denormalize(raw_image_tensor)
    message = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                },
                {"type": "text", "text": get_user_prompt(instruction)},
            ],
        }
    ]
    text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(message)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_prompt_ids = inputs["input_ids"][0].to(main_device)
    response = tokenizer(target_text, add_special_tokens=False).to(main_device)

    input_prompt_ids = inputs["input_ids"][0].to(main_device)
    target_ids = torch.tensor(response["input_ids"]).to(main_device)
    
    input_attention_mask = inputs["attention_mask"][0].to(main_device)
    target_attention_mask = torch.tensor(response["attention_mask"]).to(main_device)

    input_full_ids = torch.cat([input_prompt_ids, target_ids], dim=0)
    attention_mask = torch.cat([input_attention_mask, target_attention_mask], dim=0)
    labels = torch.full_like(input_full_ids, -100)
    labels[-target_ids.shape[0]:] = target_ids
    padding = max_len - input_full_ids.shape[0]
    if padding > 0:
        input_full_ids = torch.cat([input_full_ids, torch.full((padding,), 0, dtype=torch.long, device=main_device)])
        attention_mask = torch.cat([attention_mask, torch.full((padding,), 0, dtype=torch.long, device=main_device)])
        labels = torch.cat([labels, torch.full((padding,), -100, dtype=torch.long, device=main_device)])
    else:
        input_full_ids = input_full_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
    
    inputs["input_ids"] = input_full_ids.unsqueeze(0)
    inputs["attention_mask"] = attention_mask.unsqueeze(0)

    labels = labels.unsqueeze(0)

    hook_logger.reinit()
    adv_image.grad = None  

    outputs = model(**inputs, labels = labels, use_cache=False)
    output_loss = outputs.loss

    if not hook_logger.attns:
        raise ValueError("Attention list is empty. Hook may not have been called correctly.")
    attn_map = hook_logger.finalize()
    resized_mask = resize_mask(mask)
    attn_in_mask = (attn_map * resized_mask).sum() / resized_mask.sum()
    attn_out_mask = (attn_map * (1 - resized_mask)).sum() / (1 - resized_mask).sum()
    attn_loss = attn_out_mask / attn_in_mask
    total_loss = output_loss + attn_lambda * attn_loss
    model.zero_grad()
    total_loss.backward()

    grad_sign = adv_image.grad.sign() * mask
    with torch.no_grad():
        adv_image_denorm = denormalize(adv_image)
        adv_image_denorm = adv_image_denorm - alpha * grad_sign    
        adv_image_denorm = torch.clamp(adv_image_denorm, raw_image_tensor_denorm - epsilon, raw_image_tensor_denorm + epsilon)
        adv_image_denorm = adv_image_denorm.clamp(0.0, 1.0)            
        adv_image = normalize(adv_image_denorm)
        
    adv_image = adv_image.detach().requires_grad_(True)

    return adv_image.detach(), attn_map.detach(), resized_mask.detach(), total_loss.item(), output_loss.item(), attn_loss.item()


def pad_to_square(img, fill_color=(255, 255, 255)):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new('RGB', (max_side, max_side), fill_color)
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def paste_pos(img):
    w, h = img.size
    max_side = max(w, h)
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    return paste_x, paste_y


def init_image_tensor(image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                }
            ],
        }
        ]
    image_inputs, video_inputs = process_vision_info(messages)
    image = image_inputs[0]
    image = np.array(image)
    # print(image.min(), image.max())
    image = rescale(image)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = normalize(image_tensor)
    return image_tensor

def resize_mask(mask):
    # print(mask.shape)
    # copy mask to resized_mask
    patches = mask.unsqueeze(0).repeat(2, 1, 1, 1)
    # print("resized_mask shape:", resized_mask.shape)
    # resized_mask = F.interpolate(resized_mask, size=(resized_height//28, resized_width//28), mode='nearest').to(main_device)
    temporal_patch_size = 2
    merge_size = 2
    patch_size = 14
    grid_t = 1
    channel = 3
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
    
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

    patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
    # print(patches.shape)
    patches = patches.view(-1, patches.shape[1]*(merge_size**2))
    resized_mask = patches.any(dim = 1).float()
    return resized_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_text', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--epsilon', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--layer_index', type=int)
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--attn_lambda', type=float)
    args = parser.parse_args()

    model_name = args.model_name
    data_dir = args.data_dir
    target_text = args.target_text
    save_dir = args.save_dir
    max_len = args.max_len
    steps = args.steps
    epsilon = eval(args.epsilon)
    alpha = eval(args.alpha)
    layer_index = args.layer_index
    attn_lambda = args.attn_lambda

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "examples")):
        os.makedirs(os.path.join(save_dir, "examples"))

    tokenizer, model, processor = init_model(model_name, args.cuda)
    model.eval()
    model.requires_grad_(False)

    main_device = next(model.parameters()).device

    hook_logger = get_hook_logger(model, main_device, layer_index=layer_index)
    with open(os.path.join(data_dir, "data.json")) as f:
        data = json.load(f)
    adv_image = None 
    target_adv_image = None 
    best_adv_image = None
    min_loss = float("inf")

    keys = list(data.keys())
    for step in tqdm(range(steps), desc="Our Attack Steps"):
        key = random.choice(keys)  
        item = data[key]

        filename = item['filename']
        image_path = os.path.join(data_dir, "screenshots", filename)
        image = Image.open(image_path)
        paste_x, paste_y = paste_pos(image)

        image = pad_to_square(image)

        target_box = item['target_box']
        instructions = item['instructions']

        x0 = target_box['x'] + paste_x
        y0 = target_box['y'] + paste_y
        w = target_box['w']
        h = target_box['h']
        instr = random.choice(instructions)
        screenshot_w, screenshot_h = image.size
        original_size = image.size
        image_tensor = init_image_tensor(image).to(main_device)
        W = image_tensor.shape[2]
        H = image_tensor.shape[3]
        x0 = x0 * W // screenshot_w
        y0 = y0 * H // screenshot_h
        w = w * W // screenshot_w
        h = h * H // screenshot_h
        mask = get_original_mask(W, H, x0, y0, w, h).cuda().half().to(main_device)
        if target_adv_image is None:
            adv_image = image_tensor
            outputs = infer_one_image_tensor(adv_image, "Click on the second product in the search result list.", model, processor, main_device)
            print(f"raw output: {outputs}")
            target_h, target_w = get_h_w(mask)
            target_raw_image = extract_target_pixels(adv_image, mask, target_h, target_w)
        else:
            adv_image = image_tensor
            adv_image = paste_target_pixels(adv_image, target_adv_image, mask)

        raw_image = paste_target_pixels(adv_image.clone(), target_raw_image, mask)
        adv_image, attn_map, mask_resized_attn, total_loss, output_loss, attn_loss = one_step(
                model, tokenizer, processor, main_device, image, raw_image, adv_image, instr, target_text, 
                mask, epsilon=epsilon, alpha=alpha, max_len=max_len,
                hook_logger=hook_logger, attn_lambda=attn_lambda)
        
        target_adv_image = extract_target_pixels(adv_image, mask, target_h, target_w)
        
        if total_loss < min_loss:
            min_loss = total_loss
            best_adv_image = target_adv_image.clone().detach()

        tqdm.write(f"Step {step}/{steps-1}, Attention Loss: {attn_loss:.7f}, Output Loss: {output_loss:.7f}, Total Loss: {total_loss:.7f}")
        with open(os.path.join(save_dir, "loss.txt"), 'a') as f:
            f.write(f"Step {step}/{steps-1}, Attention Loss:{attn_loss:.7f}, Output Loss: {output_loss:.7f}, Total Loss: {total_loss:.7f}\n")    

        if step % 50 == 0:
            save_path = os.path.join(save_dir, "examples", f"step_{step}.png")
            save_norm_image(target_adv_image, save_path)
            
            adv_image_save_path = os.path.join(save_dir, "examples", f"step_{step}_adv.png")
            save_norm_image(adv_image, adv_image_save_path)
            outputs = infer_one_image_tensor(adv_image, instr, model, processor, main_device)
            with open(os.path.join(save_dir, "outputs.txt"), 'a') as f:
                f.write(f"Step {step}/{steps-1}, Outputs: {outputs}\n")
            with open(os.path.join(save_dir, "attention.txt"), "a") as f:
                attn_np = attn_map.detach().to(torch.float32).cpu().numpy().reshape(resized_height//28, resized_width//28)
                mask_np = mask_resized_attn.detach().cpu().numpy().astype(int).reshape(resized_height//28, resized_width//28)
                f.write(f"=== Step {step}/{steps-1},Mask(Attention) Map ===\n")
                for i in range(resized_height//28):
                    row = " ".join(f"{mask_np[i, j]}({attn_np[i, j]:.6f})" for j in range(resized_width//28))
                    f.write(row + "\n")
                f.write("\n")
    save_path = os.path.join(save_dir, "examples", "final_adv_image.png")
    save_norm_image(best_adv_image, save_path)
    print(f"Adversarial image saved to {save_path}")
    outputs = infer_one_image_tensor(best_adv_image, "Click on the second product in the search result list.", model, processor, main_device)
    print(f"adversarial output: {outputs}")