from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torch
import os
import json
import random
from tqdm import tqdm

import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import argparse
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode

from util import denormalize, normalize, save_norm_image, get_original_mask, extract_target_pixels, paste_target_pixels, get_h_w
from llava_hook import MaskHookLogger, init_hookmanager, get_hook_logger
from prompts import system_prompt, get_user_prompt

def one_step(model, tokenizer, main_device, 
        image_tensor, raw_image_tensor, instruction, target_text, 
        mask, epsilon=32/255, alpha=1/255, max_len=1024,
        hook_logger=None, attn_lambda=1):
    torch.cuda.empty_cache()
    
    adv_image = image_tensor.clone().detach()
    adv_image.requires_grad = True

    raw_image_tensor_denorm = denormalize(raw_image_tensor)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + get_user_prompt(instruction)

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.system = system_prompt
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_prompt_ids = tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors='pt').to(main_device)

    target_ids = tokenizer(target_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0].to(main_device)

    input_full_ids = torch.cat([input_prompt_ids, target_ids], dim=0)

    labels = torch.full_like(input_full_ids, -100)
    labels[-target_ids.shape[0]:] = target_ids

    padding = max_len - input_full_ids.shape[0]
    if padding > 0:
        input_full_ids = torch.cat([input_full_ids, torch.full((padding,), 0, dtype=torch.long, device=main_device)])
        labels = torch.cat([labels, torch.full((padding,), -100, dtype=torch.long, device=main_device)])
    else:
        input_full_ids = input_full_ids[:max_len]
        labels = labels[:max_len]

    input_ids = input_full_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)

    resize_to_attn = Resize((24, 24), interpolation=InterpolationMode.NEAREST) 
    mask_resized_attn = resize_to_attn(mask[0:1,:,:].unsqueeze(0)).to(main_device)  # shape: (1, 1, 24, 24)

    hook_logger.reinit()
    
    adv_image.grad = None
    outputs = model(input_ids=input_ids, images=adv_image, labels=labels, use_cache=False)
    output_loss = outputs.loss

    if not hook_logger.attns:
        raise ValueError("Attention list is empty. Hook may not have been called correctly.")

    attn_map = hook_logger.finalize().view(1, 1, 24, 24)

    attn_in_mask = (attn_map * mask_resized_attn).sum() / mask_resized_attn.sum()
    attn_out_mask = (attn_map * (1 - mask_resized_attn)).sum() / (1 - mask_resized_attn).sum()
    attn_loss = attn_out_mask / attn_in_mask
    total_loss = output_loss + attn_lambda * attn_loss
    # if only_output_loss:
        # total_loss = output_loss
    # total_loss = attn_loss

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
        
    return adv_image.detach(), attn_map.detach(), mask_resized_attn.detach(), total_loss.item(), output_loss.item(), attn_loss.item()


def init_model(model_path, model_base=None):
    def get_model_name_from_path(model_path):
        model_path = model_path.strip("/")
        model_paths = model_path.split("/")
        if model_paths[-1].startswith('checkpoint-'):
            return model_paths[-2] + "_" + model_paths[-1]
        else:
            return model_paths[-1]
        
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=False 
    )

    return tokenizer, model, image_processor


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

def preprocess_image(path: str) -> torch.Tensor:
    from torchvision.transforms import ToTensor, Resize, Compose 
    img = Image.open(path).convert('RGB')
    img = pad_to_square(img, fill_color=(255,255,255))
    pix = image_processor.preprocess(img, return_tensors='pt')['pixel_values'].half().cuda()  # [1, 3, H, W]

    return pix

def infer_one(image_tensor, instruction, tokenizer, model):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + get_user_prompt(instruction)
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.system = system_prompt
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
        
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=200,
            use_cache=True)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    return outputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_text')
    parser.add_argument('--save_dir')
    parser.add_argument('--max_len')
    parser.add_argument('--steps')
    parser.add_argument('--epsilon')
    parser.add_argument('--alpha')
    parser.add_argument('--screenshot_w')
    parser.add_argument('--screenshot_h')
    parser.add_argument('--layer_index')
    parser.add_argument('--attn_lambda')
    parser.add_argument('--only_output_loss')
    args = parser.parse_args()    

    model_path = args.model_path

    data_dir = args.data_dir
    target_text = args.target_text
    save_dir = args.save_dir
    max_len = args.max_len
    steps = args.steps
    epsilon = eval(args.epsilon)
    alpha = eval(args.alpha)
    W = args.screenshot_w
    H = args.screenshot_h
    layer_index = args.layer_index
    attn_lambda = args.attn_lambda
    only_output_loss = args.only_output_loss

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "examples")):
        os.makedirs(os.path.join(save_dir, "examples"))

    tokenizer, model, image_processor = init_model(model_path)
    model.eval()
    model.requires_grad_(False)
    model.model.vision_tower.requires_grad_(False)
    main_device = next(model.parameters()).device
    hook_logger = get_hook_logger(model, main_device, layer_index=layer_index)
    with open(os.path.join(data_dir, "data.json")) as f:
        data = json.load(f)
    
    adv_image = None 
    target_adv_image = None
    target_raw_image = None
    target_h = target_w = None
    H, W = args.screenshot_h, args.screenshot_w

    keys = list(data.keys())
    for step in tqdm(range(steps), desc="Our Attack Steps"):
        key = random.choice(keys)  
        item = data[key]
        filename = item['filename']
        target_box = item['target_box']
        instructions = item['instructions']
        image_path = os.path.join(data_dir, "screenshots", filename)
        x0 = target_box['x']
        y0 = target_box['y']
        w = target_box['w']
        h = target_box['h']
        instr = random.choice(instructions)
        paste_x, paste_y = paste_pos(Image.open(image_path).convert('RGB'))
        y0 += paste_y
        x0 += paste_x
        H = W = max(H, W)
        mask_orig = get_original_mask(W, H, x0, y0, w, h).cuda().half()  # shape: (3, H, W)
        resize_op = Resize((336, 336), interpolation=InterpolationMode.NEAREST)
        mask_resized = resize_op(mask_orig).to(main_device)   # shape: (3, 336, 336)
        if target_adv_image is None:
            adv_image = preprocess_image(image_path).to(main_device)
            outputs = infer_one(adv_image, "Click on the second product in the search result list.", tokenizer, model)
            print(f"raw output: {outputs}")
            target_h, target_w = get_h_w(mask_resized)
            target_raw_image = extract_target_pixels(adv_image, mask_resized, target_h, target_w)
        else:
            adv_image = preprocess_image(image_path).to(main_device)
            adv_image = paste_target_pixels(adv_image, target_adv_image, mask_resized)
        raw_image = paste_target_pixels(adv_image.clone(), target_raw_image, mask_resized)
        adv_image, attn_map, mask_resized_attn, total_loss, output_loss, attn_loss = one_step(
                model, tokenizer, main_device, adv_image, raw_image, instr, target_text, 
                mask_resized, epsilon=epsilon, alpha=alpha, max_len=max_len,
                hook_logger=hook_logger, attn_lambda=attn_lambda)
        target_adv_image = extract_target_pixels(adv_image, mask_resized, target_h, target_w)
        tqdm.write(f"Step {step}/{steps-1}, Attention Loss: {attn_loss:.7f}, Output Loss: {output_loss:.7f}, Total Loss: {total_loss:.7f}")
        with open(os.path.join(save_dir, "loss.txt"), 'a') as f:
            f.write(f"Step {step}/{steps-1}, Attention Loss:{attn_loss:.7f}, Output Loss: {output_loss:.7f}, Total Loss: {total_loss:.7f}\n")    
        if step % 50 == 0:
            save_path = os.path.join(save_dir, "examples", f"step_{step}.png")
            save_norm_image(target_adv_image, save_path)
            torch.save(target_adv_image, os.path.join(save_dir, "examples", f"step_{step}.pt"))
            outputs = infer_one(adv_image, instr, tokenizer, model)
            with open(os.path.join(save_dir, "outputs.txt"), 'a') as f:
                f.write(f"Step {step}/{steps-1}, Outputs: {outputs}\n")
            with open(os.path.join(save_dir, "attention.txt"), "a") as f:
                attn_np = attn_map.detach().cpu().numpy().reshape(24, 24)
                mask_np = mask_resized_attn.detach().cpu().numpy().astype(int).reshape(24, 24)
                f.write(f"=== Step {step}/{steps-1},Mask(Attention) Map ===\n")
                for i in range(24):
                    row = " ".join(f"{mask_np[i, j]}({attn_np[i, j]:.6f})" for j in range(24))
                    f.write(row + "\n")
                f.write("\n")
    save_path = os.path.join(save_dir, "examples", f"step_{step}.png")
    target_adv_image = extract_target_pixels(adv_image, mask_resized, target_h, target_w)
    save_norm_image(target_adv_image, save_path)
    print(f"Adversarial image saved to {save_path}")
    with open(os.path.join(save_dir, "loss.txt"), 'a') as f:
        f.write(f"Step {step}/{steps-1}, Attention Loss:{attn_loss:.7f}, Output Loss: {output_loss:.7f}, Total Loss: {total_loss:.7f}\n")    
    outputs = infer_one(adv_image, "Click on the second product in the search result list.", tokenizer, model)
    print(f"adversarial output: {outputs}")
