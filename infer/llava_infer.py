import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torch
import os
import json
import cv2
import numpy as np

from prompts import get_user_prompt, system_prompt
from torchvision.transforms import Resize, InterpolationMode

from util import denormalize, normalize, save_norm_image, get_original_mask, extract_target_pixels, paste_target_pixels, load_image_to_tensor
from llava_hook import MaskHookLogger, init_hookmanager, get_hook_logger

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def pad_to_square(img, fill_color=(255, 255, 255)):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new('RGB', (max_side, max_side), fill_color)
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def preprocess_image(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    img = pad_to_square(img, fill_color=(255,255,255))
    pix = image_processor.preprocess(img, return_tensors='pt')['pixel_values'].to(torch.float16).to(device)  # [1, 3, H, W]
    return pix


def paste_pos(img):
    w, h = img.size
    max_side = max(w, h)
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    return paste_x, paste_y


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    

def init_model(model_path, model_base=None):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    return tokenizer, model, image_processor


def infer_one(image_tensor, instruction, tokenizer, model, mask_resized):
    resize_to_attn = Resize((24, 24), interpolation=InterpolationMode.NEAREST) 
    mask_resized_attn = resize_to_attn(mask_resized[0:1,:,:].unsqueeze(0)).to(device)  # shape: (1, 1, 24, 24)
    hook_logger.reinit()
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
            # temperature=0.2,
            # top_p=0.7,
            num_beams=1,
            max_new_tokens=100,
            use_cache=True)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()

    attn_map = hook_logger.finalize().view(1, 1, 24, 24)
    attn_in_mask = (attn_map * mask_resized_attn).sum() / mask_resized_attn.sum()
    attn_out_mask = (attn_map * (1 - mask_resized_attn)).sum() / (1 - mask_resized_attn).sum()
    
    return outputs, attn_map, attn_in_mask, attn_out_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--malicious_image_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--screenshot_w', type=int)
    parser.add_argument('--screenshot_h', type=int)
    parser.add_argument('--layer_index', type=int)
    args = parser.parse_args()    
    model_path = args.model_path
    data_dir = args.data_dir
    malicious_image_path = args.malicious_image_path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "screenshots")):
        os.makedirs(os.path.join(save_dir, "screenshots"))
    if not os.path.exists(os.path.join(save_dir, "heatmaps")):
        os.makedirs(os.path.join(save_dir, "heatmaps"))
    W = args.screenshot_w
    H = args.screenshot_h
    layer_index = args.layer_index

    with open(os.path.join(save_dir, "args.txt"), 'w') as f:
        f.write(f"model_path: {model_path}\n")
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"malicious_image_path: {malicious_image_path}\n")
        f.write(f"save_dir: {save_dir}\n")
        f.write(f"screenshot_w: {W}\n")
        f.write(f"screenshot_h: {H}\n")

    tokenizer, model, image_processor = init_model(model_path)
    device = next(model.parameters()).device

    hook_logger = get_hook_logger(model, device, layer_index=layer_index)

    with open(os.path.join(data_dir, "data.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    malicious_image_tensor = load_image_to_tensor(malicious_image_path).to(device)

    res = data
    for key, item in data.items():
        screenshot_path = os.path.join(data_dir, "screenshots", item["filename"])

        x0, y0, w, h = item["target_box"]["x"], item["target_box"]["y"], item["target_box"]["w"], item["target_box"]["h"]
        paste_x, paste_y = paste_pos(Image.open(screenshot_path).convert('RGB'))
        y0 += paste_y
        x0 += paste_x
        H = W = max(H, W)
        mask_orig = get_original_mask(W, H, x0, y0, w, h).cuda().half()  # shape: (3, H, W)
        resize_op = Resize((336, 336), interpolation=InterpolationMode.NEAREST)
        mask_resized = resize_op(mask_orig).to(device)   # shape: (3, 336, 336)

        screenshot_tensor = preprocess_image(screenshot_path)

        screenshot_tensor = paste_target_pixels(screenshot_tensor, malicious_image_tensor, mask_resized)

        save_norm_image(screenshot_tensor, os.path.join(save_dir, "screenshots", item["filename"]))

        for i, instruction in enumerate(item["instructions"]):
            outputs, attn_map, attn_in_mask, attn_out_mask = infer_one(screenshot_tensor, instruction, tokenizer, model, mask_resized)
            print(f"{key}/{len(data)}-{i}: {outputs}")
            print(f"attn_in_mask: {attn_in_mask}, attn_out_mask: {attn_out_mask}")
            attn_map_np = attn_map.squeeze().cpu().numpy()
            attn_min = attn_map_np.min()
            attn_max = attn_map_np.max()
            if attn_max - attn_min < 1e-6:
                attn_map_np_norm = np.zeros_like(attn_map_np)
            else:
                q = np.quantile(attn_map_np, 0.96)
                attn_map_np_norm = np.clip((attn_map_np - attn_min) / (q - attn_min + 1e-6), 0, 1)
                attn_map_np_norm = attn_map_np_norm ** 0.5   
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_np_norm), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            image = Image.open(screenshot_path).convert("RGB")
            image = pad_to_square(image)
            noise = torch.randint(-32, 33, mask_orig.shape).cuda()  # Generate random noise in the range [-32, 32]
            mask_orig_with_noise = mask_orig.clone()  # Create a copy of mask_orig
            mask_orig_with_noise[mask_orig == 1] += noise[mask_orig == 1]  # Add noise only to the regions where mask_orig is 1
            image += mask_orig_with_noise.permute(1, 2, 0).cpu().numpy()  # Add noise to the image
            from torchvision import transforms

            transform = transforms.ToTensor()
            temp_screenshot_tensor = transform(image) 
            screenshot_np = temp_screenshot_tensor.permute(1, 2, 0).cpu().numpy()

            scr_min = screenshot_np.min()
            scr_max = screenshot_np.max()
            if scr_max - scr_min < 1e-6:
                screenshot_np_norm = np.zeros_like(screenshot_np)
            else:
                screenshot_np_norm = (screenshot_np - scr_min) / (scr_max - scr_min)
            heatmap_resized = cv2.resize(heatmap, (screenshot_np.shape[1], screenshot_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            alpha = 0.4 
            cam_image = alpha * heatmap_resized + (1 - alpha) * screenshot_np_norm
            cam_image = np.clip(cam_image, 0, 1)
            heatmap_path = os.path.join(save_dir, "heatmaps", f"{key}_{i}_heatmap.png")
            cv2.imwrite(heatmap_path, np.uint8(255 * cam_image))
            print(f"Heatmap saved to {heatmap_path}")
            if "outputs" not in res[key]:
                res[key]["outputs"] = []
            res[key]["outputs"].append(outputs)
            if "attn_in_mask" not in res[key]:
                res[key]["attn_in_mask"] = []
            res[key]["attn_in_mask"].append(attn_in_mask.item())
            if "attn_out_mask" not in res[key]:
                res[key]["attn_out_mask"] = []
            res[key]["attn_out_mask"].append(attn_out_mask.item())

    with open(os.path.join(save_dir, "res.json"), "w") as f:
        json.dump(res, f, indent=4)