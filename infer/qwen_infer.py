import argparse
from qwen_utils import *
from PIL import Image
import torch
import os
import json

from prompts import get_user_prompt, system_prompt
from torchvision.transforms import Resize, InterpolationMode

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

resized_height = 336
resized_width = 336

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--malicious_image_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--cuda', type=str)

    args = parser.parse_args()    

    data_dir = args.data_dir
    malicious_image_path = args.malicious_image_path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "screenshots")):
        os.makedirs(os.path.join(save_dir, "screenshots"))

    tokenizer, model, processor = init_model(args.model_name, args.cuda)
    device = next(model.parameters()).device

    with open(os.path.join(data_dir, "data.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    malicious_image = Image.open(malicious_image_path).convert('RGB')
    res = data
    for key, item in data.items():
        screenshot_path = os.path.join(data_dir, "screenshots", item["filename"])
        screenshot = Image.open(screenshot_path).convert('RGB')
        paste_x, paste_y = paste_pos(screenshot)
        screenshot = pad_to_square(screenshot)
        x0, y0, w, h = item["target_box"]["x"] + paste_x, item["target_box"]["y"] + paste_y, item["target_box"]["w"], item["target_box"]["h"]
        x0 = int(x0 * resized_width / screenshot.width)
        y0 = int(y0 * resized_height / screenshot.height)
        w = int(w * resized_width / screenshot.width)
        h = int(h * resized_height / screenshot.height)

        resize_op = Resize((resized_height, resized_width), interpolation=InterpolationMode.NEAREST)
        screenshot = resize_op(screenshot)
        screenshot.paste(malicious_image, (x0, y0))
        
        screenshot.save(os.path.join(save_dir, "screenshots", item["filename"]))

        for i, instruction in enumerate(item["instructions"]):
            outputs = infer_one(screenshot, instruction, model, processor, device=device)
            print(f"{key}/{len(data)}-{i}: {outputs}")
            if "outputs" not in res[key]:
                res[key]["outputs"] = []
            res[key]["outputs"].append(outputs)
    
    with open(os.path.join(save_dir, "res.json"), "w") as f:
        json.dump(res, f, indent=4)