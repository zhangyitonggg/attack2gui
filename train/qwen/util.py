import torch
import os
import json
import numpy as np
from tqdm import tqdm

from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import argparse
import torch.nn.functional as F

from torchvision.transforms import Resize
# import accelerate
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def pad_to_square(img, fill_color=(255, 255, 255)):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new('RGB', (max_side, max_side), fill_color)
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

    
def init_model(model_name, device="auto"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map=device, attn_implementation="eager", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    return tokenizer, model, processor


def infer_one(image, question, model, processor, device="cuda"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_height": 336,
                    "resized_width": 336
                },
                {"type": "text", "text": question},
            ],
        }
        ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # print(model.device)
    for key, value in inputs.items():
        inputs[key] = value.to(device)
        # print(type(value))
        # print(inputs[key].device)
    # Inference: Generation of the output
    # print("model embedding device:", next(model.parameters()).device)
    # print("input_ids device:", inputs["input_ids"].device)
    # print("infer:", inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def infer_one_image_tensor(image_tensor, question, model, processor, device="cuda"):
    norm_denorm = denormalize(image_tensor)         
    norm_denorm = norm_denorm.clamp(0.0, 1.0)     
    tensor = norm_denorm.squeeze(0).cpu()       # [3,H,W]
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()     # H×W×3, uint8
    img = Image.fromarray(array)
    return infer_one(img, question, model, processor, device)


def normalize(tensor):
    mean = OPENAI_CLIP_MEAN
    std = OPENAI_CLIP_STD
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def denormalize(tensor):
    mean = OPENAI_CLIP_MEAN
    std = OPENAI_CLIP_STD
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


def get_std_tensor(tensor):
    std = [0.26862954, 0.26130258, 0.27577711]
    return torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)


def save_norm_image(norm_image, path, image_size=None):
    norm_denorm = denormalize(norm_image)         
    norm_denorm = norm_denorm.clamp(0.0, 1.0)     
    tensor = norm_denorm.squeeze(0).cpu()       # [3,H,W]
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()     # H×W×3, uint8
    img = Image.fromarray(array)
    if image_size is not None:
        img = img.resize(image_size)
    img.save(path, format="PNG")
    return img


def get_original_mask(w_all, h_all, x0, y0, w, h):
    mask = torch.zeros((3, h_all, w_all), dtype=torch.float32)
    mask[:, y0:y0+h, x0:x0+w] = 1.0
    return mask


def extract_target_pixels(image_tensor, mask, h, w):
    nonzero_indices = torch.nonzero(mask[0], as_tuple=False)
    if nonzero_indices.numel() == 0:
        return torch.empty((1, 3, 0, 0), device=image_tensor.device, dtype=image_tensor.dtype)
    y_indices = nonzero_indices[:, 0]
    x_indices = nonzero_indices[:, 1]
    y0, y1 = y_indices.min(), y_indices.max()
    x0, x1 = x_indices.min(), x_indices.max()
    target_pixels = image_tensor[:, :, y0:y0 + h, x0:x0 + w]
    return target_pixels


def paste_target_pixels(image_tensor, target_pixels, mask):
    nonzero_indices = torch.nonzero(mask[0], as_tuple=False)
    if nonzero_indices.numel() == 0:
        return image_tensor
    y_indices = nonzero_indices[:, 0]
    x_indices = nonzero_indices[:, 1]
    y0, y1 = y_indices.min(), y_indices.max()
    x0, x1 = x_indices.min(), x_indices.max()
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    image_tensor[:, :, y0:y0 + target_pixels.shape[2], x0:x0 + target_pixels.shape[3]] = target_pixels
    return image_tensor


def rescale(image: np.ndarray, scale: float = 1.0 / 255.0, dtype=np.float32) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if scale <= 0:
        raise ValueError("Scale must be a positive number")
    return (image * scale).astype(dtype)


def back_rescale(image: np.ndarray, scale: float = 1.0 / 255.0, dtype=np.float32) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if scale <= 0:
        raise ValueError("Scale must be a positive number")
    return (image / scale).astype(dtype)


def get_h_w(mask):
    nonzero_indices = torch.nonzero(mask[0], as_tuple=False)
    y_indices = nonzero_indices[:, 0]
    x_indices = nonzero_indices[:, 1]
    y0, y1 = y_indices.min(), y_indices.max()
    x0, x1 = x_indices.min(), x_indices.max()
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    return h, w