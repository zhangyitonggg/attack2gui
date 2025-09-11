import torch 
from PIL import Image
import numpy as np

def normalize(tensor):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def denormalize(tensor):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


def get_std_tensor(tensor):
    std = [0.26862954, 0.26130258, 0.27577711]
    return torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)

def save_norm_image(norm_image, path):
    norm_denorm = denormalize(norm_image)         
    norm_denorm = norm_denorm.clamp(0.0, 1.0)     
    tensor = norm_denorm.squeeze(0).cpu()       # [3,H,W]
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()     # H×W×3, uint8
    img = Image.fromarray(array)
    img.save(path, format="PNG")


def load_image_to_tensor(path):
    img = Image.open(path).convert("RGB")
    array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # [3, H, W]
    tensor = array / 255.0
    tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
    tensor = normalize(tensor)   
    return tensor


def get_original_mask(w_all, h_all, x0, y0, w, h):
    mask = torch.zeros((3, h_all, w_all), dtype=torch.float32)
    mask[:, y0:y0+h, x0:x0+w] = 1.0
    return mask


def extract_target_pixels(image_tensor, mask):
    nonzero_indices = torch.nonzero(mask[0], as_tuple=False)
    if nonzero_indices.numel() == 0:
        return torch.empty((1, 3, 0, 0), device=image_tensor.device, dtype=image_tensor.dtype)
    y_indices = nonzero_indices[:, 0]
    x_indices = nonzero_indices[:, 1]
    y0, y1 = y_indices.min(), y_indices.max()
    x0, x1 = x_indices.min(), x_indices.max()
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    target_pixels = image_tensor[:, :, y0:y1 + 1, x0:x1 + 1]
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
