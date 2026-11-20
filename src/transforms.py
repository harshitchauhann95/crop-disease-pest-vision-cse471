"""
src/transforms.py
-----------------
Image preprocessing, normalization, and data augmentation pipelines.
Covers syllabus requirements for Unit I (color spaces, resizing, normalization)
and Unit II (spatial and photometric augmentations for CNN generalization).
Provides pure-Python/PIL/PyTorch fallbacks for universal portability.
"""

from typing import Tuple, Optional, Dict, Any, List
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Compose:
    """Sequential transform compositor."""
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, Image.BILINEAR)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotation:
    def __init__(self, degrees: float = 15.0):
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR)


class ColorJitter:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img


class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            return torch.from_numpy(arr).unsqueeze(0)
        return torch.from_numpy(arr).permute(2, 0, 1)


class Normalize:
    def __init__(self, mean: List[float] = IMAGENET_MEAN, std: List[float] = IMAGENET_STD):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def get_classification_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True
) -> Compose:
    """
    Builds data augmentation & normalization pipeline for leaf disease classification.
    Unit II: Spatial and Photometric Augmentations.
    """
    if is_training:
        return Compose([
            Resize(image_size),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=15.0),
            ColorJitter(brightness=0.2, contrast=0.2),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        return Compose([
            Resize(image_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])


def convert_color_space(image: np.ndarray, mode: str = "RGB") -> np.ndarray:
    """
    Converts image between standard color representations (Unit I: RGB, Grayscale, HSV, LAB).
    """
    pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    if mode.upper() == "GRAYSCALE":
        return np.array(pil_img.convert("L"))
    elif mode.upper() == "HSV":
        return np.array(pil_img.convert("HSV"))
    elif mode.upper() == "RGB":
        return np.array(pil_img.convert("RGB"))
    elif mode.upper() == "LAB":
        try:
            import cv2
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        except Exception:
            # Fallback LAB approximation
            rgb = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            l_channel = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 1.0) * 255.0
            a_channel = np.clip(0.5 + 0.5 * (r - g), 0.0, 1.0) * 255.0
            b_channel = np.clip(0.5 + 0.5 * (g - b), 0.0, 1.0) * 255.0
            return np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported color mode: {mode}")


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Inverts ImageNet normalization for visualization.
    Input: [3, H, W] tensor. Output: [H, W, 3] uint8 numpy array in range [0, 255].
    """
    img = tensor.clone().detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr
