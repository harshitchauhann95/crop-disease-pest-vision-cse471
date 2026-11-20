"""
src/segmenter.py
----------------
Foliar lesion semantic segmentation and disease severity scoring (Unit V & CO5):
- U-Net architecture with skip connections built from scratch
- Metrics: Sørensen-Dice coefficient, IoU (Jaccard Index), Pixel Accuracy
- Crop disease severity estimation (% infected leaf surface area)
- Alpha-blended segmentation mask visualization
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# =====================================================================
# 1. U-Net Architecture From Scratch (Unit V & Practical 9)
# =====================================================================
class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Classic U-Net architecture: contracting path (encoder), bottleneck,
    and expanding path (decoder) with cross-layer skip connections.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, features: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        curr_in = in_channels
        for feat in features:
            self.downs.append(DoubleConv(curr_in, feat))
            curr_in = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Expanding Path)
        for feat in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat * 2, feat))

        # Final projection to class logits
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling path with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # Handle odd spatial dimensions if needed
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


# =====================================================================
# 2. Segmentation Evaluation Metrics (Unit V & CO5)
# =====================================================================
def compute_segmentation_metrics(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes Sørensen-Dice, Intersection-over-Union (IoU), and Pixel Accuracy.
    """
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float().view(-1)
    target_bin = (target_mask > 0.5).float().view(-1)

    intersection = (pred_bin * target_bin).sum().item()
    total_pred = pred_bin.sum().item()
    total_target = target_bin.sum().item()
    union = total_pred + total_target - intersection

    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    iou = (intersection + eps) / (union + eps)
    correct_pixels = (pred_bin == target_bin).float().sum().item()
    pixel_acc = correct_pixels / max(1, pred_bin.numel())

    return {
        "dice_score": float(dice),
        "iou_jaccard": float(iou),
        "pixel_accuracy": float(pixel_acc)
    }


# =====================================================================
# 3. Severity Analysis & Visual Overlay (SIH26131 Stage 3)
# =====================================================================
def calculate_disease_severity(
    mask_binary: np.ndarray,
    total_leaf_pixels: Optional[int] = None
) -> Tuple[float, str]:
    """
    Calculates percentage of infected surface area and categorizes severity:
    - Mild: < 10%
    - Moderate: 10% - 30%
    - Severe: > 30%
    """
    infected_pixels = int(np.sum(mask_binary > 0))
    total_pixels = total_leaf_pixels if total_leaf_pixels is not None else mask_binary.size

    severity_pct = (infected_pixels / max(1, total_pixels)) * 100.0

    if severity_pct < 10.0:
        stage = "Mild (Stage 1)"
    elif severity_pct <= 30.0:
        stage = "Moderate (Stage 2)"
    else:
        stage = "Severe (Stage 3 - High Risk)"

    return round(severity_pct, 2), stage


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Creates an alpha-blended visualization overlay of lesion mask on RGB image.
    """
    overlay = image.copy()
    binary_mask = mask > 0

    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask] = color

    blended = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    overlay[binary_mask] = blended[binary_mask]
    return overlay
