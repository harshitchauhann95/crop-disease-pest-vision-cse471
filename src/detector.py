"""
src/detector.py
---------------
Object detection, localization, and evaluation utilities (Unit IV & CO4):
- Bounding-box coordinate transformations (xyxy, xywh, cxcywh)
- Intersection over Union (IoU) from scratch
- Non-Maximum Suppression (NMS) from scratch
- Mean Average Precision (mAP@0.5 and mAP@0.5:0.95)
- Faster R-CNN transfer learning pipeline for crop pests & foliar symptoms
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    import torchvision.models.detection as tv_detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    HAS_TORCHVISION_DETECTION = True
except Exception:
    tv_detection = None
    FastRCNNPredictor = None
    HAS_TORCHVISION_DETECTION = False


# =====================================================================
# 1. Bounding Box Geometry & Conversions (Unit IV)
# =====================================================================
def box_xyxy_to_xywh(boxes: np.ndarray or torch.Tensor):
    """Converts [x1, y1, x2, y2] to [x1, y1, width, height]."""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)


def box_xyxy_to_cxcywh(boxes: np.ndarray or torch.Tensor):
    """Converts [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=-1)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1)


def box_cxcywh_to_xyxy(boxes: np.ndarray or torch.Tensor):
    """Converts [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=-1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Intersection over Union (IoU) between two sets of boxes [N, 4] and [M, 4].
    Format: [x1, y1, x2, y2].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    return iou


def non_max_suppression_scratch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25
) -> torch.Tensor:
    """
    Classical Non-Maximum Suppression (NMS) algorithm implemented from first principles.
    Returns indices of retained boxes.
    """
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    indices = torch.where(mask)[0]

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(indices[i].item())

        if order.numel() == 1:
            break

        current_box = boxes[i].unsqueeze(0)
        remaining_boxes = boxes[order[1:]]
        ious = compute_iou(current_box, remaining_boxes).squeeze(0)

        remaining_indices = torch.where(ious <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return torch.tensor(keep, dtype=torch.long)


# =====================================================================
# 2. Faster R-CNN Detector (Two-Stage Detector, Unit IV, Practical 8)
# =====================================================================
class SimpleCropDetector(nn.Module):
    """
    Single-stage lightweight convolutional detector fallback.
    Outputs box deltas [B, 4, H, W] and class probabilities [B, C, H, W].
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.box_head = nn.Conv2d(128, 4, 1)
        self.cls_head = nn.Conv2d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        return self.box_head(feat), self.cls_head(feat)


def build_crop_faster_rcnn(
    num_classes: int = 7,
    pretrained_backbone: bool = True
) -> nn.Module:
    """
    Builds Faster R-CNN with ResNet-50 FPN backbone customized for crop pest/disease detection.
    Falls back to SimpleCropDetector if torchvision detection is unavailable.
    """
    if HAS_TORCHVISION_DETECTION and tv_detection is not None:
        try:
            weights = tv_detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
            model = tv_detection.fasterrcnn_resnet50_fpn(weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model
        except Exception:
            pass

    return SimpleCropDetector(num_classes=num_classes)


# =====================================================================
# 3. Detection Evaluation: Precision, Recall & mAP (Unit IV & CO4)
# =====================================================================
def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using standard 11-point interpolation or AUC.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)
