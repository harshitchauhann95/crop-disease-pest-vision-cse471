"""
src/vlm_and_gradcam.py
----------------------
Explainable AI (XAI) and Vision-Language Models (Unit VI & CO6):
- Grad-CAM (Gradient-weighted Class Activation Mapping) from scratch
- Vanilla Gradient Saliency Maps
- Side-by-side comparison of correct vs misclassified samples
- Zero-Shot CLIP-style text prompt matching for disease diagnosis
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# =====================================================================
# 1. Grad-CAM Implementation From Scratch (Unit VI & Practical 10)
# =====================================================================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) from first principles.
    Hooks into target convolutional layer, extracts feature maps and gradients,
    and produces visual attribution heatmaps.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self._hook_handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient)
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for specified target class or model prediction.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=-1).item()

        # Backward pass for target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global Average Pooling of gradients across spatial dimensions (W, H)
        alpha = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Linear combination of feature maps weighted by alpha
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        # Resize CAM to match input tensor spatial resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays colored Grad-CAM heatmap onto RGB image.
    Uses custom jet colormap without requiring cv2 if necessary.
    """
    # Colorize heatmap: Red (high importance) to Blue (low importance)
    h, w, _ = image.shape
    resized_heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0

    # Synthetic Jet color mapping: R, G, B channels
    r = np.clip(1.5 - np.abs(resized_heatmap * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(resized_heatmap * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(resized_heatmap * 4.0 - 1.0), 0.0, 1.0)
    colored_cam = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    blended = (image.astype(np.float32) * (1.0 - alpha) + colored_cam.astype(np.float32) * alpha).astype(np.uint8)
    return blended


# =====================================================================
# 2. Vanilla Saliency Map (Unit VI & Practical 10)
# =====================================================================
def compute_saliency_map(model: nn.Module, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
    """
    Computes vanilla gradient saliency map: max gradient across color channels.
    """
    model.eval()
    tensor_copy = input_tensor.clone().detach().requires_grad_(True)
    output = model(tensor_copy)

    if target_class is None:
        target_class = output.argmax(dim=-1).item()

    score = output[0, target_class]
    score.backward()

    # Saliency is the absolute maximum gradient across RGB channels
    saliency, _ = torch.max(tensor_copy.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 1e-7:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency


# =====================================================================
# 3. Vision-Language Prompt Matching (Unit VI & CO6)
# =====================================================================
class CropVisionLanguageMatcher:
    """
    Lightweight zero-shot text-image matching module inspired by CLIP.
    Encodes agricultural symptoms as descriptive natural language prompts
    and computes cosine similarity with feature embeddings.
    """
    def __init__(self, class_descriptions: Optional[Dict[str, str]] = None):
        self.descriptions = class_descriptions or {
            "Tomato___Early_Blight": "A tomato leaf displaying dark brown concentric circular spots surrounded by a chlorotic yellow halo.",
            "Tomato___Late_Blight": "A tomato leaf exhibiting large water-soaked irregular dark grayish lesions.",
            "Tomato___Healthy": "A vibrant green healthy tomato leaf with smooth surface and intact veins.",
            "Rice___Leaf_Blast": "A rice leaf with spindle-shaped necrotic lesions with diamond brown borders.",
            "Apple___Apple_Scab": "An apple leaf with olive-green to black velvety spots causing distortion.",
            "Potato___Early_Blight": "A potato leaf showing target-board necrotic lesions with yellow margins."
        }

    def zero_shot_match(self, feature_vector: np.ndarray, text_embeddings: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Computes cosine similarity between image feature embedding and candidate class text embeddings.
        """
        feat_norm = feature_vector / (np.linalg.norm(feature_vector) + 1e-7)
        scores = []
        for name, text_emb in text_embeddings.items():
            t_norm = text_emb / (np.linalg.norm(text_emb) + 1e-7)
            sim = float(np.dot(feat_norm, t_norm))
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
