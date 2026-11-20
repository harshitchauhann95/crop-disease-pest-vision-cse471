"""
src/models_cnn.py
-----------------
Convolutional Neural Network architectures and training utilities:
- LeNet-5 (Unit II baseline)
- Custom ResNet-18 with skip connections (Unit II)
- Transfer learning wrappers: MobileNetV2, ResNet50, EfficientNet (Unit III)
- Loss functions: Cross-Entropy, Focal Loss, Dice Loss (Unit I & Unit V)
- Layer-wise freezing & differential learning rate helpers (Unit III)
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except Exception:
    tv_models = None
    HAS_TORCHVISION = False


# =====================================================================
# 1. Activation Functions & Loss Functions (Unit I & CO1)
# =====================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance in crop pest/lesion detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for foliar lesion segmentation (Unit V & CO5).
    Dice = 2 * |X n Y| / (|X| + |Y|)
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


# =====================================================================
# 2. LeNet-5 Architecture (Unit II, Practical 4)
# =====================================================================
class LeNet5(nn.Module):
    """
    Classic LeNet-5 adapted for 1-channel or 3-channel input images.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


# =====================================================================
# 3. Residual Block & ResNet-18 From Scratch (Unit II, CO2)
# =====================================================================
class ResidualBlock(nn.Module):
    """
    Standard ResNet residual block with identity shortcut connection.
    F(x) + x addresses vanishing gradients in deep networks.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet18Scratch(nn.Module):
    """
    Custom ResNet-18 architecture with skip connections built from scratch.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# =====================================================================
# 4. Transfer Learning & Fine-Tuning Strategies (Unit III & CO3)
# =====================================================================
def build_transfer_classifier(
    backbone_name: str = "mobilenet_v2",
    num_classes: int = 12,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Instantiates pre-trained backbone and adapts classification head.
    Unit III: Feature Extraction vs Fine-Tuning.
    """
    backbone_name = backbone_name.lower()

    if HAS_TORCHVISION and tv_models is not None:
        weights = "DEFAULT" if pretrained else None
        if backbone_name == "mobilenet_v2":
            model = tv_models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            if freeze_backbone:
                for param in model.features.parameters():
                    param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            return model
        elif backbone_name == "resnet18":
            model = tv_models.resnet18(weights=weights)
            in_features = model.fc.in_features
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            return model

    # Pure PyTorch fallback if torchvision models aren't installed
    model = ResNet18Scratch(in_channels=3, num_classes=num_classes)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model


def unfreeze_top_layers(model: nn.Module, num_layers: int = 3) -> None:
    """
    Unfreezes the topmost N layers of a pre-trained feature extractor for gradual fine-tuning.
    """
    params = list(model.parameters())
    for param in params[-num_layers:]:
        param.requires_grad = True


def get_differential_lr_params(model: nn.Module, base_lr: float = 1e-4, head_lr_mult: float = 10.0) -> List[Dict[str, Any]]:
    """
    Sets up differential learning rates: lower LR for pretrained backbone, higher LR for new head.
    """
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * head_lr_mult}
    ]
