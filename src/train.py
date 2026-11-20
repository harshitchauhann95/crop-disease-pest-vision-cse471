"""
src/train.py
------------
Transfer Learning: Fine-tune MobileNetV2 on PlantVillage dataset.

Downloads the full PlantVillage dataset (~54k images, 38 disease classes)
using kagglehub, applies standard augmentations, fine-tunes a pre-trained
MobileNetV2 backbone, and saves the trained model weights + class list.

Usage:
    python src/train.py

Output:
    configs/crop_classifier.pt   -- model weights
    configs/class_names.txt      -- one class label per line
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

MODEL_PATH = CONFIGS_DIR / "crop_classifier.pt"
CLASS_NAMES_PATH = CONFIGS_DIR / "class_names.txt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 6           # 6 epochs is optimal: 2 for head + 4 for backbone fine-tuning
IMG_SIZE   = 224
LR_HEAD    = 1e-3        # Learning rate for the classification head
LR_BACKBONE = 1e-4       # Learning rate when fine-tuning backbone
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.05
SEED       = 42

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[train] Using device: {DEVICE}")


# ── 1. Download Dataset via kagglehub ─────────────────────────────────────────
def get_plantvillage_path() -> Path:
    """Download PlantVillage from Kaggle and return the full-color image directory."""
    try:
        import kagglehub
        print("[train] Locating PlantVillage full-color dataset ...")
        path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
        dataset_root = Path(path)
        # Prioritize natural full-color RGB images over black-masked segmented ones
        for subdir in ["plantvillage dataset/color",
                       "color",
                       "plantvillage dataset/segmented",
                       "segmented", "."]:
            candidate = dataset_root / subdir
            if candidate.exists() and any(candidate.iterdir()):
                print(f"[train] Dataset root found: {candidate}")
                return candidate
        return dataset_root
    except Exception as e:
        print(f"[train] Error locating dataset: {e}")
        print("[train] Falling back to local data/samples/classification/lab_plantvillage/")
        return REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage"


# ── 2. Data Transforms ─────────────────────────────────────────────────────────
def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── 3. Build MobileNetV2 with custom head ─────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """
    Load pre-trained MobileNetV2 (ImageNet weights) and replace
    the classifier head with a new Linear layer matching num_classes.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── 4. Training Loop ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, 100.0 * correct / total


# ── 5. Main Training Entry ────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    # ── Dataset ────────────────────────────────────────────────────────────────
    data_root = get_plantvillage_path()
    train_tf, val_tf = build_transforms()

    full_dataset = datasets.ImageFolder(str(data_root), transform=train_tf)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"[train] Classes: {num_classes}  |  Total images: {len(full_dataset)}")

    # Save class names for inference
    CLASS_NAMES_PATH.write_text("\n".join(class_names))
    print(f"[train] Class names saved → {CLASS_NAMES_PATH}")

    # ── Splits ─────────────────────────────────────────────────────────────────
    n_total = len(full_dataset)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = int(n_total * TEST_SPLIT)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Apply val transforms to validation/test sets
    val_set.dataset  = datasets.ImageFolder(str(data_root), transform=val_tf)
    test_set.dataset = datasets.ImageFolder(str(data_root), transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[train] Split → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: Train only the head (backbone frozen)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_HEAD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_val_acc = 0.0
    best_epoch   = 0

    print("\n[train] ── PHASE 1: Training head only (backbone frozen) ──")
    for epoch in range(1, NUM_EPOCHS + 1):

        # Unfreeze backbone starting from epoch 3 for full fine-tuning
        if epoch == 3:
            print("\n[train] ── PHASE 2: Unfreezing backbone for fine-tuning ──")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {"params": model.features.parameters(), "lr": LR_BACKBONE},
                {"params": model.classifier.parameters(), "lr": LR_HEAD},
            ])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - 2, eta_min=1e-6
            )

        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        summary = (f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
                   f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
                   f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}% | "
                   f"Time: {elapsed:.0f}s")
        print(summary, flush=True)
        import sys; print(summary, file=sys.stderr, flush=True)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc":     val_acc,
            }, MODEL_PATH)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.1f}%)")

    # ── Final Test Evaluation ──────────────────────────────────────────────────
    print(f"\n[train] Loading best checkpoint (epoch {best_epoch}, Val Acc: {best_val_acc:.1f}%)")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n[train] ══ FINAL TEST ACCURACY: {test_acc:.2f}% ══")
    print(f"[train] Model weights → {MODEL_PATH}")
    print(f"[train] Class names   → {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
