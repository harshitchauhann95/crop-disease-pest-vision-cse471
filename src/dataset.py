"""
src/dataset.py
--------------
PyTorch Datasets & DataLoaders for:
1. Plant Leaf Disease Classification (PlantVillage & PlantDoc domain adaptation)
2. Crop Pest & Foliar Lesion Bounding-Box Detection (Pascal VOC / YOLO)
3. Foliar Lesion Semantic Segmentation (Image-Mask pairs)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


# =====================================================================
# 1. Leaf Disease Classification Dataset (CO1, CO2, CO3)
# =====================================================================
class PlantDiseaseClassificationDataset(Dataset):
    """
    Dataset for leaf disease image classification.
    Supports lab conditions (PlantVillage) and field conditions (PlantDoc).
    """
    def __init__(
        self,
        root_dir: str or Path,
        classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        domain: str = "all"  # 'lab', 'field', or 'all'
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.domain = domain
        self.samples: List[Tuple[Path, int]] = []

        # Graceful fallback: check if directory exists, otherwise use samples
        if not self.root_dir.exists():
            fallback_dir = SAMPLES_DIR / "classification" / ("lab_plantvillage" if domain == "lab" else "field_plantdoc")
            if fallback_dir.exists():
                self.root_dir = fallback_dir

        # Identify class directories
        if self.root_dir.exists():
            class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if classes is not None:
                self.classes = sorted(list(classes))
            else:
                self.classes = sorted([d.name for d in class_dirs])
            
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = self.root_dir / cls_name
                if cls_folder.exists():
                    for img_file in cls_folder.glob("*.*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            self.samples.append((img_file, self.class_to_idx[cls_name]))
        else:
            self.classes = classes or []
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # Basic tensor conversion [3, H, W] in [0, 1]
            arr = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(arr).permute(2, 0, 1)

        return image, label


# =====================================================================
# 2. Crop Pest & Foliar Lesion Detection Dataset (CO4)
# =====================================================================
class CropDetectionDataset(Dataset):
    """
    Dataset for crop pest and foliar lesion object detection.
    Parses Pascal VOC XML annotations and maps bounding boxes to tensors.
    """
    def __init__(
        self,
        images_dir: str or Path,
        annotations_dir: str or Path,
        classes: List[str],
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (416, 416)
    ):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.classes = classes
        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}  # 0 is reserved for background
        self.transform = transform
        self.target_size = target_size

        # Fallback to samples if paths do not exist
        if not self.images_dir.exists():
            self.images_dir = SAMPLES_DIR / "detection" / "images"
            self.annotations_dir = SAMPLES_DIR / "detection" / "annotations_voc"

        self.image_files: List[Path] = []
        if self.images_dir.exists():
            for f in sorted(self.images_dir.glob("*.*")):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    xml_path = self.annotations_dir / f"{f.stem}.xml"
                    if xml_path.exists():
                        self.image_files.append(f)

    def __len__(self) -> int:
        return len(self.image_files)

    def _parse_voc_xml(self, xml_path: Path, orig_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        orig_w, orig_h = orig_size
        scale_x = self.target_size[0] / orig_w
        scale_y = self.target_size[1] / orig_h

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in self.class_to_idx:
                continue
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) * scale_x
            ymin = float(bndbox.find("ymin").text) * scale_y
            xmax = float(bndbox.find("xmax").text) * scale_x
            ymax = float(bndbox.find("ymax").text) * scale_y

            # Clamp bounding boxes
            xmin = max(0.0, min(float(self.target_size[0] - 1), xmin))
            ymin = max(0.0, min(float(self.target_size[1] - 1), ymin))
            xmax = max(xmin + 1.0, min(float(self.target_size[0]), xmax))
            ymax = max(ymin + 1.0, min(float(self.target_size[1]), ymax))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return boxes, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize image to target dimensions
        image_resized = image.resize(self.target_size, Image.BILINEAR)
        img_arr = np.array(image_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        xml_path = self.annotations_dir / f"{img_path.stem}.xml"
        boxes, labels = self._parse_voc_xml(xml_path, (orig_w, orig_h))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return img_tensor, target


def detection_collate_fn(batch):
    """Custom collate function for object detection batches with varying box counts."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# =====================================================================
# 3. Foliar Lesion Semantic Segmentation Dataset (CO5)
# =====================================================================
class CropSegmentationDataset(Dataset):
    """
    Dataset for leaf lesion semantic segmentation.
    Pairs leaf images with ground-truth binary or multi-class lesion masks.
    """
    def __init__(
        self,
        images_dir: str or Path,
        masks_dir: str or Path,
        target_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        self.transform = transform

        if not self.images_dir.exists():
            self.images_dir = SAMPLES_DIR / "segmentation" / "images"
            self.masks_dir = SAMPLES_DIR / "segmentation" / "masks"

        self.paired_files: List[Tuple[Path, Path]] = []
        if self.images_dir.exists():
            for img_file in sorted(self.images_dir.glob("*.*")):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    mask_file = self.masks_dir / f"{img_file.stem}.png"
                    if mask_file.exists():
                        self.paired_files.append((img_file, mask_file))

    def __len__(self) -> int:
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.paired_files[idx]
        image = Image.open(img_path).convert("RGB").resize(self.target_size, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.target_size, Image.NEAREST)

        img_arr = np.array(image, dtype=np.float32) / 255.0
        mask_arr = (np.array(mask, dtype=np.float32) > 128).astype(np.float32)  # Binary 0/1

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)      # [3, H, W]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)        # [1, H, W]

        return img_tensor, mask_tensor
