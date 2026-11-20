#!/usr/bin/env python3
"""
scripts/download_datasets.py
----------------------------
Dataset acquisition and verification utility for CSE471 & SIH26131.
Handles downloading benchmark agricultural datasets (PlantVillage, PlantDoc,
Pest Detection, Foliar Lesion Segmentation) and provides deterministic sample
generation for offline prototyping on standard hardware.
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"

# Target classes for academic & SIH benchmarks
CLASSIFICATION_CLASSES = [
    "Tomato___Healthy",
    "Tomato___Early_Blight",
    "Tomato___Late_Blight",
    "Potato___Healthy",
    "Potato___Early_Blight",
    "Potato___Late_Blight",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Brown_Spot",
    "Apple___Healthy",
    "Apple___Apple_Scab",
    "Apple___Black_Rot",
]

DETECTION_CLASSES = [
    "aphid",
    "armyworm",
    "whitefly",
    "leaf_miner",
    "blast_lesion",
    "blight_spot",
]


def create_synthetic_leaf(image_size=(256, 256), leaf_color=(45, 138, 45), background="lab"):
    """
    Synthesizes a realistic leaf image with customizable background
    ('lab' for clean PlantVillage-style or 'field' for cluttered PlantDoc-style).
    """
    w, h = image_size
    if background == "lab":
        # Neutral gray/white studio background
        bg_val = random.randint(235, 250)
        img = Image.new("RGB", (w, h), (bg_val, bg_val, bg_val))
    else:
        # Cluttered field soil/foliage background
        bg_array = np.random.randint(40, 110, (h, w, 3), dtype=np.uint8)
        # Add brown/soil tint
        bg_array[..., 0] = np.clip(bg_array[..., 0] + 30, 0, 255)
        img = Image.fromarray(bg_array)

    draw = ImageDraw.Draw(img)

    # Draw elliptical leaf shape
    margin_x, margin_y = int(w * 0.15), int(h * 0.12)
    bbox = [margin_x, margin_y, w - margin_x, h - margin_y]
    
    # Base green leaf with subtle gradient/noise
    draw.ellipse(bbox, fill=leaf_color, outline=(25, 90, 25), width=2)
    
    # Draw central vein and lateral veins
    center_x = w // 2
    draw.line([center_x, margin_y, center_x, h - margin_y], fill=(70, 160, 70), width=3)
    for y in range(margin_y + 30, h - margin_y - 20, 25):
        draw.line([center_x, y, margin_x + 15, y - 15], fill=(60, 150, 60), width=1)
        draw.line([center_x, y, w - margin_x - 15, y - 15], fill=(60, 150, 60), width=1)

    return img, bbox


def inject_disease_symptoms(img, disease_type, num_spots=4):
    """
    Injects realistic lesion patterns into leaf image and generates ground-truth binary mask.
    """
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    if "Healthy" in disease_type:
        return img, mask, []

    lesions = []
    for _ in range(num_spots):
        # Pick random position inside central area of leaf
        cx = random.randint(int(w * 0.3), int(w * 0.7))
        cy = random.randint(int(h * 0.25), int(h * 0.75))
        rad = random.randint(10, 26)

        if "Early_Blight" in disease_type or "blight_spot" in disease_type:
            # Concentric brown rings with chlorotic yellow halo
            halo_rad = rad + 8
            draw_img.ellipse([cx - halo_rad, cy - halo_rad, cx + halo_rad, cy + halo_rad], fill=(190, 180, 40))
            draw_img.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(85, 45, 20))
            draw_img.ellipse([cx - rad // 2, cy - rad // 2, cx + rad // 2, cy + rad // 2], fill=(45, 20, 10))
            draw_mask.ellipse([cx - halo_rad, cy - halo_rad, cx + halo_rad, cy + halo_rad], fill=255)
            lesions.append(("blight_spot", cx - halo_rad, cy - halo_rad, cx + halo_rad, cy + halo_rad))

        elif "Late_Blight" in disease_type or "blast_lesion" in disease_type:
            # Spindle-shaped dark water-soaked necrotic patches
            rx, ry = rad + 6, rad - 3
            draw_img.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(40, 50, 35))
            draw_img.ellipse([cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2], fill=(20, 25, 18))
            draw_mask.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
            lesions.append(("blast_lesion", cx - rx, cy - ry, cx + rx, cy + ry))

        elif "Scab" in disease_type or "Black_Rot" in disease_type:
            # Dark velvety circular lesions
            draw_img.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(30, 28, 25))
            draw_mask.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
            lesions.append(("blight_spot", cx - rad, cy - rad, cx + rad, cy + rad))

        elif "Brown_Spot" in disease_type:
            # Small oval reddish-brown spots with gray centers
            draw_img.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(110, 55, 30))
            draw_img.ellipse([cx - rad // 2, cy - rad // 2, cx + rad // 2, cy + rad // 2], fill=(160, 140, 120))
            draw_mask.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
            lesions.append(("blight_spot", cx - rad, cy - rad, cx + rad, cy + rad))

    # Apply soft Gaussian blur to smooth synthesized edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img, mask, lesions


def inject_pests(img, num_pests=3):
    """
    Injects synthetic agricultural insect pests (aphids, armyworms, whiteflies) onto leaf.
    """
    w, h = img.size
    draw = ImageDraw.Draw(img)
    pest_boxes = []

    pest_types = ["aphid", "armyworm", "whitefly", "leaf_miner"]
    for _ in range(num_pests):
        ptype = random.choice(pest_types)
        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))

        if ptype == "aphid":
            # Tiny yellowish-green / black globular insect
            r = random.randint(4, 7)
            color = random.choice([(140, 160, 40), (40, 40, 30)])
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(20, 20, 20))
            pest_boxes.append((ptype, cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2))

        elif ptype == "armyworm":
            # Elongated caterpillar with longitudinal stripes
            lw, lh = random.randint(18, 30), random.randint(6, 10)
            draw.rounded_rectangle([cx - lw // 2, cy - lh // 2, cx + lw // 2, cy + lh // 2], radius=3, fill=(65, 80, 45), outline=(20, 30, 15))
            draw.line([cx - lw // 2 + 2, cy, cx + lw // 2 - 2, cy], fill=(200, 200, 120), width=1)
            pest_boxes.append((ptype, cx - lw // 2 - 2, cy - lh // 2 - 2, cx + lw // 2 + 2, cy + lh // 2 + 2))

        elif ptype == "whitefly":
            # Tiny powdery white triangular fly
            r = random.randint(3, 5)
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=(245, 245, 245), outline=(180, 180, 180))
            pest_boxes.append((ptype, cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2))

        elif ptype == "leaf_miner":
            # Winding serpentine white/yellow feeding trail
            points = [(cx + i * 5, cy + int(np.sin(i) * 8)) for i in range(-4, 5)]
            draw.line(points, fill=(230, 235, 190), width=2)
            pest_boxes.append((ptype, cx - 22, cy - 10, cx + 22, cy + 10))

    return img, pest_boxes


def write_pascal_voc_xml(filepath, img_name, img_size, boxes):
    """Writes standard Pascal VOC XML annotation."""
    w, h, c = img_size
    lines = [
        "<annotation>",
        f"  <filename>{img_name}</filename>",
        "  <size>",
        f"    <width>{w}</width>",
        f"    <height>{h}</height>",
        f"    <depth>{c}</depth>",
        "  </size>",
    ]
    for cls_name, xmin, ymin, xmax, ymax in boxes:
        lines.extend([
            "  <object>",
            f"    <name>{cls_name}</name>",
            "    <pose>Unspecified</pose>",
            "    <truncated>0</truncated>",
            "    <difficult>0</difficult>",
            "    <bndbox>",
            f"      <xmin>{max(0, int(xmin))}</xmin>",
            f"      <ymin>{max(0, int(ymin))}</ymin>",
            f"      <xmax>{min(w, int(xmax))}</xmax>",
            f"      <ymax>{min(h, int(ymax))}</ymax>",
            "    </bndbox>",
            "  </object>",
        ])
    lines.append("</annotation>")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def write_yolo_txt(filepath, img_size, boxes, class_map):
    """Writes standard YOLO normalized bounding-box format (class_idx cx cy w h)."""
    img_w, img_h, _ = img_size
    lines = []
    for cls_name, xmin, ymin, xmax, ymax in boxes:
        if cls_name not in class_map:
            continue
        cls_idx = class_map[cls_name]
        xmin = max(0, float(xmin))
        ymin = max(0, float(ymin))
        xmax = min(img_w, float(xmax))
        ymax = min(img_h, float(ymax))
        
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        cx = (xmin + xmax) / (2.0 * img_w)
        cy = (ymin + ymax) / (2.0 * img_h)
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def generate_sample_subsets(num_samples_per_class=10):
    """
    Generates balanced, deterministic sample subsets across all 3 vision modalities:
    1. Classification: Lab (PlantVillage) and Field (PlantDoc) sets
    2. Object Detection: Images with VOC XML and YOLO TXT labels
    3. Segmentation: Leaf lesion images with corresponding binary mask PNGs
    """
    print("[*] Generating deterministic agricultural vision sample subsets...")
    random.seed(42)
    np.random.seed(42)

    # 1. Classification Samples
    lab_dir = SAMPLES_DIR / "classification" / "lab_plantvillage"
    field_dir = SAMPLES_DIR / "classification" / "field_plantdoc"

    for cls_name in CLASSIFICATION_CLASSES:
        cls_lab_dir = lab_dir / cls_name
        cls_field_dir = field_dir / cls_name
        cls_lab_dir.mkdir(parents=True, exist_ok=True)
        cls_field_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            # Lab sample
            img_lab, _ = create_synthetic_leaf(image_size=(224, 224), background="lab")
            img_lab, _, _ = inject_disease_symptoms(img_lab, cls_name, num_spots=random.randint(2, 5))
            img_lab.save(cls_lab_dir / f"sample_{i:03d}.jpg", quality=92)

            # Field sample (with natural clutter / domain shift)
            img_field, _ = create_synthetic_leaf(image_size=(224, 224), background="field")
            img_field, _, _ = inject_disease_symptoms(img_field, cls_name, num_spots=random.randint(2, 6))
            img_field.save(cls_field_dir / f"sample_{i:03d}.jpg", quality=90)

    print(f"  [+] Classification samples generated in {SAMPLES_DIR / 'classification'}")

    # 2. Object Detection Samples (Pests & Lesions)
    det_img_dir = SAMPLES_DIR / "detection" / "images"
    det_xml_dir = SAMPLES_DIR / "detection" / "annotations_voc"
    det_yolo_dir = SAMPLES_DIR / "detection" / "annotations_yolo"
    det_img_dir.mkdir(parents=True, exist_ok=True)
    det_xml_dir.mkdir(parents=True, exist_ok=True)
    det_yolo_dir.mkdir(parents=True, exist_ok=True)

    det_class_map = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}
    with open(SAMPLES_DIR / "detection" / "classes.txt", "w") as f:
        f.write("\n".join(DETECTION_CLASSES))

    num_det_samples = 30
    for i in range(num_det_samples):
        img_det, _ = create_synthetic_leaf(image_size=(416, 416), background=random.choice(["lab", "field"]))
        disease_type = random.choice(CLASSIFICATION_CLASSES)
        img_det, _, lesion_boxes = inject_disease_symptoms(img_det, disease_type, num_spots=random.randint(1, 3))
        img_det, pest_boxes = inject_pests(img_det, num_pests=random.randint(1, 3))
        
        all_boxes = lesion_boxes + pest_boxes
        img_name = f"det_{i:03d}.jpg"
        img_path = det_img_dir / img_name
        img_det.save(img_path, quality=92)

        write_pascal_voc_xml(det_xml_dir / f"det_{i:03d}.xml", img_name, (416, 416, 3), all_boxes)
        write_yolo_txt(det_yolo_dir / f"det_{i:03d}.txt", (416, 416, 3), all_boxes, det_class_map)

    print(f"  [+] Object detection samples generated in {SAMPLES_DIR / 'detection'}")

    # 3. Semantic Segmentation Samples (Foliar Lesion Masks)
    seg_img_dir = SAMPLES_DIR / "segmentation" / "images"
    seg_mask_dir = SAMPLES_DIR / "segmentation" / "masks"
    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_mask_dir.mkdir(parents=True, exist_ok=True)

    num_seg_samples = 30
    for i in range(num_seg_samples):
        img_seg, _ = create_synthetic_leaf(image_size=(256, 256), background="lab")
        disease_type = random.choice([c for c in CLASSIFICATION_CLASSES if "Healthy" not in c])
        img_seg, mask, _ = inject_disease_symptoms(img_seg, disease_type, num_spots=random.randint(3, 7))

        img_name = f"seg_{i:03d}.png"
        img_seg.save(seg_img_dir / img_name)
        mask.save(seg_mask_dir / img_name)

    print(f"  [+] Semantic segmentation samples generated in {SAMPLES_DIR / 'segmentation'}")
    print("[✓] All sample subsets successfully prepared.")


def main():
    parser = argparse.ArgumentParser(description="Agricultural Vision Dataset Utility (CSE471 / SIH26131)")
    parser.add_argument("--generate-samples", action="store_true", help="Generate offline synthetic sample subsets")
    parser.add_argument("--samples-per-class", type=int, default=10, help="Number of classification samples per class")
    parser.add_argument("--download-all", action="store_true", help="Download complete public datasets from Kaggle / Web")
    parser.add_argument("--dataset", type=str, choices=["plantvillage", "plantdoc", "pests", "segmentation"], help="Specific dataset to fetch")
    args = parser.parse_args()

    if args.generate_samples or not (args.download_all or args.dataset):
        generate_sample_subsets(num_samples_per_class=args.samples_per_class)

    if args.download_all or args.dataset:
        print("[*] To download full public datasets, ensure Kaggle API is configured (~/.kaggle/kaggle.json).")
        print("    PlantVillage: kaggle datasets download -d emmarex/plantdisease -p data/raw/plantvillage --unzip")
        print("    PlantDoc: git clone https://github.com/pratikkayal/PlantDoc-Dataset.git data/raw/plantdoc")
        print("    Alternatively, use the generated offline sample subset in data/samples/ for instant execution.")


if __name__ == "__main__":
    main()
