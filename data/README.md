# Agricultural Datasets for CSE471 & SIH26131

This directory contains configuration, scripts, and documentation for accessing public benchmark agricultural vision datasets, along with a deterministic sample generator for rapid offline prototyping.

---

## 1. Public Benchmark Datasets

### A. Leaf Disease Classification & Domain Adaptation
- **PlantVillage Dataset:**
  - *Source:* [Kaggle: PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) / [Penn State Open Access](https://plantvillage.psu.edu/)
  - *Characteristics:* 54,306 laboratory-controlled leaf images across 38 classes of healthy and diseased crops (tomato, potato, apple, corn, grape, rice).
  - *Academic Role:* Benchmark baseline for CNN training, transfer learning (Unit II, Unit III, Practicals 4–6).
- **PlantDoc Field Dataset:**
  - *Source:* [GitHub: PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) / arXiv:1911.10314
  - *Characteristics:* 2,598 real-world agricultural field images with complex backgrounds, illumination variations, and multiple overlapping leaves.
  - *Academic Role:* Domain adaptation and cross-domain generalization evaluation (Unit III & CO3).

### B. Crop Pest & Foliar Disease Object Detection
- **Crop Pest & Foliar Disease Detection Dataset (YOLO / VOC Format):**
  - *Source:* [Roboflow Universe: Crop Pest Detection](https://universe.roboflow.com/) or Kaggle AgriPest / IP102 subset.
  - *Classes:* Common agricultural pests (`aphid`, `armyworm`, `whitefly`, `leaf_miner`) and localized symptoms (`blast_lesion`, `blight_spot`).
  - *Annotations:* Bounding boxes in `[class, x_min, y_min, x_max, y_max]` (Pascal VOC) and normalized `[class, x_center, y_center, width, height]` (YOLO).
  - *Academic Role:* Unit IV, Practical 7 (YOLO inference), Practical 8 (Faster R-CNN fine-tuning).

### C. Leaf & Lesion Semantic Segmentation
- **Plant Disease Lesion Segmentation Dataset:**
  - *Source:* PlantDoc Segmentation Mask Repository / Kaggle Leaf Disease Segmentation.
  - *Annotations:* Pixel-level binary or multi-class masks (`0`: healthy/background, `1`: infected lesion).
  - *Academic Role:* Unit V, Practical 9 (U-Net architecture and Dice/IoU loss).

---

## 2. Automated Download & Sample Generation

Run the dedicated dataset script from the repository root:

```bash
# Option 1: Generate lightweight, realistic sample subsets for immediate testing (Recommended for quick runs)
python scripts/download_datasets.py --generate-samples

# Option 2: Download full public datasets via Kaggle API / KaggleHub
# (Requires kaggle credentials configured in ~/.kaggle/kaggle.json)
python scripts/download_datasets.py --download-all

# Option 3: Download specific target modality
python scripts/download_datasets.py --dataset plantvillage --limit 1000
python scripts/download_datasets.py --dataset plantdoc
python scripts/download_datasets.py --dataset pests
```

---

## 3. Directory Layout

```text
data/
├── README.md               # This documentation file
├── samples/                # Generated sample subsets for tests & notebooks
│   ├── classification/
│   │   ├── lab_plantvillage/
│   │   └── field_plantdoc/
│   ├── detection/
│   │   ├── images/
│   │   └── annotations/    # Pascal VOC XML and YOLO TXT
│   └── segmentation/
│       ├── images/
│       └── masks/          # Binary PNG masks
├── raw/                    # Untouched downloaded archives (ignored by git)
└── processed/              # Normalized tensors / train-val-test splits (ignored by git)
```
