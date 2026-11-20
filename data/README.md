# Agricultural Vision Datasets

This directory organizes data sources and folder structures for crop disease classification, pest object detection, and lesion semantic segmentation.

---

## Benchmark Datasets

### 1. Leaf Disease Classification & Domain Adaptation
- **PlantVillage Dataset:** Controlled laboratory leaf images across 38 crop and disease categories (e.g., Tomato, Potato, Apple, Rice).
- **PlantDoc Dataset:** Field-collected crop foliage images featuring complex background clutter, natural illumination shifts, and multiple overlapping leaves to evaluate real-world generalization.

### 2. Crop Pest & Foliar Lesion Detection
- **Foliar Pest Detection Dataset:** Annotated bounding boxes covering common agricultural pests (`aphid`, `armyworm`, `whitefly`, `leaf_miner`) and localized symptoms (`blast_lesion`, `blight_spot`) in Pascal VOC XML and YOLO formats.

### 3. Foliar Lesion Semantic Segmentation
- **Lesion Segmentation Dataset:** High-resolution leaf images paired with pixel-level binary masks (`0`: background/healthy leaf, `1`: infected necrotic lesion).

---

## Directory Organization

```text
data/
├── README.md               # Dataset descriptions and layout
├── samples/                # Sample image subsets for local testing (ignored by git)
│   ├── classification/
│   │   ├── lab_plantvillage/
│   │   └── field_plantdoc/
│   ├── detection/
│   │   ├── images/
│   │   └── annotations_voc/
│   └── segmentation/
│       ├── images/
│       └── masks/
├── raw/                    # Downloaded full raw archives (ignored by git)
└── processed/              # Normalized tensors & preprocessed splits (ignored by git)
```
