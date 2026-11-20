# 🌾 AgriVision: Deep Learning System for Crop Disease Identification, Pest Localization & Agronomic Advisory

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Computer Vision and Deep Learning framework for precision agriculture. AgriVision provides automated leaf disease diagnosis, discrete foliar pest localization, pixel-level lesion surface area quantification, model explainability via Grad-CAM, and actionable Integrated Pest Management (IPM) extension advisories.

---

## 📌 Overview & Problem Statement

Agricultural crop diseases and pest infestations cause significant yield losses every year. Traditional field scouting is manual, subjective, and often conducted too late after symptoms have spread widely across the crop canopy.

**AgriVision** provides a unified vision pipeline that addresses four key diagnostics:
1. **Disease Identification:** Identifies crop species and pathogenic condition (Tomato, Potato, Rice, Apple, Corn).
2. **Pest & Spot Localization:** Predicts bounding boxes for active insects (`aphid`, `armyworm`, `whitefly`, `leaf_miner`) and necrotic spots using Non-Maximum Suppression (NMS).
3. **Damage Severity Quantification:** Segments affected foliar lesions at the pixel level using U-Net, computing the exact percentage of damaged leaf area (Mild $<10\%$, Moderate $10-30\%$, Severe $>30\%$).
4. **Model Explainability & Field Advisory:** Audits model focus using Grad-CAM heatmaps to verify attention on biological symptoms, then maps the findings to structured cultural, biological, and targeted chemical recommendations.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    In["Input Crop Leaf Image (RGB)"] --> S1["Stage 1: Disease Classification (ResNet / MobileNet)"]
    In --> S2["Stage 2: Pest & Spot Detection (YOLO / Faster R-CNN)"]
    In --> S3["Stage 3: Lesion Semantic Segmentation (U-Net)"]

    S1 --> D1["Crop & Pathogen Diagnosis + Confidence Score"]
    S2 --> D2["Bounding Boxes, Pest Labels & Confidence"]
    S3 --> D3["Binary Mask & % Affected Surface Area"]

    D1 & D2 & D3 --> S4["Stage 4: Explainability & Agronomic Advisory Engine"]
    S4 --> XAI["Grad-CAM Saliency Attribution Map"]
    S4 --> IPM["Integrated Pest Management (IPM) Field Guide"]

    IPM --> Adv1["🌿 Cultural & Sanitation Practices"]
    IPM --> Adv2["🐞 Biological Bio-Controls (Neem Oil, Trichoderma)"]
    IPM --> Adv3["🧪 Targeted Chemical Sprays & Dosage Guidelines"]
```

---

## 🚀 Simple Steps to Run

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone git@github.com:harshitchauhann95/crop-disease-pest-vision-cse471.git
cd crop-disease-pest-vision-cse471

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Model Training & Notebooks
The `notebooks/` directory contains step-by-step Jupyter notebooks covering each phase of the project:
- **`01` to `03`:** Image preprocessing, color space conversions (RGB, HSV, LAB), and 2D spatial convolution from scratch in NumPy.
- **`04` to `06`:** CNN training (LeNet vs. ResNet-18) and transfer learning on MobileNet with differential learning rates.
- **`07` to `08`:** Object detection with IoU, Non-Maximum Suppression (NMS) from scratch, and Faster R-CNN fine-tuning.
- **`09`:** U-Net semantic segmentation for lesion masks and surface damage percentage calculation.
- **`10`:** Grad-CAM attribution heatmaps auditing correctly classified vs. misclassified samples.
- **`11` to `12`:** Vision Transformers (ViT patch attention), zero-shot matching, and the integrated end-to-end pipeline.

To run any notebook:
```bash
jupyter notebook notebooks/
```

### 3. Launch the Interactive Web Dashboard
Run the multi-stage visual diagnostic application in your web browser:
```bash
streamlit run src/app.py
```
- Open `http://localhost:8501` in your browser.
- Select from preset benchmark leaf samples or upload your own leaf photo.
- View real-time outputs: disease classification, bounding box detection, U-Net mask overlay, Grad-CAM heatmap, and the field management advisory.

---

## 📁 Repository Structure

```text
crop-disease-pest-vision-cse471/
├── README.md                                   # Project documentation and workflow
├── requirements.txt                           # Dependency specifications
├── .gitignore                                 # Excludes local caches, binaries, and datasets
├── configs/
│   └── config.yaml                            # Centralized hyperparameter & class settings
├── data/
│   └── README.md                              # Dataset organization and descriptions
├── notebooks/
│   ├── 01_prac01_opencv_preprocessing.ipynb    # Color spaces (RGB, HSV, LAB) & histograms
│   ├── 02_prac02_numpy_2d_conv_filters.ipynb  # 2D spatial convolution from scratch
│   ├── 03_prac03_conv_stride_padding_features.ipynb # Stride, padding, pooling & receptive field
│   ├── 04_prac04_lenet_vs_resnet18_mnist.ipynb# LeNet-5 vs. ResNet-18 skip connections
│   ├── 05_prac05_feature_extraction_classifier.ipynb # Feature extraction & linear head training
│   ├── 06_prac06_mobilenet_finetuning_unfreezing.ipynb # Fine-tuning with differential learning rates
│   ├── 07_prac07_yolo_object_detection.ipynb  # Pest object detection & NMS from scratch
│   ├── 08_prac08_faster_rcnn_finetuning.ipynb # Two-stage Faster R-CNN fine-tuning
│   ├── 09_prac09_unet_segmentation_overlay.ipynb # U-Net lesion segmentation & % severity
│   ├── 10_prac10_gradcam_saliency_analysis.ipynb # Grad-CAM explainability & failure auditing
│   ├── 11_sih_vit_swin_clip_generative_experiments.ipynb # Vision Transformers & CLIP prompt matching
│   └── 12_sih_end_to_end_crop_pipeline.ipynb  # Full multi-stage pipeline integration
└── src/
    ├── __init__.py                            # Package metadata
    ├── dataset.py                             # PyTorch Datasets & DataLoaders
    ├── transforms.py                          # Augmentations & color conversions
    ├── models_cnn.py                          # LeNet, ResNet18Scratch, transfer wrappers, loss functions
    ├── detector.py                            # IoU, NMS from scratch, Faster R-CNN, mAP calculation
    ├── segmenter.py                           # U-Net architecture, Dice score, severity scoring
    ├── vlm_and_gradcam.py                     # Grad-CAM from scratch, saliency maps, CLIP matcher
    ├── advisory.py                            # Agronomic Knowledge Base & ICAR/CIBRC guidelines
    └── app.py                                 # Interactive Streamlit Web App & CLI entrypoint
```

---

## 📊 Benchmark Results

| Vision Modality / Task | Model Architecture | Metric | Result |
| :--- | :--- | :--- | :--- |
| **Disease Classification** | ResNet-18 (Residual from scratch) | Validation Accuracy | **93.8%** |
| **Disease Classification** | MobileNetV2 (Fine-Tuned) | Top-1 Accuracy | **96.5%** |
| **Multi-Class Evaluation** | Overall Diagnostic Pipeline | Test Accuracy | **94.0%** |
| **Pest / Lesion Detection** | Two-Stage Detector with NMS | Mean Average Precision (mAP@0.5) | **81.4%** |
| **Lesion Segmentation** | U-Net with Skip Connections | Sørensen-Dice Score | **0.884** |
| **Lesion Segmentation** | U-Net with Skip Connections | Pixel Accuracy | **96.8%** |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
