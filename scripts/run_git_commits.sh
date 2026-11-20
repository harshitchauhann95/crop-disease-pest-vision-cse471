#!/usr/bin/env bash
# ==============================================================================
# scripts/run_git_commits.sh
# ------------------------------------------------------------------------------
# Automated Atomic Git Commit & Remote Initialization Script
# Project: CSE471 (Deep Learning for CV) & SIH26131 (Crop Disease & Pest Vision)
# Author: Harshit Chauhan (harshitchauhan05spn@gmail.com / harshitchauhann95)
# ==============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "========================================================================"
echo "Initializing & Configuring Git Repository for Harshit Chauhan"
echo "========================================================================"

# 1. Initialize repository if not already initialized
if [ ! -d ".git" ]; then
    git init -b main
    echo "[✓] Initialized new Git repository on branch 'main'."
else
    echo "[*] Existing Git repository detected."
fi

# 2. Configure Git Identity
git config user.name "Harshit Chauhan"
git config user.email "harshitchauhan05spn@gmail.com"
echo "[✓] Configured Git Identity:"
echo "    User Name : $(git config user.name)"
echo "    User Email: $(git config user.email)"

# 3. Configure Remote URL
# Recommended repository name: crop-disease-pest-vision-cse471
TARGET_REMOTE="git@github.com:harshitchauhann95/crop-disease-pest-vision-cse471.git"
FALLBACK_REMOTE="git@github.com:harshitchauhann95/fake.git"

if git remote | grep -q "^origin$"; then
    git remote set-url origin "$TARGET_REMOTE"
    echo "[✓] Updated existing remote 'origin' to $TARGET_REMOTE"
else
    git remote add origin "$TARGET_REMOTE"
    echo "[✓] Added remote 'origin' -> $TARGET_REMOTE"
fi

# Ensure datasets and cache binaries are ignored
git rm -r --cached data/samples/ 2>/dev/null || true
git rm -r --cached data/raw/ 2>/dev/null || true
git rm -r --cached __pycache__/ 2>/dev/null || true

echo ""
echo "========================================================================"
echo "Executing Atomic Commits across Academic & SIH Milestones"
echo "========================================================================"

# Milestone 1: Environment, Configuration & Dataset Acquisition Pipeline
git add .gitignore requirements.txt configs/config.yaml data/README.md scripts/download_datasets.py
if ! git diff --cached --quiet; then
    git commit -m "feat(setup): initialize environment, config, and dataset acquisition pipeline"
    echo "[✓] Committed Milestone 1: Environment & Dataset Pipeline"
else
    echo "[-] Milestone 1 already committed."
fi

# Milestone 2: Foundations & Low-Level Vision (Unit I & II, Practicals 1-4)
git add src/transforms.py src/models_cnn.py \
    notebooks/01_prac01_opencv_preprocessing.ipynb \
    notebooks/02_prac02_numpy_2d_conv_filters.ipynb \
    notebooks/03_prac03_conv_stride_padding_features.ipynb \
    notebooks/04_prac04_lenet_vs_resnet18_mnist.ipynb
if ! git diff --cached --quiet; then
    git commit -m "feat(foundations): implement color spaces, 2d conv from scratch, and lenet vs resnet18"
    echo "[✓] Committed Milestone 2: Unit I-II Foundations & CNN Architectures"
else
    echo "[-] Milestone 2 already committed."
fi

# Milestone 3: Transfer Learning & Fine-Tuning Strategies (Unit III, Practicals 5-6)
git add src/dataset.py \
    notebooks/05_prac05_feature_extraction_classifier.ipynb \
    notebooks/06_prac06_mobilenet_finetuning_unfreezing.ipynb
if ! git diff --cached --quiet; then
    git commit -m "feat(transfer-learning): add feature extraction, layer freezing, and mobilenet fine-tuning"
    echo "[✓] Committed Milestone 3: Transfer Learning & Domain Adaptation"
else
    echo "[-] Milestone 3 already committed."
fi

# Milestone 4: Object Detection & Localization (Unit IV, Practicals 7-8)
git add src/detector.py \
    notebooks/07_prac07_yolo_object_detection.ipynb \
    notebooks/08_prac08_faster_rcnn_finetuning.ipynb
if ! git diff --cached --quiet; then
    git commit -m "feat(detection): implement iou, nms from scratch, and yolo/faster r-cnn pest detection"
    echo "[✓] Committed Milestone 4: Object Detection & Bounding Box Localization"
else
    echo "[-] Milestone 4 already committed."
fi

# Milestone 5: Semantic Segmentation & Disease Severity Scoring (Unit V, Practical 9)
git add src/segmenter.py \
    notebooks/09_prac09_unet_segmentation_overlay.ipynb
if ! git diff --cached --quiet; then
    git commit -m "feat(segmentation): implement u-net from scratch and foliar lesion severity estimation"
    echo "[✓] Committed Milestone 5: U-Net Segmentation & Severity Analysis"
else
    echo "[-] Milestone 5 already committed."
fi

# Milestone 6: Vision Transformers, Generative Models & Explainability (Unit VI, Practicals 10-11)
git add src/vlm_and_gradcam.py \
    notebooks/10_prac10_gradcam_saliency_analysis.ipynb \
    notebooks/11_sih_vit_swin_clip_generative_experiments.ipynb
if ! git diff --cached --quiet; then
    git commit -m "feat(xai-vlm): implement grad-cam from scratch, saliency maps, vit, and clip prompt matching"
    echo "[✓] Committed Milestone 6: Grad-CAM Explainability & Multi-Modal Models"
else
    echo "[-] Milestone 6 already committed."
fi

# Milestone 7: SIH26131 End-to-End Pipeline & Extension Advisory Engine
git add src/advisory.py src/app.py src/__init__.py \
    notebooks/12_sih_end_to_end_crop_pipeline.ipynb
if ! git diff --cached --quiet; then
    git commit -m "feat(sih26131): build multi-stage crop vision pipeline and agronomic advisory engine"
    echo "[✓] Committed Milestone 7: SIH26131 Full Pipeline & Extension Advisory"
else
    echo "[-] Milestone 7 already committed."
fi

# Milestone 8: Documentation, Commit Script & Final Academic Polish
git add README.md scripts/
if ! git diff --cached --quiet; then
    git commit -m "docs(academic): add comprehensive syllabus mapping, traceability matrix, and user guide"
    echo "[✓] Committed Milestone 8: Documentation & Traceability"
else
    echo "[-] Milestone 8 already committed."
fi

echo ""
echo "========================================================================"
echo "Git Commit History:"
git log --oneline -n 10
echo "========================================================================"
echo ""
echo "[*] Next Steps to Push to GitHub:"
echo "    1. On GitHub (https://github.com/harshitchauhann95/fake):"
echo "       Settings -> General -> Repository name -> Rename 'fake' to 'crop-disease-pest-vision-cse471'"
echo "    2. Push to GitHub:"
echo "       git push -u origin main --force"
echo "    (If keeping the name 'fake', run: git remote set-url origin git@github.com:harshitchauhann95/fake.git && git push -u origin main)"
echo "========================================================================"
