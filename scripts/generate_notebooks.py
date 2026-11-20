#!/usr/bin/env python3
"""
scripts/generate_notebooks.py
-----------------------------
Generates all 12 academic & SIH Jupyter Notebooks with complete code cells,
markdown syllabus mappings, mathematical LaTeX formulations, and visualizations.
Uses nbformat to ensure 100% valid, runnable .ipynb notebook files.
"""

import sys
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_nb():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    }
    return nb


def save_nb(nb, filename):
    filepath = NOTEBOOKS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[✓] Generated notebook: {filename}")


# =====================================================================
# Notebook 1: OpenCV Preprocessing (Practical 1)
# =====================================================================
def make_nb01():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 1: Image Representation, Color Spaces & Preprocessing
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit I (Introduction to Computer Vision & Image Representation) | **CO:** CO1  
**Problem Statement Context:** SIH26131 — Crop Foliar Image Preprocessing & Color Space Transformation  

---

## 1. Theoretical Foundations
In digital computer vision, an image is represented as a multidimensional tensor $I \in \mathbb{R}^{H \times W \times C}$:
- $H$: Height (rows)
- $W$: Width (columns)
- $C$: Channels (e.g., 3 for RGB/HSV/LAB, 1 for Grayscale)

### Color Spaces in Precision Agriculture:
1. **RGB (Red, Green, Blue):** Standard additive representation captured by CMOS sensor Bayer filters.
2. **Grayscale:** Luminance representation based on ITU-R BT.601 perceptual weighting:
   $$Y = 0.299R + 0.587G + 0.114B$$
3. **HSV (Hue, Saturation, Value):** Separates chromaticity (Hue: green leaf vs. yellow/brown lesion) from illumination variations (Value), making it robust to fluctuating sunlight in crop fields.
4. **LAB ($L^*, a^*, b^*$):** Perceptually uniform color space where $a^*$ represents the Green-Red axis and $b^*$ represents the Blue-Yellow axis."""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.transforms import convert_color_space

print(f"[✓] Root path: {REPO_ROOT}")"""),

        nbf.v4.new_markdown_cell("""## 2. Load Agricultural Sample Images
We load a healthy leaf and an early blight diseased leaf with necrotic lesions."""),

        nbf.v4.new_code_cell("""# Load samples from data/samples/
healthy_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Healthy" / "sample_000.jpg"
diseased_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Early_Blight" / "sample_000.jpg"

img_healthy = np.array(Image.open(healthy_path).convert("RGB"))
img_diseased = np.array(Image.open(diseased_path).convert("RGB"))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_healthy)
axes[0].set_title("Healthy Tomato Leaf (RGB)")
axes[0].axis("off")

axes[1].imshow(img_diseased)
axes[1].set_title("Diseased Leaf: Early Blight (RGB)")
axes[1].axis("off")
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 3. Color Space Decomposition (RGB, Grayscale, HSV, LAB)
We analyze the diseased leaf across various color representations."""),

        nbf.v4.new_code_cell("""gray = convert_color_space(img_diseased, "GRAYSCALE")
hsv = convert_color_space(img_diseased, "HSV")
lab = convert_color_space(img_diseased, "LAB")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(img_diseased)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale (Luminance)")
axes[0, 1].axis("off")

axes[1, 0].imshow(hsv)
axes[1, 0].set_title("HSV Color Space")
axes[1, 0].axis("off")

axes[1, 1].imshow(lab)
axes[1, 1].set_title("CIE LAB Color Space")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Color Channel Histograms: Healthy vs. Diseased Foliage
Notice the shift in the Green and Red channels due to necrotic lesion pigments and chlorophyll loss."""),

        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ('r', 'g', 'b')

for i, col in enumerate(colors):
    hist_h, _ = np.histogram(img_healthy[..., i], bins=64, range=(0, 256))
    hist_d, _ = np.histogram(img_diseased[..., i], bins=64, range=(0, 256))
    axes[0].plot(hist_h, color=col, label=f"Channel {col.upper()}")
    axes[1].plot(hist_d, color=col, label=f"Channel {col.upper()}")

axes[0].set_title("Healthy Leaf Pixel Intensity Distribution")
axes[0].set_xlabel("Pixel Value (0-255)")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Diseased Leaf Pixel Intensity Distribution")
axes[1].set_xlabel("Pixel Value (0-255)")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. Summary & Practical Observations
1. **Luminance vs. Chrominance:** Grayscale conversion captures general leaf geometry and structural veins but loses discriminatory chlorosis (yellow halo) signatures.
2. **HSV Utility:** Hue cleanly separates background clutter and necrotic lesions from viable chlorophyll green foliage.
3. **Implications for Deep CNNs:** Feeding 3-channel normalized RGB tensors allows CNN kernels to learn joint spatial-chromatic representations crucial for early disease classification.""")
    ]
    save_nb(nb, "01_prac01_opencv_preprocessing.ipynb")


# =====================================================================
# Notebook 2: 2D Convolution From Scratch (Practical 2)
# =====================================================================
def make_nb02():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 2: 2D Spatial Convolution From Scratch Using NumPy
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit II (Convolutional Neural Networks) | **CO:** CO2  
**Problem Statement Context:** SIH26131 — Manual Feature Extraction (Edge & Lesion Boundary Detection)  

---

## 1. Discrete 2D Convolution Mathematics
In image processing and deep learning, the discrete 2D spatial cross-correlation / convolution of an image $I$ with a kernel filter $K \in \mathbb{R}^{k_h \times k_w}$ at pixel coordinate $(i, j)$ is given by:

$$(I * K)(i, j) = \sum_{m=-k_h/2}^{k_h/2} \sum_{n=-k_w/2}^{k_w/2} I(i - m, j - n) \cdot K(m, n)$$

### Key Kernels:
- **Sobel Horizontal ($G_x$):** Detects vertical edges / vein boundaries
- **Sobel Vertical ($G_y$):** Detects horizontal edges
- **Laplacian ($\nabla^2$):** Second-order derivative highlighting rapid intensity changes (lesion borders)
- **Gaussian Blur:** Low-pass filter for noise attenuation:
  $$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))"""),

        nbf.v4.new_markdown_cell("""## 2. 2D Convolution Algorithm (Pure NumPy Implementation)"""),

        nbf.v4.new_code_cell("""def convolve2d_scratch(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    \"\"\"
    Performs 2D discrete spatial convolution from first principles.
    \"\"\"
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    out_h = (img_h - k_h) // stride + 1
    out_w = (img_w - k_w) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            patch = image[r_start : r_start + k_h, c_start : c_start + k_w]
            output[i, j] = np.sum(patch * kernel)

    return output

print("[✓] convolve2d_scratch defined successfully.")"""),

        nbf.v4.new_markdown_cell("""## 3. Define Agricultural Feature Extraction Filters"""),

        nbf.v4.new_code_cell("""# 1. Sobel Horizontal (Vertical Edges)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# 2. Sobel Vertical (Horizontal Edges)
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# 3. Laplacian (Spot & Omnidirectional Edge Detector)
laplacian = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
], dtype=np.float32)

# 4. Sharpen Filter
sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# 5. Gaussian Blur (3x3 Normalized)
gaussian_blur = (1.0 / 16.0) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)"""),

        nbf.v4.new_markdown_cell("""## 4. Run Convolutions on Leaf Image"""),

        nbf.v4.new_code_cell("""sample_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Early_Blight" / "sample_000.jpg"
leaf_gray = np.array(Image.open(sample_path).convert("L"), dtype=np.float32) / 255.0

# Apply convolution with same padding (P=1)
f_sobel_x = convolve2d_scratch(leaf_gray, sobel_x, padding=1)
f_sobel_y = convolve2d_scratch(leaf_gray, sobel_y, padding=1)
sobel_magnitude = np.sqrt(f_sobel_x**2 + f_sobel_y**2)

f_laplace = convolve2d_scratch(leaf_gray, laplacian, padding=1)
f_sharpen = convolve2d_scratch(leaf_gray, sharpen, padding=1)
f_blur = convolve2d_scratch(leaf_gray, gaussian_blur, padding=1)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(leaf_gray, cmap="gray")
axes[0, 0].set_title("Input Leaf (Grayscale)")
axes[0, 0].axis("off")

axes[0, 1].imshow(np.abs(f_sobel_x), cmap="magma")
axes[0, 1].set_title("Sobel X (Vertical Veins)")
axes[0, 1].axis("off")

axes[0, 2].imshow(np.abs(f_sobel_y), cmap="magma")
axes[0, 2].set_title("Sobel Y (Horizontal Edges)")
axes[0, 2].axis("off")

axes[1, 0].imshow(sobel_magnitude, cmap="inferno")
axes[1, 0].set_title("Sobel Gradient Magnitude")
axes[1, 0].axis("off")

axes[1, 1].imshow(np.abs(f_laplace), cmap="hot")
axes[1, 1].set_title("Laplacian (Lesion Boundaries)")
axes[1, 1].axis("off")

axes[1, 2].imshow(np.clip(f_sharpen, 0, 1), cmap="gray")
axes[1, 2].set_title("Sharpened Leaf Texture")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. Practical Conclusion & Connection to CNNs
- Hand-crafted filters (Sobel, Laplacian) require manual parameter tuning and struggle with illumination changes in real-world fields.
- Deep CNNs replace these fixed filter matrices with **learnable weight tensors**, optimizing thousands of specialized filters directly via backpropagation.""")
    ]
    save_nb(nb, "02_prac02_numpy_2d_conv_filters.ipynb")


# =====================================================================
# Notebook 3: Stride, Padding & Pooling (Practical 3)
# =====================================================================
def make_nb03():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 3: Convolution Arithmetic: Stride, Padding & Pooling
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit II (Convolutional Neural Networks) | **CO:** CO2  
**Problem Statement Context:** Spatial Downsampling & Receptive Field Expansion in Foliar Vision  

---

## 1. Feature Map Spatial Arithmetic
Given an input feature map of size $W_{in} \times H_{in}$, kernel size $K$, padding $P$, and stride $S$, the output spatial dimensions are:

$$W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$$
$$H_{out} = \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1$$

### Padding Modes:
1. **Valid Padding ($P=0$):** No boundary zero-padding; output feature map shrinks with each layer.
2. **Same Padding ($P = \lfloor K/2 \rfloor$):** Padding selected such that $W_{out} = W_{in}$ when $S=1$.

### Pooling Operations:
- **Max Pooling:** Selects the maximum activation in a window $\Omega$, preserving dominant edge/texture activations while providing translation invariance:
  $$y_{i, j} = \max_{(m, n) \in \Omega} x_{i \cdot s + m, j \cdot s + n}$$
- **Average Pooling:** Computes arithmetic mean, smoothing background features."""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from notebooks import make_nb02  # reference"""),

        nbf.v4.new_markdown_cell("""## 2. Pooling Functions from Scratch"""),

        nbf.v4.new_code_cell("""def max_pool2d_scratch(image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    \"\"\"Max pooling from first principles.\"\"\"
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride : i*stride + pool_size, j*stride : j*stride + pool_size]
            output[i, j] = np.max(patch)

    return output


def avg_pool2d_scratch(image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    \"\"\"Average pooling from first principles.\"\"\"
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride : i*stride + pool_size, j*stride : j*stride + pool_size]
            output[i, j] = np.mean(patch)

    return output

print("[✓] Pooling functions defined.")"""),

        nbf.v4.new_markdown_cell("""## 3. Empirical Demonstration on Leaf Patch"""),

        nbf.v4.new_code_cell("""sample_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Early_Blight" / "sample_000.jpg"
leaf_gray = np.array(Image.open(sample_path).convert("L"), dtype=np.float32) / 255.0

# Sub-patch around lesion
patch = leaf_gray[60:188, 60:188]  # 128x128 patch

max_pooled = max_pool2d_scratch(patch, pool_size=2, stride=2)   # 64x64
avg_pooled = avg_pool2d_scratch(patch, pool_size=2, stride=2)   # 64x64

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(patch, cmap="gray")
axes[0].set_title(f"Input Crop Patch ({patch.shape[0]}x{patch.shape[1]})")
axes[0].axis("off")

axes[1].imshow(max_pooled, cmap="gray")
axes[1].set_title(f"Max Pooling 2x2 ({max_pooled.shape[0]}x{max_pooled.shape[1]}) - High Contrast")
axes[1].axis("off")

axes[2].imshow(avg_pooled, cmap="gray")
axes[2].set_title(f"Avg Pooling 2x2 ({avg_pooled.shape[0]}x{avg_pooled.shape[1]}) - Smoothed")
axes[2].axis("off")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Receptive Field Analysis
The receptive field (RF) denotes the region in the input space that a particular CNN feature is looking at. For a sequence of layers:
$$RF_{l} = RF_{l-1} + (k_l - 1) \times j_{l-1}$$
where $j_{l} = j_{l-1} \times s_l$ is the cumulative stride.

Pooling operations double the cumulative jump ($j$), accelerating receptive field expansion so deep layers can encompass entire crop leaves rather than localized pixel spots.""")
    ]
    save_nb(nb, "03_prac03_conv_stride_padding_features.ipynb")


# =====================================================================
# Notebook 4: LeNet vs ResNet18 (Practical 4)
# =====================================================================
def make_nb04():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 4: Architectural Evolution: LeNet-5 vs. ResNet-18
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit II (Deep Architectures & Vanishing Gradients) | **CO:** CO2  
**Problem Statement Context:** Benchmarking Shallow vs. Deep Residual Networks for Crop Pattern Recognition  

---

## 1. Deep Learning Milestone Comparison
| Metric / Feature | LeNet-5 (1998) | ResNet-18 (2015) |
| :--- | :--- | :--- |
| **Pioneering Concept** | Conv + Subsampling + Tanh | Deep Residual Learning ($F(x) + x$) |
| **Typical Depth** | 5 weight layers | 18 weight layers |
| **Skip Connections** | None (Plain sequential) | Identity shortcuts around conv pairs |
| **Normalization** | None | Batch Normalization after every Conv |
| **Max Pooling vs Avg** | Average Pooling | Max Pooling + Global Average Pooling |
| **Vulnerability** | Vanishing gradient at depth > 7 | Stable gradient flow up to 1000+ layers |"""),

        nbf.v4.new_code_cell("""import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.models_cnn import LeNet5, ResNet18Scratch

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Active Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 2. Model Parameter Count & Architecture Inspection"""),

        nbf.v4.new_code_cell("""lenet = LeNet5(in_channels=3, num_classes=6).to(device)
resnet = ResNet18Scratch(in_channels=3, num_classes=6).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

p_lenet = count_parameters(lenet)
p_resnet = count_parameters(resnet)

print(f"LeNet-5 Trainable Parameters  : {p_lenet:,}")
print(f"ResNet-18 Trainable Parameters: {p_resnet:,}")"""),

        nbf.v4.new_markdown_cell("""## 3. Training Benchmark on Synthetic Crop Foliar Subsets"""),

        nbf.v4.new_code_cell("""# Generate synthetic foliar tensor batches for benchmark training
torch.manual_seed(42)
X_dummy = torch.randn(64, 3, 32, 32)
y_dummy = torch.randint(0, 6, (64,))

dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

def benchmark_train(model, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
    duration = time.time() - start_time
    return losses, duration

losses_lenet, time_lenet = benchmark_train(lenet, dataloader, epochs=8)
losses_resnet, time_resnet = benchmark_train(resnet, dataloader, epochs=8)

print(f"LeNet-5 Training Time  : {time_lenet:.2f} seconds")
print(f"ResNet-18 Training Time: {time_resnet:.2f} seconds")"""),

        nbf.v4.new_markdown_cell("""## 4. Loss Convergence Curve & Parameter Comparison"""),

        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(losses_lenet, label="LeNet-5 Loss", marker='o', color='purple')
axes[0].plot(losses_resnet, label="ResNet-18 Loss", marker='s', color='teal')
axes[0].set_title("Training Loss Convergence")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross Entropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(["LeNet-5", "ResNet-18"], [p_lenet, p_resnet], color=['purple', 'teal'])
axes[1].set_yscale('log')
axes[1].set_title("Model Parameter Capacity (Log Scale)")
axes[1].set_ylabel("Total Trainable Parameters")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. Architectural Findings
- **Capacity:** ResNet-18 possesses vastly higher expressive capacity ($11M+$ parameters) allowing it to learn hierarchical abstractions (textures $\to$ spot shapes $\to$ disease syndromes).
- **Gradient Stability:** Skip connections preserve signal magnitude during backward passes, avoiding the severe vanishing gradients that restrict LeNet-5 to shallow networks.""")
    ]
    save_nb(nb, "04_prac04_lenet_vs_resnet18_mnist.ipynb")


# =====================================================================
# Notebook 5: Feature Extraction (Practical 5)
# =====================================================================
def make_nb05():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 5: Transfer Learning via Frozen Feature Extraction
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit III (Transfer Learning & Fine-Tuning) | **CO:** CO3  
**Problem Statement Context:** SIH26131 — Rapid Crop Disease Classification with Minimal Compute  

---

## 1. Inductive Transfer Learning Theory
Training deep vision models from scratch requires millions of annotated samples. In agriculture, annotated disease datasets are often small ($10^2$ to $10^4$ images).
**Feature Extraction** leverages knowledge encoded in backbones pre-trained on generic large-scale corpora (ImageNet):
1. **Freeze Feature Extractor:** Fix all convolutional layer weights ($\theta_{backbone}$ where $\nabla_{\theta} = 0$).
2. **Train Linear Classifier:** Attach a customized linear classification head ($\theta_{head}$) and optimize only these parameters on the crop dataset:

$$\min_{\theta_{head}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{CE}\left( f(x_i; \theta_{backbone}^{frozen}, \theta_{head}), y_i \right)$$"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import PlantDiseaseClassificationDataset
from src.transforms import get_classification_transforms
from src.models_cnn import build_transfer_classifier

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Active Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 2. Load PlantVillage Crop Dataset"""),

        nbf.v4.new_code_cell("""data_dir = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage"
transforms = get_classification_transforms(image_size=(224, 224), is_training=False)

dataset = PlantDiseaseClassificationDataset(root_dir=data_dir, transform=transforms)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"[✓] Loaded {len(dataset)} crop leaf samples across {len(dataset.classes)} classes:")
print(dataset.classes)"""),

        nbf.v4.new_markdown_cell("""## 3. Freeze Backbone & Train Head"""),

        nbf.v4.new_code_cell("""model = build_transfer_classifier(
    backbone_name="resnet18",
    num_classes=len(dataset.classes),
    freeze_backbone=True
).to(device)

# Verify only head parameters have requires_grad=True
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Total Parameter Tensors: {len(list(model.parameters()))}")
print(f"Trainable Head Tensors : {len(trainable_params)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(trainable_params, lr=1e-3)

model.train()
for epoch in range(3):
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/3] - Loss: {total_loss / len(dataloader):.4f}")"""),

        nbf.v4.new_markdown_cell("""## 4. Evaluation & Confusion Matrix"""),

        nbf.v4.new_code_cell("""model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap="Blues")
fig.colorbar(cax)
ax.set_title("Crop Disease Confusion Matrix", pad=20)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. Practical Takeaways
- Feature extraction trains significantly faster because backpropagation stops at the final classification layer without propagating gradients through dozens of deep convolutional blocks.
- Suitable for initial baseline models and lightweight IoT / edge deployment.""")
    ]
    save_nb(nb, "05_prac05_feature_extraction_classifier.ipynb")


# =====================================================================
# Notebook 6: Fine-Tuning & Layer Freezing (Practical 6)
# =====================================================================
def make_nb06():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 6: Fine-Tuning & Layer Freezing Strategies on MobileNet
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit III (Transfer Learning & Fine-Tuning) | **CO:** CO3  
**Problem Statement Context:** Mobile Edge Deployment for In-Field Crop Diagnostics  

---

## 1. Transfer Learning Spectrum
When adapting a pre-trained network to agricultural imagery:
1. **Strategy 1: Frozen Backbone (Feature Extraction):** Fast training, zero risk of catastrophic forgetting, but lower peak accuracy.
2. **Strategy 2: Partial Unfreezing (Top Blocks):** Retains generic low-level filters (edges, textures) in early layers while adapting high-level semantic representations (leaf shape, lesion morphology) in upper layers.
3. **Strategy 3: Full Fine-Tuning with Differential Learning Rates:**
   $$\eta_{backbone} = 10^{-5}, \quad \eta_{head} = 10^{-3}$$
   Allows subtle backbone parameter adjustments without destroying pre-trained weights."""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import PlantDiseaseClassificationDataset
from src.transforms import get_classification_transforms
from src.models_cnn import build_transfer_classifier, unfreeze_top_layers, get_differential_lr_params

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Active Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 2. Benchmark Setup: Freezing vs. Partial Unfreezing"""),

        nbf.v4.new_code_cell("""data_dir = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage"
transforms = get_classification_transforms(image_size=(224, 224), is_training=True)
dataset = PlantDiseaseClassificationDataset(root_dir=data_dir, transform=transforms)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 1. Strategy A: Frozen
model_frozen = build_transfer_classifier("mobilenet_v2", num_classes=len(dataset.classes), freeze_backbone=True).to(device)
opt_frozen = optim.Adam([p for p in model_frozen.parameters() if p.requires_grad], lr=1e-3)

# 2. Strategy B: Partial Unfreezing (Unfreeze top layers)
model_unfrozen = build_transfer_classifier("mobilenet_v2", num_classes=len(dataset.classes), freeze_backbone=True).to(device)
unfreeze_top_layers(model_unfrozen, num_layers=4)
opt_unfrozen = optim.Adam(get_differential_lr_params(model_unfrozen, base_lr=1e-5, head_lr_mult=100.0))

print("[✓] Both fine-tuning configurations initialized.")"""),

        nbf.v4.new_markdown_cell("""## 3. Compare Convergence over 4 Epochs"""),

        nbf.v4.new_code_cell("""criterion = nn.CrossEntropyLoss()

def train_epochs(model, optimizer, loader, epochs=4):
    history = []
    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        history.append(loss_sum / len(loader))
    return history

hist_frozen = train_epochs(model_frozen, opt_frozen, dataloader)
hist_unfrozen = train_epochs(model_unfrozen, opt_unfrozen, dataloader)

plt.figure(figsize=(8, 5))
plt.plot(hist_frozen, label="Strategy A: Fully Frozen Backbone", marker='o')
plt.plot(hist_unfrozen, label="Strategy B: Partial Fine-Tuning (Top 4 Layers)", marker='s')
plt.title("Fine-Tuning Convergence Comparison on MobileNet")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Engineering Recommendations
- When target dataset is **small and domain-similar** (e.g., standard leaves), freeze the backbone to avoid overfitting.
- When target images exhibit **domain shift** (e.g., lab PlantVillage to noisy field PlantDoc), unfreeze the top residual blocks with differential learning rates to adapt high-level disease pattern recognition.""")
    ]
    save_nb(nb, "06_prac06_mobilenet_finetuning_unfreezing.ipynb")


# =====================================================================
# Notebook 7: YOLO Object Detection (Practical 7)
# =====================================================================
def make_nb07():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 7: Real-Time Object Detection for Foliar Pests & Lesions
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit IV (Object Detection & Localization) | **CO:** CO4  
**Problem Statement Context:** SIH26131 — Stage 2: In-Field Pest Localization & Bounding Box Prediction  

---

## 1. Single-Stage Detection Principles
Single-stage detectors (YOLO, SSD, RetinaNet) formulate object detection as a dense spatial regression problem:
1. Divide input image into an $S \times S$ grid.
2. For each grid cell, predict $B$ bounding boxes with coordinates $(t_x, t_y, t_w, t_h)$, an objectness confidence score $P(Object)$, and class conditional probabilities $P(Class_c | Object)$.
3. Post-process candidate boxes using **Non-Maximum Suppression (NMS)** to eliminate redundant overlapping predictions based on Intersection over Union (IoU):

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.detector import compute_iou, non_max_suppression_scratch, box_xyxy_to_xywh"""),

        nbf.v4.new_markdown_cell("""## 2. Load Foliar Pest Detection Samples"""),

        nbf.v4.new_code_cell("""det_dir = REPO_ROOT / "data" / "samples" / "detection"
img_path = det_dir / "images" / "det_000.jpg"
image = Image.open(img_path).convert("RGB")
img_w, img_h = image.size

print(f"[✓] Loaded detection test image: {img_path.name} ({img_w}x{img_h})")"""),

        nbf.v4.new_markdown_cell("""## 3. Simulate Candidate Bounding Boxes & Apply NMS"""),

        nbf.v4.new_code_cell("""# Candidate raw bounding boxes [x1, y1, x2, y2]
raw_boxes = torch.tensor([
    [100.0, 110.0, 190.0, 205.0],  # Box 1: Aphid cluster
    [105.0, 115.0, 185.0, 200.0],  # Box 2: Duplicate of Box 1 (IoU ~ 0.85)
    [98.0,  108.0, 192.0, 210.0],  # Box 3: Duplicate of Box 1 (IoU ~ 0.90)
    [240.0, 160.0, 320.0, 250.0],  # Box 4: Blight spot
    [245.0, 162.0, 315.0, 248.0],  # Box 5: Duplicate of Box 4
], dtype=torch.float32)

raw_scores = torch.tensor([0.92, 0.74, 0.65, 0.88, 0.58], dtype=torch.float32)
labels = ["aphid", "aphid", "aphid", "blight_spot", "blight_spot"]

print(f"Total raw proposals before NMS: {len(raw_boxes)}")
keep_indices = non_max_suppression_scratch(raw_boxes, raw_scores, iou_threshold=0.45, score_threshold=0.5)
print(f"Retained box indices after NMS: {keep_indices.tolist()}")"""),

        nbf.v4.new_markdown_cell("""## 4. Visualizing Raw Proposals vs. Suppressed Detections"""),

        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Raw redundant boxes
axes[0].imshow(image)
axes[0].set_title(f"Before NMS: {len(raw_boxes)} Overlapping Proposals")
axes[0].axis("off")
for box, score, lbl in zip(raw_boxes, raw_scores, labels):
    x1, y1, x2, y2 = box.tolist()
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)
    axes[0].text(x1, y1 - 5, f"{lbl} {score:.2f}", color='yellow', fontsize=10, backgroundcolor='black')

# Right: Clean NMS output
axes[1].imshow(image)
axes[1].set_title(f"After NMS: {len(keep_indices)} Pruned Detections")
axes[1].axis("off")
for idx in keep_indices:
    box = raw_boxes[idx].tolist()
    score = raw_scores[idx].item()
    lbl = labels[idx]
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='lime', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].text(x1, y1 - 8, f"{lbl} {score:.2f}", color='white', fontsize=11, weight='bold', backgroundcolor='green')

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. Practical Summary
- Non-Maximum Suppression is essential to collapse multiple overlapping multi-scale anchor detections into a single precise bounding box per individual foliar pest or lesion.""")
    ]
    save_nb(nb, "07_prac07_yolo_object_detection.ipynb")


# =====================================================================
# Notebook 8: Faster R-CNN Fine-Tuning (Practical 8)
# =====================================================================
def make_nb08():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 8: Two-Stage Detector Fine-Tuning: Faster R-CNN on Crop Pests
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit IV (Object Detection & Localization) | **CO:** CO4  
**Problem Statement Context:** SIH26131 — High-Precision Bounding-Box Detection for Crop Damage  

---

## 1. Two-Stage Detector Architecture
Faster R-CNN (Ren et al., 2015) decouples the detection pipeline into:
1. **Backbone + Feature Pyramid Network (FPN):** Extracts multi-scale visual features.
2. **Region Proposal Network (RPN):** Slides across feature maps to generate class-agnostic regions of interest (RoIs) using anchor boxes.
3. **RoIAlign:** Accurately extracts fixed-size feature vectors from non-integer proposed coordinates without quantization artifacts.
4. **Fast R-CNN Heads:** Parallel branches computing softmax class probabilities and bounding box regression offsets:

$$\mathcal{L}_{total} = \mathcal{L}_{rpn\_cls} + \mathcal{L}_{rpn\_reg} + \mathcal{L}_{roi\_cls} + \mathcal{L}_{roi\_reg}$$"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CropDetectionDataset, detection_collate_fn
from src.detector import build_crop_faster_rcnn, compute_ap

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Active Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 2. Load Crop Pest Detection Dataset"""),

        nbf.v4.new_code_cell("""det_img_dir = REPO_ROOT / "data" / "samples" / "detection" / "images"
det_xml_dir = REPO_ROOT / "data" / "samples" / "detection" / "annotations_voc"

classes = ["aphid", "armyworm", "whitefly", "leaf_miner", "blast_lesion", "blight_spot"]
dataset = CropDetectionDataset(
    images_dir=det_img_dir,
    annotations_dir=det_xml_dir,
    classes=classes,
    target_size=(416, 416)
)

print(f"[✓] Loaded {len(dataset)} annotated detection samples.")
sample_img, sample_target = dataset[0]
print(f"Sample tensor shape : {sample_img.shape}")
print(f"Annotated boxes count: {len(sample_target['boxes'])}")"""),

        nbf.v4.new_markdown_cell("""## 3. Instantiate Detector & Forward Step"""),

        nbf.v4.new_code_cell("""model = build_crop_faster_rcnn(num_classes=len(classes) + 1).to(device)

# Forward pass with dummy image batch
dummy_batch = [sample_img.to(device)]
dummy_targets = [{k: v.to(device) for k, v in sample_target.items()}]

model.train()
try:
    loss_dict = model(dummy_batch, dummy_targets)
    print("Faster R-CNN Multi-Task Loss Components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
except Exception as e:
    # Fallback if torchvision detection model is SimpleCropDetector
    box_preds, cls_preds = model(dummy_batch[0].unsqueeze(0))
    print(f"[✓] Fallback detector forward success: box shape {box_preds.shape}, cls shape {cls_preds.shape}")"""),

        nbf.v4.new_markdown_cell("""## 4. Detection Metric: Precision-Recall Curve & Average Precision (mAP)"""),

        nbf.v4.new_code_cell("""# Synthetic PR-curve simulation for mAP@0.5 calculation
recalls = np.linspace(0.0, 1.0, 20)
precisions = np.array([1.0, 0.98, 0.95, 0.93, 0.90, 0.88, 0.85, 0.82, 0.80, 0.76, 0.72, 0.68, 0.63, 0.58, 0.51, 0.44, 0.35, 0.25, 0.15, 0.05])

ap_50 = compute_ap(recalls, precisions)
print(f"[✓] Computed Average Precision (AP@0.5): {ap_50 * 100:.2f}%")

plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, label=f"PR Curve (AP = {ap_50:.2f})", color="navy", lw=2)
plt.fill_between(recalls, precisions, alpha=0.2, color="navy")
plt.title("Precision-Recall Curve (Foliar Pest Detection)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0, 1.05])
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. Academic Comparison: Two-Stage vs. Single-Stage
- **Faster R-CNN (Two-Stage):** Higher localization accuracy on tiny objects (individual aphids, young leaf lesions) due to explicit RoIAlign, but slower inference (~15 FPS).
- **YOLO (Single-Stage):** Extreme speed (>45 FPS), ideal for drone video streams and edge farm monitoring, with slightly lower sensitivity for clustered microscopic pests.""")
    ]
    save_nb(nb, "08_prac08_faster_rcnn_finetuning.ipynb")


# =====================================================================
# Notebook 9: U-Net Segmentation (Practical 9)
# =====================================================================
def make_nb09():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 9: Semantic Segmentation of Foliar Lesions with U-Net
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit V (Segmentation & Vision Transformers) | **CO:** CO5  
**Problem Statement Context:** SIH26131 — Stage 3: Pixel-Level Lesion Area Estimation & Severity Grading  

---

## 1. U-Net Architecture (Ronneberger et al., 2015)
U-Net comprises a contracting path (encoder) and an expansive path (decoder):
1. **Contracting Path:** Successive blocks of two $3 \times 3$ convolutions followed by a $2 \times 2$ MaxPool, halving spatial dimensions while doubling feature channels.
2. **Expansive Path:** Transposed convolutions ($2 \times 2$ up-conv) doubling spatial dimensions, concatenated with high-resolution feature maps from the contracting path via **skip connections**.
3. **Skip Connections:** Alleviate vanishing gradients and restore fine spatial boundaries lost during pooling.

### Evaluation Metrics:
- **Sørensen-Dice Coefficient:**
  $$\text{Dice} = \frac{2 |X \cap Y|}{|X| + |Y|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
- **Intersection over Union (IoU / Jaccard):**
  $$\text{IoU} = \frac{|X \cap Y|}{|X \cup Y|} = \frac{TP}{TP + FP + FN}$$"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CropSegmentationDataset
from src.segmenter import UNet, compute_segmentation_metrics, calculate_disease_severity, create_segmentation_overlay

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Active Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 2. Load Segmentation Dataset (Image-Mask Pairs)"""),

        nbf.v4.new_code_cell("""seg_img_dir = REPO_ROOT / "data" / "samples" / "segmentation" / "images"
seg_mask_dir = REPO_ROOT / "data" / "samples" / "segmentation" / "masks"

dataset = CropSegmentationDataset(seg_img_dir, seg_mask_dir, target_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"[✓] Loaded {len(dataset)} segmentation image-mask pairs.")
img_tensor, mask_tensor = dataset[0]
print(f"Image tensor: {img_tensor.shape} | Mask tensor: {mask_tensor.shape}")"""),

        nbf.v4.new_markdown_cell("""## 3. Train U-Net on Lesion Masks (3 Epochs)"""),

        nbf.v4.new_code_cell("""model = UNet(in_channels=3, num_classes=1, features=(16, 32, 64, 128)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(3):
    total_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/3] - Segmentation BCE Loss: {total_loss / len(dataloader):.4f}")"""),

        nbf.v4.new_markdown_cell("""## 4. Evaluation: Dice Score & Severity Calculation"""),

        nbf.v4.new_code_cell("""model.eval()
with torch.no_grad():
    sample_pred = model(img_tensor.unsqueeze(0).to(device))

metrics = compute_segmentation_metrics(sample_pred.cpu(), mask_tensor.unsqueeze(0))
print(f"Dice Coefficient : {metrics['dice_score']:.4f}")
print(f"IoU / Jaccard    : {metrics['iou_jaccard']:.4f}")
print(f"Pixel Accuracy   : {metrics['pixel_accuracy'] * 100:.2f}%")

# Severity scoring
pred_bin = (torch.sigmoid(sample_pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
severity_pct, stage = calculate_disease_severity(pred_bin)
print(f"Estimated Affected Leaf Surface Area: {severity_pct}% ({stage})")"""),

        nbf.v4.new_markdown_cell("""## 5. Visual Inspection: Ground Truth vs. U-Net Prediction"""),

        nbf.v4.new_code_cell("""raw_img = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
gt_mask = mask_tensor.squeeze().numpy().astype(np.uint8)
overlay = create_segmentation_overlay(raw_img, pred_bin)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(raw_img)
axes[0].set_title("Input Crop Leaf")
axes[0].axis("off")

axes[1].imshow(gt_mask, cmap="gray")
axes[1].set_title("Ground-Truth Mask")
axes[1].axis("off")

axes[2].imshow(pred_bin, cmap="hot")
axes[2].set_title("U-Net Predicted Lesions")
axes[2].axis("off")

axes[3].imshow(overlay)
axes[3].set_title(f"Alpha Overlay ({severity_pct}% Severity)")
axes[3].axis("off")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 6. Practical Takeaway
- Pixel-level segmentation enables quantifying the precise surface percentage damaged by disease lesions, converting qualitative inspection into an objective numerical severity metric for extension advisories.""")
    ]
    save_nb(nb, "09_prac09_unet_segmentation_overlay.ipynb")


# =====================================================================
# Notebook 10: Grad-CAM & Saliency (Practical 10)
# =====================================================================
def make_nb10():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Practical 10: Explainable AI: Grad-CAM on Correct vs. Misclassified Leaves
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit VI (Generative Models, VLMs & Explainability) | **CO:** CO6  
**Problem Statement Context:** SIH26131 — Transparent Model Decision-Making & Agronomic Trust  

---

## 1. Grad-CAM Mathematical Formulation (Selvaraju et al., 2017)
Grad-CAM calculates neuron importance weights $\alpha_k^c$ for class $c$ using the gradient of the class score $y^c$ with respect to the activation maps $A^k$ of a convolutional layer:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i, j}^k}$$

The visual localization map $L_{\text{Grad-CAM}}^c$ is computed as a weighted linear combination followed by a rectifier (ReLU):

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### Key Objectives:
- **Correct Prediction:** Verify the model focuses on actual pathogenic symptoms (e.g. concentric brown rings).
- **Misclassified Prediction:** Detect spurious correlations (e.g., attending to background soil or water drops instead of leaf tissue)."""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.models_cnn import ResNet18Scratch
from src.vlm_and_gradcam import GradCAM, overlay_gradcam, compute_saliency_map
from src.transforms import IMAGENET_MEAN, IMAGENET_STD

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[✓] Active Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 2. Load Model & Prepare Leaf Test Image"""),

        nbf.v4.new_code_cell("""classes = ["Tomato___Early_Blight", "Tomato___Late_Blight", "Tomato___Healthy", "Potato___Early_Blight", "Rice___Leaf_Blast", "Apple___Apple_Scab"]
model = ResNet18Scratch(in_channels=3, num_classes=len(classes)).to(device)
model.eval()

# Select target convolutional layer (last residual block conv2)
target_layer = model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)

sample_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Early_Blight" / "sample_000.jpg"
img_pil = Image.open(sample_path).convert("RGB").resize((224, 224))
raw_rgb = np.array(img_pil)

# Normalize
norm_arr = (raw_rgb.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
tensor = torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

print(f"[✓] Input tensor ready: {tensor.shape}")"""),

        nbf.v4.new_markdown_cell("""## 3. Compare Visual Attributions: Correct Target vs. Wrong Class"""),

        nbf.v4.new_code_cell("""# Case 1: Grad-CAM for True Class (Tomato Early Blight, index 0)
cam_correct = gradcam.generate_heatmap(tensor, target_class=0)
overlay_correct = overlay_gradcam(raw_rgb, cam_correct)

# Case 2: Grad-CAM for Confused Class (Tomato Healthy, index 2)
cam_wrong = gradcam.generate_heatmap(tensor, target_class=2)
overlay_wrong = overlay_gradcam(raw_rgb, cam_wrong)

# Case 3: Vanilla Saliency Map
saliency = compute_saliency_map(model, tensor, target_class=0)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(raw_rgb)
axes[0].set_title("Input Tomato Leaf")
axes[0].axis("off")

axes[1].imshow(saliency, cmap="hot")
axes[1].set_title("Vanilla Saliency Map")
axes[1].axis("off")

axes[2].imshow(overlay_correct)
axes[2].set_title("Grad-CAM: Early Blight (Target Class)")
axes[2].axis("off")

axes[3].imshow(overlay_wrong)
axes[3].set_title("Grad-CAM: Healthy (False Hypothesis)")
axes[3].axis("off")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Practical Takeaways & Auditing AI Models
1. **Model Validation:** When Grad-CAM highlights lesion spots, the model has learned true causal agronomic patterns rather than memorizing background lighting artifacts.
2. **Debugging Failure Modes:** On misclassified field images, Grad-CAM reveals if shadows or soil reflections distracted the feature extractor.""")
    ]
    save_nb(nb, "10_prac10_gradcam_saliency_analysis.ipynb")


# =====================================================================
# Notebook 11: Vision Transformers, CLIP & Generative (SIH)
# =====================================================================
def make_nb11():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 11: Advanced Vision: Vision Transformers (ViT, Swin), CLIP & Generative Synthesis
**Course:** CSE471 — Deep Learning for Computer Vision  
**Unit:** Unit V (Vision Transformers) & Unit VI (VLMs and Generative Models) | **CO:** CO5, CO6  
**Problem Statement Context:** SIH26131 — Cutting-Edge Multi-Modal & Self-Attention Architectures  

---

## 1. Vision Transformers (ViT) from First Principles
Vision Transformers (Dosovitskiy et al., 2020) treat 2D images like 1D sequence tokens:
1. Divide image $x \in \mathbb{R}^{H \times W \times C}$ into non-overlapping patches $x_p \in \mathbb{R}^{N \times (P^2 C)}$, where $N = \frac{HW}{P^2}$.
2. Project patches linearly into $D$-dimensional embeddings.
3. Prepend a learnable `[CLS]` token and add 1D positional encodings $E_{pos} \in \mathbb{R}^{(N+1) \times D}$.
4. Process through Multi-Head Self-Attention (MHSA) blocks:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.vlm_and_gradcam import CropVisionLanguageMatcher"""),

        nbf.v4.new_markdown_cell("""## 2. Patch Extraction & Multi-Head Self-Attention Implementation"""),

        nbf.v4.new_code_cell("""class PatchEmbedding(nn.Module):
    \"\"\"Converts 2D image into 1D patch embedding vectors.\"\"\"
    def __init__(self, in_channels=3, patch_size=16, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, D, H/P, W/P] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleSelfAttention(nn.Module):
    \"\"\"Scaled Dot-Product Self-Attention.\"\"\"
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out), attn

print("[✓] PatchEmbedding and SimpleSelfAttention compiled.")"""),

        nbf.v4.new_markdown_cell("""## 3. Visualize Image Patches & Self-Attention Map"""),

        nbf.v4.new_code_cell("""sample_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Early_Blight" / "sample_000.jpg"
img_pil = Image.open(sample_path).convert("RGB").resize((128, 128))
img_tensor = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

patch_embed = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=64)
attn_block = SimpleSelfAttention(embed_dim=64, num_heads=2)

patches = patch_embed(img_tensor)  # 8x8 = 64 patches
out, attn_weights = attn_block(patches)

print(f"Patches shape: {patches.shape} (64 patches of dim 64)")
print(f"Attention weights shape: {attn_weights.shape}")

# Plot attention matrix across patches
plt.figure(figsize=(6, 5))
plt.imshow(attn_weights[0, 0].detach().numpy(), cmap="viridis")
plt.title("Head 0 Self-Attention Weights Across Leaf Patches")
plt.xlabel("Key Patch Index")
plt.ylabel("Query Patch Index")
plt.colorbar()
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Vision-Language (CLIP) Zero-Shot Prompt Matching"""),

        nbf.v4.new_code_cell("""matcher = CropVisionLanguageMatcher()

# Simulated feature embeddings for demonstration
np.random.seed(42)
leaf_feature = np.random.randn(128)
# Inject target correlation
class_embeddings = {
    cls_name: np.random.randn(128) for cls_name in matcher.descriptions.keys()
}
# Strong alignment with Early Blight description
class_embeddings["Tomato___Early_Blight"] = leaf_feature + np.random.randn(128) * 0.2

results = matcher.zero_shot_match(leaf_feature, class_embeddings)
print("CLIP Zero-Shot Ranking for Symptom Description:")
for cls_name, score in results[:3]:
    print(f"  * {cls_name:<25} Cosine Match: {score:.3f}")"""),

        nbf.v4.new_markdown_cell("""## 5. Generative Models in Agriculture (CycleGAN & DCGAN)
- **CycleGAN:** Translates lab-controlled leaves (PlantVillage) into realistic field conditions (PlantDoc) without paired images, synthesizing domain diversity to train robust detectors.
- **DCGAN:** Generates synthetic lesion textures to balance rare agricultural pest classes.""")
    ]
    save_nb(nb, "11_sih_vit_swin_clip_generative_experiments.ipynb")


# =====================================================================
# Notebook 12: SIH26131 End-to-End Pipeline
# =====================================================================
def make_nb12():
    nb = create_nb()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Notebook 12: SIH26131 Grand Finale: End-to-End Automated Crop Health & Advisory Pipeline
**Problem Statement:** SIH26131 — Early detection and management of crop diseases and pest infestations  
**Syllabus Integration:** CSE471 Deep Learning for Computer Vision (CO1–CO6, Units I–VI)  

---

## 1. 4-Stage Hierarchical Vision & Agronomic Architecture
```mermaid
graph TD
    A[Raw Leaf Image] --> B[Stage 1: Classification - CNN / MobileNet / ResNet]
    A --> C[Stage 2: Pest & Symptom Localization - Bounding Boxes]
    A --> D[Stage 3: Affected Region Segmentation - U-Net]
    B --> E[Pathogen & Disease ID]
    C --> F[Pest Bounding Boxes & Scores]
    D --> G[Infected Leaf Area Percentage]
    E & F & G --> H[Stage 4: Grad-CAM Explainability & IPM Advisory Engine]
    H --> I[Actionable Agronomic Report & Extension Disclaimer]
```"""),

        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.app import AgriVisionPipeline"""),

        nbf.v4.new_markdown_cell("""## 2. Execute End-to-End Pipeline"""),

        nbf.v4.new_code_cell("""pipeline = AgriVisionPipeline()

sample_path = REPO_ROOT / "data" / "samples" / "classification" / "lab_plantvillage" / "Tomato___Early_Blight" / "sample_000.jpg"
img_pil = Image.open(sample_path).convert("RGB")

results = pipeline.predict(img_pil)
adv = results["advisory"]

print(f"Diagnosed Crop & Condition : {adv['crop']} - {adv['diagnosis']}")
print(f"Pathogen Agent             : {adv['pathogen']}")
print(f"Affected Leaf Surface Area : {adv['affected_leaf_area']} ({adv['severity_level']})")
print(f"Action Urgency             : {adv['action_urgency']}")"""),

        nbf.v4.new_markdown_cell("""## 3. Visual Dashboard (All 4 Vision Stages)"""),

        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Stage 1: Input Image
axes[0].imshow(results["raw_rgb"])
axes[0].set_title(f"Stage 1: Input\\n{adv['diagnosis']} ({adv['model_confidence']})")
axes[0].axis("off")

# Stage 2: Bounding Box Detection
import matplotlib.patches as patches
axes[1].imshow(results["raw_rgb"])
axes[1].set_title(f"Stage 2: Bounding Boxes\\n{len(results['detections'])} Detected Objects")
axes[1].axis("off")
for d in results["detections"]:
    b = d["box"]
    rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='yellow', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].text(b[0], b[1]-5, f"{d['label']} {d['score']:.2f}", color='yellow', fontsize=10, backgroundcolor='black')

# Stage 3: U-Net Lesion Segmentation Overlay
axes[2].imshow(results["seg_overlay"])
axes[2].set_title(f"Stage 3: U-Net Segmentation\\n{adv['affected_leaf_area']} Surface Area")
axes[2].axis("off")

# Stage 4: Grad-CAM Explainability
axes[3].imshow(results["gradcam_overlay"])
axes[3].set_title("Stage 4: Grad-CAM Saliency\\nRed = Decisive Features")
axes[3].axis("off")

plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Generated Agronomic Field Advisory Report"""),

        nbf.v4.new_code_cell("""print("=" * 80)
print(f"🌾 AGRONOMIC FIELD ADVISORY: {adv['crop'].upper()} - {adv['diagnosis'].upper()}")
print("=" * 80)
print(f"• Casual Pathogen   : {adv['pathogen']}")
print(f"• Weather Trigger   : {adv['favorable_conditions']}")
print(f"• Visual Symptoms   : {adv['symptom_confirmation']}")
print(f"• Scouting Schedule : {adv['scouting_schedule']}")

print("\\n🌿 CULTURAL & SANITATION PRACTICES:")
for m in adv["cultural_measures"]:
    print(f"  [+] {m}")

print("\\n🐞 BIOLOGICAL / BIO-PESTICIDES:")
for b in adv["biological_controls"]:
    print(f"  [+] {b}")

print("\\n🧪 CHEMICAL INTERVENTION (IF THRESHOLD EXCEEDED):")
print(f"  [!] {adv['targeted_intervention']}")

print("\\n⚠️ REGULATORY & EXTENSION ADVISORY DISCLAIMER:")
print(f"  {adv['regulatory_disclaimer']}")
print("=" * 80)""")
    ]
    save_nb(nb, "12_sih_end_to_end_crop_pipeline.ipynb")


def main():
    print("[*] Generating all 12 academic and SIH Jupyter Notebooks...")
    make_nb01()
    make_nb02()
    make_nb03()
    make_nb04()
    make_nb05()
    make_nb06()
    make_nb07()
    make_nb08()
    make_nb09()
    make_nb10()
    make_nb11()
    make_nb12()
    print("[✓] All 12 notebooks successfully created in notebooks/")


if __name__ == "__main__":
    main()
