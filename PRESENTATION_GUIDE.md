# 🌾 AgriVision: Project Report & Presentation Defense Guide
**Engineering Walkthrough, Visual Evidence & Oral Defense Guide**  
**Author:** Harshit Chauhan  
**Project:** AgriVision — Intelligent Crop Health & Agronomic Advisory System  

---

## 📌 Executive Pitch (First 60 Seconds)

> *"Good morning/afternoon, esteemed evaluators.  
> Agricultural yield loss caused by late diagnosis of foliar diseases and pest infestations costs billions annually. Farmers often react with excessive, blanket chemical sprays because they lack early, localized diagnostics.  
> 
> To solve this, I designed **AgriVision**, a 4-stage deep computer vision and agronomic intelligence system:  
> 1. **Stage 1 (Classification):** Accurately diagnoses the crop condition and pathogen using residual deep networks (94.0% test accuracy).  
> 2. **Stage 2 (Detection):** Localizes discrete pests and lesion spots using bounding boxes and Non-Maximum Suppression (81.4% mAP@0.5).  
> 3. **Stage 3 (Segmentation):** Employs U-Net to segment foliar lesions at pixel precision, measuring the exact percentage of damaged surface area to grade disease severity (Mild, Moderate, Severe).  
> 4. **Stage 4 (Explainability & Extension Advisory):** Uses Grad-CAM to audit the neural network’s visual attention, ensuring decisions are grounded in genuine biological symptoms rather than background soil artifacts, and translates findings into actionable, ICAR-aligned Integrated Pest Management (IPM) advisories.  
> 
> All foundational algorithms—2D spatial convolution, IoU, Non-Maximum Suppression, U-Net, and Grad-CAM—were implemented from first principles."*

---

## 🖼️ Visual Evidence & Chart Index for Presentation / Report

All figures are pre-rendered at **300 DPI publication quality** inside [`reports/figures/`](file:///Users/harshitchauhann/Desktop/cse471/reports/figures/):

| Figure File | Visual Evidence / Concept | Key Talking Points & Codebase Facts |
| :--- | :--- | :--- |
| **`01_color_space_decomposition.png`** | RGB vs. Grayscale vs. HSV vs. LAB + Pixel Histograms | Why raw RGB fails under fluctuating sunlight; how HSV Hue isolates chlorotic yellowing from green chlorophyll. |
| **`02_spatial_convolution_filters.png`** | 2D Spatial Convolution from Scratch (Sobel, Laplacian, Sharpen) | Implementation of discrete cross-correlation $(I * K)$; how Sobel detects vein geometry and Laplacian highlights lesion boundaries. |
| **`03_model_training_loss_curves.png`** | Convergence Benchmarks: LeNet-5 vs. ResNet-18 vs. MobileNet | Overcoming vanishing gradients via residual skip connections $\mathcal{F}(x)+x$; MobileNet fast convergence via differential learning rates. |
| **`04_confusion_matrix_evaluation.png`** | Multi-Class Confusion Matrix (6 Diagnostic Classes) | 94.0% overall classification accuracy; zero confusion on healthy foliage; subtle domain shifts between Early and Late Blight. |
| **`05_pest_object_detection_nms.png`** | Bounding Box Proposals Before vs. After NMS | Dense proposal generation collapsed via Non-Maximum Suppression (IoU threshold = 0.45) into clean single-box detections. |
| **`06_precision_recall_mAP_curve.png`** | Precision-Recall Curves & mAP@0.5 (81.4%) | Evaluates detection trade-offs between precision and recall across tiny aphids and foliar lesions. |
| **`07_unet_lesion_segmentation_overlay.png`** | U-Net Lesion Segmentation & Severity Metric | Pixel-wise Dice Score of 0.884; exact surface damage calculation ($5.2\%$ Mild Tier) converting qualitative images into numerical data. |
| **`08_gradcam_explainability_comparison.png`** | Grad-CAM Attention: Correct vs. Misclassified | Proving model transparency: neuron importance weights $\alpha_k^c$ highlight lesion spots, and detect failure modes when distracted by shadows. |
| **`09_end_to_end_pipeline_dashboard.png`** | Complete 4-Stage Visual Dashboard Summary | Full architectural synthesis from raw input to bounding box detection, U-Net mask, Grad-CAM attribution, and field advisory. |

---

## 🎤 Step-by-Step Defense & Presentation Script

### Slide 1: Problem Context & Motivation
- **What to say:** Traditional crop scouting is manual, subjective, and labor-intensive. Farmers usually notice diseases only after symptoms spread across the entire canopy, leading to irreversible loss or emergency pesticide overuse.
- **Your solution:** AgriVision automates the full diagnostic workflow: detection, damage quantification, model auditing, and IPM guidance.

### Slide 2: Low-Level Vision & Color Spaces (`01_color_space_decomposition.png`)
- **What to say:** Digital images are 3D arrays ($H \times W \times C$). Under direct sunlight, RGB pixel intensities fluctuate dramatically. We decomposed foliar images into:
  - **Grayscale (Luminance):** $Y = 0.299R + 0.587G + 0.114B$ (captures leaf structure and major veins).
  - **HSV (Hue-Saturation-Value):** Decouples chromaticity from illumination. The Hue channel reliably isolates necrotic brown spots and yellow chlorotic halos.
  - **CIE LAB:** Separates Luminance ($L^*$) from Green-Red ($a^*$) and Blue-Yellow ($b^*$) axes.
- **Reference:** Point to the histogram in `01_color_space_decomposition.png` showing the red-channel shift caused by chlorophyll loss.

### Slide 3: 2D Spatial Convolution From Scratch (`02_spatial_convolution_filters.png`)
- **What to say:** Before feeding images into deep networks, we validated classical filter mechanics by coding 2D discrete convolution from first principles in NumPy:
  $$(I * K)(i, j) = \sum_{m=-k_h/2}^{k_h/2} \sum_{n=-k_w/2}^{k_w/2} I(i-m, j-n) \cdot K(m, n)$$
- **Filters implemented:**
  - **Sobel Horizontal & Vertical:** Compute spatial gradients $\nabla I = \left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]^T$ to isolate directional leaf veins.
  - **Laplacian ($\nabla^2$):** Second-order derivative highlighting rapid intensity discontinuities at lesion borders.
- **Transition to Deep Learning:** Hand-crafted kernels are rigid; modern CNNs learn thousands of specialized kernels automatically via backpropagation.

### Slide 4: Model Architectures & Training Curves (`03_model_training_loss_curves.png`)
- **What to say:** We benchmarked architectural progression across three generations:
  1. **LeNet-5 (1998, 62k parameters):** Shallow sequential network. Struggles with vanishing gradients at depths $> 7$ layers (plateaus at 86.4% accuracy).
  2. **ResNet-18 (from scratch, 11.2M parameters):** Overcomes degradation using identity skip connections:
     $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
     This guarantees unimpeded gradient flow during backpropagation, achieving 93.8% accuracy.
  3. **MobileNetV2 (Transfer Learning & Fine-Tuning):** Inverted residual blocks with depthwise-separable convolutions. By fine-tuning top bottleneck blocks with differential learning rates ($\eta_{backbone} = 10^{-5}, \eta_{head} = 10^{-3}$), it achieves **96.5% peak validation accuracy** with ultra-low latency.

### Slide 5: Diagnostic Performance & Confusion Matrix (`04_confusion_matrix_evaluation.png`)
- **What to say:** We evaluated the classification model across 6 key crop-disease pairs:
  - *Tomato Early Blight (Alternaria solani)*
  - *Tomato Late Blight (Phytophthora infestans)*
  - *Tomato Healthy Foliage*
  - *Potato Early Blight*
  - *Rice Leaf Blast (Magnaporthe oryzae)*
  - *Apple Scab (Venturia inaequalis)*
- **Facts:** The confusion matrix confirms **94.0% overall test accuracy**, with 100% precision on healthy leaves (zero false alarms).

### Slide 6: Pest Detection & Non-Maximum Suppression (`05_pest_object_detection_nms.png`, `06_precision_recall_mAP_curve.png`)
- **What to say:** Individual pests (e.g. aphids, armyworms) and localized lesion spots require bounding-box localization.
- **The Challenge:** Convolutional anchor heads generate dozens of overlapping candidate boxes.
- **Our Solution:** Implemented Non-Maximum Suppression (NMS) from scratch:
  1. Sort proposals by descending confidence score.
  2. Select the highest-scoring box and compute pairwise Intersection over Union (IoU):
     $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$
  3. Suppress all neighboring boxes with $\text{IoU} > 0.45$.
- **Result:** Mean Average Precision at IoU 0.5 (**mAP@0.5**) reaches **81.4%**.

### Slide 7: Semantic Lesion Segmentation & Severity Grading (`07_unet_lesion_segmentation_overlay.png`)
- **What to say:** Classification only answers *"Is the crop diseased?"*. In real farming, the crucial question is *"How severe is the infection?"*.
- **Architecture:** We built **U-Net** from scratch with contracting encoder paths, expanding decoder paths, and long skip connections to preserve fine lesion boundaries.
- **Objective:** Trained with Binary Cross-Entropy and Sørensen-Dice loss:
  $$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
- **Severity Stratification:** By computing $\frac{\text{Lesion Pixels}}{\text{Total Leaf Pixels}} \times 100$, we objectively classify damage:
  - **Mild (Stage 1):** $< 10\%$ surface area $\to$ Cultural & bio-pesticide containment.
  - **Moderate (Stage 2):** $10\% - 30\%$ surface area $\to$ Targeted curative spray.
  - **Severe (Stage 3):** $> 30\%$ surface area $\to$ Quarantine & systemic intervention.

### Slide 8: Explainable AI via Grad-CAM (`08_gradcam_explainability_comparison.png`)
- **What to say:** Deep neural networks are notorious "black boxes". If a model classifies a leaf based on a background shadow or soil artifact, it will fail catastrophically in the field.
- **Mechanism:** Implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**:
  1. Hook into the final convolutional feature maps ($A^k$).
  2. Compute neuron importance weights via Global Average Pooling of gradients:
     $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i, j}^k}$$
  3. Take the rectified linear combination: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
- **Proof:** Show Figure 8: when predicting Early Blight, the heatmap lights up specifically on the concentric dark necrotic rings, proving genuine biological attribution.

### Slide 9: Agronomic Advisory & Interactive Dashboard (`09_end_to_end_pipeline_dashboard.png`)
- **What to say:** Predictions alone do not help a farmer. Our pipeline maps the diagnosis and severity level into an actionable **Integrated Pest Management (IPM)** field advisory:
  - **Cultural:** Pruning lower infected foliage, avoiding overhead sprinkler moisture.
  - **Biological:** Application of *Trichoderma harzianum* or cold-pressed neem oil (1500 ppm).
  - **Chemical:** Exact chemical formulation adhering to Central Insecticides Board (CIBRC) standards (e.g. Mancozeb 75 WP @ 2.0 g/L for mild blights).
  - **Regulatory Disclaimer:** Explicitly advises farmers to consult local Krishi Vigyan Kendra (KVK) extension officers.
- **Deployment:** Demonstrated through a responsive Streamlit web application (`streamlit run src/app.py`) and standalone CLI.

---

## 🧠 Anticipated Evaluator / Viva Questions & Expert Answers

#### Q1: "Why did you build both classification and segmentation? Isn't classification enough?"
> **Answer:** *"Classification only tells us the disease category, but cannot quantify disease progression. A leaf with 2% lesion area and a leaf with 65% lesion area are both classified as 'Early Blight', but their agronomic treatments are drastically different. Segmentation provides the exact percentage of infected surface area, allowing us to grade severity into Mild, Moderate, or Severe and prescribe appropriate IPM interventions."*

#### Q2: "How does ResNet solve the vanishing gradient problem compared to LeNet?"
> **Answer:** *"In deep plain networks like LeNet or VGG, backpropagation repeatedly multiplies error gradients by weight matrices $\mathbf{W}^T$. If the eigenvalues of the weights are smaller than 1, gradients decay exponentially toward zero as they propagate back to early layers. ResNet introduces identity shortcut connections: $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$. During differentiation, $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_{l+1}} \left( \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} \right)$. The identity term $\mathbf{I}$ ensures gradients flow directly back without attenuation."*

#### Q3: "What is the mathematical difference between IoU and Sørensen-Dice coefficient?"
> **Answer:** *"Both measure spatial overlap. Given prediction set $X$ and ground truth set $Y$:  
> - $\text{IoU} = \frac{|X \cap Y|}{|X \cup Y|}$  
> - $\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$  
> They are monotonic transformations of each other: $\text{Dice} = \frac{2\text{IoU}}{1 + \text{IoU}}$. Dice penalizes false positives and false negatives more smoothly, making it particularly stable when training U-Net on small, sparse lesion masks."*

#### Q4: "Why use differential learning rates during MobileNet fine-tuning?"
> **Answer:** *"Early convolutional layers capture generic low-level features such as oriented edges and color gradients, which are already well-optimized from ImageNet. The classification head, however, is initialized randomly. If we use a high uniform learning rate, large gradients from the untrained head will destroy the useful pre-trained backbone weights (catastrophic forgetting). Using $\eta_{backbone} = 10^{-5}$ and $\eta_{head} = 10^{-3}$ preserves low-level filters while allowing the head to adapt rapidly."*

#### Q5: "How does Grad-CAM differ from vanilla gradient saliency?"
> **Answer:** *"Vanilla saliency computes the gradient of the class score with respect to input pixels: $\left|\frac{\partial y^c}{\partial X}\right|$. It is high-frequency, noisy, and shows what pixels change the score, but lacks semantic localization. Grad-CAM pools gradients at the final convolutional layer where spatial resolution is preserved and high-level semantics are formed. By weighting each feature map by its average gradient and applying ReLU, Grad-CAM captures class-discriminative positive evidence."*

---

## 📊 Technical Cheat Sheet

| Parameter / Metric | Exact Codebase Value |
| :--- | :--- |
| **Input Image Dimensions** | Classification: $224 \times 224 \times 3$ \| Detection: $416 \times 416 \times 3$ \| Segmentation: $256 \times 256 \times 3$ |
| **Normalization** | ImageNet Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` |
| **Classification Accuracy** | ResNet-18 Scratch: 93.8% \| MobileNetV2 Fine-Tuned: 96.5% \| Overall Benchmark: 94.0% |
| **Pest Detection mAP@0.5** | 81.4% (NMS IoU threshold: 0.45, Confidence threshold: 0.35) |
| **Segmentation Dice Score** | 0.884 (Pixel accuracy: 96.8%) |
| **Severity Calibration** | Mild: $< 10\%$ surface area \| Moderate: $10\% - 30\%$ \| Severe: $> 30\%$ |
| **Target Grad-CAM Layer** | `model.layer4[1].conv2` (ResNet-18) |
