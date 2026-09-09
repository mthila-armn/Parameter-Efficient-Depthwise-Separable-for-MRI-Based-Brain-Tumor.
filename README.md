# 🧠 Parameter-Efficient-Depthwise-Separable-for-MRI-Based-Brain-Tumor.

![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow&logoColor=white)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-green)
![MRI](https://img.shields.io/badge/Modality-Brain%20MRI-blue)
![Classification](https://img.shields.io/badge/Task-4--Class%20Classification-purple)
![CNN](https://img.shields.io/badge/Architecture-Depthwise%20Separable%20CNN-blueviolet)
![Attention](https://img.shields.io/badge/Attention-SE%20%2B%20Spatial-teal)
![Parameters](https://img.shields.io/badge/Parameters-1.46M-informational)
![Model Size](https://img.shields.io/badge/Model%20Size-~5.6%20MB-lightgrey)
![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-97.27%25-brightgreen)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.05%25-success)
![TTA](https://img.shields.io/badge/TTA-5%20Views-yellow)

A parameter-efficient, attention-augmented depthwise-separable convolutional neural network designed for **four-class MRI-based brain tumor classification**.

The model combines lightweight convolutional operators with **Squeeze-and-Excitation (SE) channel attention**, **CBAM-style spatial attention**, **GELU activations**, and a compact **Global Average Pooling–Dense classification head**.

The framework is designed around an explicit parameter budget of approximately **1.46 million trainable parameters**, targeting scenarios where model footprint, computational cost, and reproducibility are important.

On the evaluated Brain Tumor MRI benchmark, PruDensNet achieves:

```text
Validation Accuracy : 97.27%
Test Accuracy       : 96.05%
Parameters          : ~1.46M
Model Size          : ~5.6 MB
```

The four diagnostic categories are:

```text
Glioma
Meningioma
Pituitary
No Tumor
```

---

# 🔎 Overview

MRI is widely used for non-invasive brain tumor assessment, but automated classification remains challenging because of:

- heterogeneous image characteristics,
- imaging artifacts,
- inter-tumor variability,
- computational constraints,
- data leakage risks,
- reproducibility concerns.

Many high-performing medical image classifiers rely on large CNNs, Transformer architectures, hybrid models, or ensembles.

PruDensNet instead explores whether a carefully designed lightweight CNN can provide strong classification performance under a constrained parameter budget.

The overall framework combines:

```text
MRI Images
    │
    ▼
Leakage-Aware Data Curation
    │
    ▼
Image Preprocessing
    │
    ▼
Curriculum-Based Augmentation
    │
    ▼
PruDensNet
    │
    ├── Depthwise-Separable Convolution
    ├── SE Channel Attention
    ├── Spatial Attention
    ├── GELU Activation
    └── GAP-Dense Head
    │
    ▼
Four-Class Softmax
    │
    ▼
5-View Test-Time Augmentation
    │
    ▼
Final Prediction
```

---

# ✨ Key Contributions

## 🪶 1. Parameter-Efficient Architecture

PruDensNet is designed around an explicit trainable-parameter target of:

```text
1,461,587 parameters
```

or approximately:

```text
1.46M trainables
~5.6 MB model footprint
```

The architecture prioritizes compact representation learning without relying on a large Transformer or conventional high-capacity CNN backbone.

---

## 🧩 2. Depthwise-Separable Convolutions

Standard convolution has a parameter cost approximately proportional to:

```text
k² × C × C'
```

where:

```text
k  = kernel size
C  = input channels
C' = output channels
```

Depthwise-separable convolution factorizes this operation into:

```text
Depthwise Convolution
        │
        ▼
Pointwise 1 × 1 Convolution
```

with parameter cost:

```text
k²C + CC'
```

This significantly reduces the number of trainable parameters and arithmetic operations.

---

## 🎯 3. Channel Attention

PruDensNet uses **Squeeze-and-Excitation attention** to emphasize diagnostically informative feature channels.

For feature map:

```text
U ∈ R^(H × W × C)
```

global average pooling produces:

```text
z_c =
1 / (HW)
Σ U_ijc
```

The excitation operation is:

```text
s = σ(W₂ δ(W₁z))
```

with reduction ratio:

```text
r = 16
```

The original features are recalibrated using:

```text
Û_ijc = U_ijc × s_c
```

This allows the model to strengthen informative channels while suppressing less useful responses.

---

## 📍 4. Spatial Attention

Channel attention is complemented by a **CBAM-style spatial attention mechanism**.

Average and maximum responses are aggregated across channels:

```text
M_avg
M_max
```

These feature maps are concatenated and processed using a:

```text
7 × 7 convolution
```

followed by sigmoid activation:

```text
S =
σ(
Conv_7×7
[
M_avg ; M_max
]
)
```

The feature representation is spatially recalibrated as:

```text
Û_ijc = U_ijc × S_ij
```

This mechanism helps the network focus on spatial regions carrying relevant tumor information.

---

# 🧠 GELU Activation

PruDensNet uses the **Gaussian Error Linear Unit (GELU)** instead of a conventional ReLU-only design.

The activation is expressed as:

```text
GELU(z)
=
0.5z
[
1 +
tanh(
sqrt(2/π)
(z + 0.044715z³)
)
]
```

GELU provides smooth nonlinear gating and supports stable gradient propagation through the lightweight network.

---

# 🏗️ PruDensNet Architecture

The model follows the high-level pipeline:

```text
Input
256 × 256 × 3
      │
      ▼
Gaussian Noise
σ = 0.1
      │
      ▼
Conv + BatchNorm + GELU
      │
      ▼
Conv + BatchNorm + GELU
      │
      ▼
Max Pooling
      │
      ▼
SepBlock
F₁ = 128
      │
      ▼
Max Pooling
      │
      ▼
SepBlock
F₂ = 320
      │
      ▼
Max Pooling
      │
      ▼
SepBlock
F₃ = 640
      │
      ▼
Max Pooling
      │
      ▼
Global Average Pooling
      │
      ▼
Dense 256
      │
      ▼
GELU
      │
      ▼
Dropout
p = 0.35
      │
      ▼
Softmax
      │
      ▼
4 Brain MRI Classes
```

Each SepBlock integrates:

```text
Depthwise Convolution
        +
Pointwise Convolution
        +
Batch Normalization
        +
GELU
        +
Channel Attention
        +
Spatial Attention
```

---

# 🔢 Feature Widths

The three main depthwise-separable stages use progressively increasing channel widths:

| Stage | Feature Width |
|---|---:|
| SepBlock 1 | `128` |
| SepBlock 2 | `320` |
| SepBlock 3 | `640` |

The final classification head uses:

```text
GAP
 ↓
Dense 256
 ↓
GELU
 ↓
Dropout 0.35
 ↓
Softmax
```

---

# 🔬 Task Definition

The study addresses **image-level four-class brain MRI classification**.

The prediction categories are:

| Class | Description |
|---|---|
| Glioma | Glioma tumor MRI |
| Meningioma | Meningioma tumor MRI |
| Pituitary | Pituitary tumor MRI |
| No Tumor | MRI without visible tumor |

The task is specifically image-level classification.

The dataset does **not** provide:

```text
Longitudinal Timeline Information
Tumor Staging
Clinical Time-to-Diagnosis Labels
```

Therefore, tumor detection in this work refers to identification of the tumor category from an MRI image rather than clinical early-stage detection.

---

# 🗂️ Data Curation Pipeline

PruDensNet incorporates a reproducibility-focused and leakage-aware data preparation pipeline.

```text
Raw Dataset
     │
     ▼
Automatic Source Discovery
     │
     ▼
Class Canonicalization
     │
     ▼
Near-Duplicate Removal
     │
     ▼
Training / Testing Detection
     │
     ▼
Stratified Training / Validation Split
     │
     ▼
Clean Dataset
```

---

# 🔍 Automatic Dataset Discovery

The data loader automatically searches candidate dataset roots.

A valid class root is recognized when a directory contains at least:

```text
2 class subdirectories
```

The pipeline can also detect common Brain Tumor MRI folder structures containing:

```text
Training/
Testing/
```

---

# 🏷️ Label Canonicalization

Different datasets may use inconsistent class names.

The pipeline normalizes synonymous labels into a canonical set.

Examples include:

```text
glioma_tumor
      ↓
glioma
```

```text
meningioma_tumor
      ↓
meningioma
```

```text
pituitary_tumor
      ↓
pituitary
```

and variants such as:

```text
notumor
no_tumor
no-tumor
      ↓
no tumor
```

This prevents artificial class fragmentation.

---

# 🧹 Near-Duplicate Removal

Technical duplicates and near-duplicate images may inflate performance when correlated samples occur across splits.

PruDensNet applies per-class near-duplicate filtering using an:

```text
8 × 8 grayscale average hash
```

The process is:

```text
Input Image
    │
    ▼
Convert to Grayscale
    │
    ▼
Resize to 8 × 8
    │
    ▼
Compute Mean Intensity
    │
    ▼
Generate Binary Average Hash
    │
    ▼
SHA-1 Digest
    │
    ▼
Duplicate-Key Check
    │
    ├── Existing Key → Skip
    │
    └── New Key      → Keep
```

This helps reduce technical duplication before splitting and training.

---

# ✂️ Dataset Splitting

The dataset contains predefined:

```text
Training/
Testing/
```

containers.

The predefined Testing set is retained as the final:

```text
Held-Out Test Set
```

and is not used for:

- model selection,
- early stopping,
- validation.

The original Training set is divided using:

```text
90% Training
10% Validation
```

with:

```text
Fixed Random Seed = 42
```

and class-stratified file-level sampling.

---

# ⚠️ Patient-Level Split Limitation

Patient identifiers are not available in the image-folder dataset.

Therefore, the pipeline guarantees:

```text
File-Level Disjointness
        +
Near-Duplicate Filtering
```

but cannot guarantee:

```text
Patient-Level Disjointness
```

The reported results should consequently be interpreted as image-level benchmark performance.

---

# 🖼️ Image Preprocessing

All images are resized to:

```text
256 × 256
```

and pixel values are rescaled to:

```text
[0, 1]
```

---

# 🔄 Standard Augmentation

The training pipeline applies moderate geometric and photometric augmentation.

## Geometric Transformations

```text
Rotation        : ±10°
Width Shift     : ≤ 8%
Height Shift    : ≤ 8%
Shear           : ≤ 6°
Zoom            : [0.8, 1.2]
Horizontal Flip : Enabled
```

## Photometric Transformation

```text
Brightness Scaling : [0.92, 1.10]
```

These transformations are intended to increase variation without severely altering anatomical image content.

---

# 🌗 CLAHE

Contrast Limited Adaptive Histogram Equalization is applied according to the curriculum schedule.

CLAHE operates on the:

```text
L channel
```

of the CIE-LAB color representation.

Conceptually:

```text
RGB Image
    │
    ▼
RGB → LAB
    │
    ▼
CLAHE on L Channel
    │
    ▼
LAB → RGB
```

The operation enhances local contrast while limiting excessive noise amplification.

---

# 🧽 Random Erasing

Random erasing replaces a randomly selected rectangular region with random values.

```text
Image
  │
  ▼
Random Rectangle
  │
  ▼
Replace Region with U(0,1) Noise
```

This encourages the network to avoid relying exclusively on small localized image regions.

---

# 🔀 MixUp

MixUp combines two aligned training examples.

For:

```text
(x, y)
and
(x', y')
```

with:

```text
λ ~ Beta(α, α)
α = 0.3
```

the mixed sample becomes:

```text
x̃ = λx + (1 - λ)x'
```

and:

```text
ỹ = λy + (1 - λ)y'
```

---

# ✂️ CutMix

CutMix replaces a rectangular region of one image using the corresponding region of another image.

The labels are mixed according to the retained image area.

```text
Image A
   +
Patch from Image B
        │
        ▼
Mixed Image
        │
        ▼
Area-Weighted Label
```

---

# 📚 Curriculum-Based Regularization

Instead of maintaining a fixed augmentation strength throughout training, PruDensNet uses an epoch-dependent curriculum.

The regularization schedule gradually:

```text
Starts Mild
    │
    ▼
Increases During Mid-Training
    │
    ▼
Reaches Strong Regularization
    │
    ▼
Tapers to Zero
```

The schedule coordinates:

- MixUp,
- CutMix,
- random erasing,
- CLAHE,
- label smoothing.

This design allows early epochs to focus on basic feature acquisition before stronger regularization is introduced.

---

# 🗓️ Curriculum Schedule

The manuscript defines the following broad training stages:

### Early Stage

```text
Epoch < 2
MixUp        : Off
CutMix       : Off
Random Erase : Off
Label Smooth : 0.05
```

### Early-to-Mid Stage

```text
Epoch 2–9
MixUp        : 0.15
CutMix       : 0.15
Random Erase : 0.08
CLAHE        : 0.06
Label Smooth : 0.05
```

### Main Regularization Stage

```text
Epoch 10–35
MixUp        : 0.25
CutMix       : 0.25
Random Erase : 0.12
CLAHE        : 0.08
Label Smooth : 0.03
```

### Late Stage

```text
Epoch ≥ 36
Strong Curriculum Regularizers → Off
Label Smoothing                → 0
```

---

# ⚖️ Class-Imbalance Reweighting

Class imbalance is handled using inverse-frequency weighting.

For class `k`:

```text
w_k =
N / (K n_k)
```

where:

```text
N   = total number of samples
K   = number of classes
n_k = samples belonging to class k
```

This gives minority classes greater contribution to the optimization objective without changing the underlying data distribution.

---

# 📉 Dynamic Label Smoothing

Instead of using fixed hard one-hot labels throughout training, PruDensNet applies dynamic label smoothing.

The smoothed target is:

```text
ỹ =
(1 - ε)y
+
ε/K
```

The smoothing strength changes according to the curriculum schedule.

The optimization objective remains categorical cross-entropy.

---

# ⚙️ Optimization

The model is trained using:

```text
Optimizer       : AdamW
Weight Decay    : 1e-4
Gradient Clip   : Global Norm ≤ 1.0
Base LR         : 3e-4
Minimum LR      : 1e-6
Warmup          : 10% of Steps
LR Decay        : Cosine
Batch Size      : 8
Maximum Epochs  : 50
Early Stopping  : Patience 12
```

---

# 📈 Warmup–Cosine Learning Rate

The optimization schedule combines:

```text
Linear Warmup
      │
      ▼
Cosine Decay
```

with:

```text
Base Learning Rate = 3 × 10⁻⁴
Minimum LR         = 1 × 10⁻⁶
```

Warmup improves stability during early mixed-precision training, while cosine decay provides smooth late-stage optimization.

---

# ✋ Gradient Clipping

Global gradient norm is constrained to:

```text
||g||₂ ≤ 1
```

to reduce optimization instability when strong augmentations are active.

---

# ⏹️ Early Stopping

Training proceeds for up to:

```text
50 epochs
```

with:

```text
Early-Stopping Patience = 12
```

based on validation accuracy.

The best-performing validation checkpoint is restored for final evaluation.

---

# ⚡ Mixed Precision

PruDensNet uses TensorFlow mixed precision.

```text
Internal Computation : float16
Classifier Output    : float32
```

The float32 output layer is retained to improve numerical stability during loss computation.

---

# 🧮 Parameter Target Padding

A distinctive component of the experimental methodology is **Parameter Target Padding — ParamPad**.

The target parameter budget is:

```text
P* = 1,461,587
```

If a model contains fewer than the required number of trainable parameters, a trainable vector is added:

```text
p ∈ R^(P* - P_base)
```

but its contribution to the forward pass is multiplied by zero:

```text
y' = y + 0 × Σ p_j
```

Therefore:

```text
Forward Function → Unchanged
Parameter Count  → Matched
```

This allows parameter-controlled architectural comparisons without altering model predictions.

---

# 🔬 Why ParamPad Is Used

Without capacity matching, a larger architecture may perform better simply because it contains more trainable parameters.

ParamPad is designed to separate:

```text
Architecture Effect
       from
Parameter-Count Effect
```

by enforcing approximately the same parameter budget during capacity-controlled comparisons.

---

# 🧪 Five-View Test-Time Augmentation

At inference time, predictions are averaged across:

```text
5 deterministic views
```

including:

```text
1. Identity
2. Horizontal Flip
3. +8° Rotation
4. -8° Rotation
5. Mild Gamma Adjustment
```

The final probability distribution is:

```text
p̃ =
1/T
Σ f(τ_t(x))
```

with:

```text
T = 5
```

The final class is:

```text
ŷ = argmax_k p̃_k
```

This reduces prediction variance for borderline cases.

---

# 💻 Experimental Environment

Experiments were conducted using:

```text
NVIDIA Tesla P100 GPU
Kaggle-style environment
TensorFlow mixed precision
```

Reproducibility controls include fixed random seeds for:

```text
Python
NumPy
TensorFlow
```

and single-process data loading to reduce nondeterministic behaviour.

---

# 📏 Evaluation Metrics

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Expected Calibration Error
- Brier Score

Macro and weighted averages are also used to summarize class-level performance.

---

# 🏆 Main Results

## Validation Performance

```text
Validation Samples  : 477
Validation Accuracy : 97.27%
Validation Loss     : 0.1027
```

The validation classification report is:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9688 | 0.9841 | 0.9764 | 126 |
| Meningioma | 0.9762 | 0.9462 | 0.9609 | 130 |
| No Tumor | 0.9881 | 0.9651 | 0.9765 | 86 |
| Pituitary | 0.9640 | 0.9926 | 0.9781 | 135 |
| **Macro Avg** | **0.9743** | **0.9720** | **0.9730** | **477** |
| **Weighted Avg** | **0.9729** | **0.9727** | **0.9727** | **477** |

---

# 🧪 Held-Out Test Results

Final evaluation is performed on:

```text
481 held-out test images
```

The overall test performance is:

```text
Test Accuracy : 96.05%
Macro F1      : 96.09%
Weighted F1   : 96.04%
```

The complete test classification report is:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Glioma | **0.9844** | **0.9844** | **0.9844** | 128 |
| Meningioma | 0.9375 | 0.9231 | 0.9302 | 130 |
| No Tumor | 0.9551 | 0.9770 | 0.9659 | 87 |
| Pituitary | 0.9632 | 0.9632 | 0.9632 | 136 |
| **Macro Avg** | **0.9600** | **0.9619** | **0.9609** | **481** |
| **Weighted Avg** | **0.9604** | **0.9605** | **0.9604** | **481** |

---

# 🎯 Per-Class Findings

### Glioma

```text
Precision : 98.44%
Recall    : 98.44%
F1        : 98.44%
```

Glioma shows the strongest overall test-set classification performance.

---

### Meningioma

```text
Precision : 93.75%
Recall    : 92.31%
F1        : 93.02%
```

Meningioma represents the most challenging class among the four test categories.

---

### No Tumor

```text
Precision : 95.51%
Recall    : 97.70%
F1        : 96.59%
```

The high recall indicates strong recognition of no-tumor MRI images within the evaluated dataset.

---

### Pituitary

```text
Precision : 96.32%
Recall    : 96.32%
F1        : 96.32%
```

---

# 📊 Architecture Comparison

The manuscript evaluates PruDensNet against a broad range of modern architecture families, including:

```text
MobileNetV4
EfficientNetV2
ConvNeXt V2
MaxViT
CoAtNet
Swin Transformer V2
DeiT III
Vision Transformer
RegNetY
LeViT
EfficientFormerV2
MobileViT v2
TinyViT
FocalNet
VAN
PVTv2
NextViT
EdgeNeXt
PoolFormer
XCiT
RepVGG
NFNet
GhostNetV2
ConvMixer
MLP-Mixer
ResNet-RS
InceptionNeXt
MobileOne
HorNet
```

Under the capacity-controlled comparison reported in the manuscript, PruDensNet obtains the strongest accuracy value in the model-comparison table.

---

# ⚡ Computational Characteristics

The deployment-oriented comparison reports:

| Model | Parameters | MACs | FLOPs | Peak GPU Memory | Latency |
|---|---:|---:|---:|---:|---:|
| **PruDensNet** | **1.46159M** | 3.68751G | 7.37503G | 4690.53 MB | 12.031 ms |
| MobileNetV4-T | 3.7740M | **0.2410G** | **0.4819G** | **177.025 MB** | **4.6233 ms** |
| EfficientNetV2-B0 | 7.1397M | 0.8468G | 1.6936G | 193.530 MB | 8.9015 ms |
| ConvNeXt V2-Tiny | 28.6355M | 5.8193G | 11.6386G | 292.222 MB | 8.3387 ms |
| Swin Transformer V2-Tiny | 28.3472M | 4.3709G | 8.7418G | 522.529 MB | 14.8396 ms |
| RegNetY-8GF | 39.1801M | 10.3987G | 20.7974G | 565.389 MB | 11.4717 ms |
| MaxViT-Tiny | 30.9165M | 5.3330G | 10.6660G | 430.379 MB | 25.2587 ms |

These measurements were reported on an:

```text
NVIDIA Tesla P100
```

The table also highlights an important distinction:

```text
Small Parameter Count
        ≠
Lowest FLOPs / Memory / Latency
```

PruDensNet is highly parameter-efficient, but MobileNetV4-T and several other architectures report lower raw computational cost or latency in this hardware benchmark.

---

# 📐 Calibration and Reliability

The manuscript reports calibration metrics for both validation and test sets.

| Split | Samples | ECE ↓ | Brier Score ↓ |
|---|---:|---:|---:|
| Validation | 477 | **0.009690** | **0.058287** |
| Test | 481 | **0.021510** | **0.070982** |

Reliability diagrams compare:

```text
Mean Predicted Confidence
           vs.
Observed Accuracy
```

with the diagonal representing ideal calibration.

---

# ⚠️ Calibration Note

The manuscript contains an internal wording inconsistency regarding calibration.

The experimental results explicitly provide:

```text
ECE
Brier Score
Reliability Diagrams
```

for the validation and held-out test sets.

However, the limitations section later states that calibration or reliability analyses beyond label smoothing are not reported.

For this README, the numerical calibration results presented in the experimental section are retained, while further external calibration validation remains an important future direction.

---

# 📈 Training Behaviour

Across the reported training curves:

```text
Accuracy
   ↑
increases steadily
```

while:

```text
Loss
   ↓
decreases and stabilizes
```

toward the end of training.

The manuscript interprets these curves as evidence of convergence with limited visible overfitting under the selected optimization and curriculum-regularization strategy.

---

# 🧹 Leakage-Aware Evaluation

The framework incorporates several protections against common benchmark leakage problems:

```text
Canonical Label Mapping
        +
Per-Class Near-Duplicate Removal
        +
Predefined Held-Out Testing Set
        +
Stratified File-Level Train/Val Split
```

However, because patient identifiers are unavailable, the study does not claim guaranteed patient-wise disjointness.

---

# ♻️ Reproducibility

The pipeline includes several reproducibility controls:

- fixed random seed,
- deterministic file-level split,
- canonical class labels,
- near-duplicate filtering,
- single-process data loading,
- mixed-precision configuration,
- best-checkpoint restoration,
- predefined held-out test set.

These choices are intended to reduce variation caused by data preparation and training nondeterminism.

---

# 💡 Key Findings

The study supports several main observations.

### 1. A Small CNN Can Achieve Strong Accuracy

PruDensNet achieves:

```text
Validation Accuracy : 97.27%
Test Accuracy       : 96.05%
```

with approximately:

```text
1.46M Parameters
```

---

### 2. Efficient Operators and Attention Can Be Combined

The architecture combines:

```text
Depthwise-Separable Convolution
              +
SE Channel Attention
              +
Spatial Attention
```

to preserve representational power while controlling parameter count.

---

### 3. Training Strategy Is a Major Part of the Framework

Performance is not attributed to architecture alone.

The full pipeline also uses:

```text
MixUp
CutMix
Random Erasing
CLAHE
Dynamic Label Smoothing
Class Weighting
AdamW
Warmup-Cosine Decay
Gradient Clipping
Mixed Precision
TTA
```

---

### 4. Data Hygiene Is Explicitly Addressed

The pipeline attempts to reduce technical leakage through:

```text
Near-Duplicate Removal
        +
Canonicalization
        +
Held-Out Testing
```

rather than relying only on a random image split.

---

### 5. Parameter Efficiency Does Not Mean Minimum Runtime Cost

Although PruDensNet contains fewer trainable parameters than the principal hardware-comparison baselines, its reported MACs, GPU-memory measurement, and inference latency are not the lowest in the table.

This highlights the difference between:

```text
Parameter Efficiency
        and
Hardware Efficiency
```



**a compact attention-augmented depthwise-separable CNN for reproducible four-class MRI brain tumor classification under an explicit parameter budget.**
