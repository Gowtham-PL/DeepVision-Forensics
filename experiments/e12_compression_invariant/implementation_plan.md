# Implementation Plan: Experiment E12 — Compression-Invariant Feature Regularization & Domain-Conditioned Adaptation

## Executive Summary

Experiment E12 directly targets the core failure mechanism discovered in **E10-A Diagnostic** and audited in **E11**:
When real camera photos undergo WhatsApp / social-media transmission (bilinear downsampling followed by 4:2:0 chroma-subsampled JPEG compression), high-frequency spatial gradients and FFT components are severely attenuated or distorted. Under a frozen dual-branch backbone, the extracted feature embeddings for compressed real images shift into regions of the 1792-dimensional embedding space that overlap with AI-generated features, causing false alarm inflation (FPR rising from 5.88% in E6-C to 17.65% in E10-B).

Rather than unfreezing the backbone (which risks catastrophic forgetting and clean performance collapse), **Experiment E12** investigates two principled, lightweight adaptation strategies:
1. **E12-A: Compression-Invariant Feature Regularization (Primary Recommended Approach)**:
   Extracts multi-view attention-aggregated feature representations ($h \in \mathbb{R}^{1792}$) for paired clean and compressed versions of each training image, penalizing representation drift via a normalized cosine feature-consistency loss ($\mathcal{L}_{cons}$) alongside standard binary cross-entropy classification.
2. **E12-B: Domain-Conditioned Normalization (Alternative Exploration)**:
   Evaluates lightweight sample-wise LayerNorm or bottleneck feature modulation directly on the aggregated multi-view embedding $h_{agg}$ before classification, mitigating domain covariance shift without altering backbone weights.

---

## 1. User Review & Safety Declarations

> [!IMPORTANT]
> **STRICT PROTOCOL COMPLIANCE DECLARATIONS**:
> 1. **Zero Training on WhatsApp**: The 67-image WhatsApp benchmark ($N=67$: 34 Real, 33 AI) remains **permanently frozen** and isolated. It will NEVER be seen during training, validation, early stopping, checkpoint selection, or threshold tuning.
> 2. **Evaluation Protocol**: Training uses ONLY the official E5 train split ($N=23,165$). Checkpoint selection is governed ONLY by the 100% clean E5 validation split ($N=5,775$). WhatsApp evaluation will be strictly **single-shot** after model freezing.
> 3. **Production Isolation**: Zero modifications to production backend (`backend/`), frontend (`frontend/`), models (`models/`), or the production threshold ($\tau=0.50$). Production remains safely running **E6-C**.
> 4. **No Git Commits or Pushes**: All code, configurations, logs, and checkpoints will be stored exclusively in `experiments/e12_compression_invariant/`.
> 5. **Training Execution**: NO training will take place until the user explicitly reviews and approves this implementation plan.

---

## 2. Technical Architecture & Parameter Analysis

### Base Architecture (Unchanged):
- **Model**: `MultiViewE5Model` (5 views: 1 global resized $224 \times 224$ + 4 corner crops $224 \times 224$).
- **Spatial Backbone**: EfficientNet-B3 (output dim: 1536).
- **Frequency Backbone**: Dual-Stream FFT CNN (output dim: 256).
- **Fused View Feature**: Concatenation $[e_{spatial}; e_{freq}] \in \mathbb{R}^{1792}$.
- **View-Attention Module**: Multi-view learned attention head:
  $$\text{score}_v = \mathbf{w}_2^\top \tanh(\mathbf{W}_1 e_v + \mathbf{b}_1) + b_2, \quad \alpha_v = \frac{\exp(\text{score}_v)}{\sum_{k=1}^5 \exp(\text{score}_k)}$$
  $$\mathbf{W}_1 \in \mathbb{R}^{128 \times 1792}, \quad \mathbf{w}_2 \in \mathbb{R}^{1 \times 128}$$
- **Aggregated Representation**:
  $$h_{agg} = \sum_{v=1}^5 \alpha_v e_v \in \mathbb{R}^{1792}$$
- **Classification Head**:
  $$\text{Linear}(1792 \to 512) \to \text{BatchNorm1d}(512) \to \text{LeakyReLU}(0.2) \to \text{Dropout}(0.5) \to \text{Linear}(512 \to 128) \to \text{LeakyReLU}(0.2) \to \text{Dropout}(0.5) \to \text{Linear}(128 \to 1)$$

---

### E12-A: Compression-Invariant Feature Regularization Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │  Input Source Image x_i (E5 Training Manifest, N=23K)  │
                       └───────────────────┬────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Clean Branch (x_i^clean)                    Compressed Branch (x_i^comp)
          [100% Uncompressed Image]                   [Online Resize + JPEG 4:2:0]
                    │                                             │
          Generate 5 Views (224x224)                    Generate 5 Views (224x224)
                    │                                             │
          Frozen Base Model (f)                         Frozen Base Model (f)
          [torch.no_grad()]                             [torch.no_grad()]
                    │                                             │
          View Embeddings (5 x 1792)                    View Embeddings (5 x 1792)
                    │                                             │
          Trainable View-Attention                      Trainable View-Attention
                    │                                             │
                    ▼                                             ▼
         h_clean in R^1792                             h_comp in R^1792
                    │                                             │
                    ├───────────────────────┬─────────────────────┤
                    │                       │                     │
                    ▼                       ▼                     ▼
          Trainable Classifier     L_cons (Cosine Drift)  Trainable Classifier
                    │               [1 - cos(h_c, h_comp)]        │
                    ▼                                             ▼
            Logits y_hat_clean                            Logits y_hat_comp
                    │                                             │
                    └───────────────────────┬─────────────────────┘
                                            ▼
                                  L_cls (BCE on Both)
                                            │
                                            ▼
                               Total Loss = L_cls + lambda * L_cons
```

#### Trainable vs. Frozen Parameter Breakdown:
| Module | Layer Details | E12-A Status | Parameters |
| :--- | :--- | :---: | :---: |
| **Spatial Backbone** | EfficientNet-B3 convolutional blocks | **FROZEN** (`requires_grad=False`) | 10,696,232 |
| **Frequency Backbone** | FFT 6-channel ConvNet blocks | **FROZEN** (`requires_grad=False`) | 454,624 |
| **View Attention** | Linear(1792, 128) + Linear(128, 1) | **TRAINABLE** (`requires_grad=True`) | **229,633** |
| **Classifier** | Linear(1792, 512) + BN + Linear(512, 128) + Linear(128, 1) | **TRAINABLE** (`requires_grad=True`) | **984,833** |
| **Total Model Parameters** | - | - | **12,365,322** |
| **Total Trainable Parameters** | View-Attention + Classifier | **9.82% of model** | **1,214,466** |
| **Total Frozen Parameters** | EfficientNet-B3 + Frequency CNN | **90.18% of model** | **11,150,856** |

---

### E12-B: Domain-Conditioned Normalization Architecture

In E12-B, the network architecture remains identical except that directly before feeding $h_{agg}$ into the classifier, a lightweight **LayerNorm** or **Feature Standardization** layer is applied:
$$h_{norm} = \text{LayerNorm}(1792)(h_{agg}) = \frac{h_{agg} - \mu_{h}}{\sqrt{\sigma_{h}^2 + \epsilon}} \odot \gamma + \beta$$
- **Rationale**: When social-media compression attenuates high frequencies, the $L_2$ norm and scale of the 1792-dim representation shrinks or shifts. Standard batch normalization inside the classifier uses frozen running statistics computed during training. LayerNorm normalizes across the feature dimension per sample independently of batch statistics, forcing clean and compressed features into identical scale and distribution.
- **Additional Trainable Parameters**: $\gamma \in \mathbb{R}^{1792}, \beta \in \mathbb{R}^{1792} \implies \mathbf{3,584\text{ parameters}}$.
- **Total Trainable Parameters in E12-B**: $1,214,466 + 3,584 = \mathbf{1,218,050\text{ parameters}}$.

---

## 3. Loss Functions & Mathematical Formulation

### 1. Supervised Classification Loss ($\mathcal{L}_{cls}$):
Binary Cross-Entropy with Logits computed symmetrically over both clean and compressed views:
$$\mathcal{L}_{cls}(x_i^{clean}, x_i^{comp}, y_i) = \frac{1}{2} \left[ \ell_{BCE}(\hat{y}_i^{clean}, y_i) + \ell_{BCE}(\hat{y}_i^{comp}, y_i) \right]$$
where:
$$\ell_{BCE}(\hat{y}, y) = - \left[ y \log \sigma(\hat{y}) + (1 - y) \log (1 - \sigma(\hat{y})) \right]$$

### 2. Compression-Invariant Feature Consistency Loss ($\mathcal{L}_{cons}$):
Normalized cosine distance between the aggregated multi-view representation of the clean image $h_i^{clean}$ and its compressed counterpart $h_i^{comp}$:
$$\mathcal{L}_{cons}(h_i^{clean}, h_i^{comp}) = 1 - \frac{h_i^{clean} \cdot h_i^{comp}}{\|h_i^{clean}\|_2 \, \|h_i^{comp}\|_2}$$

Equivalently, in terms of normalized $L_2$ Euclidean distance:
$$\left\| \frac{h_i^{clean}}{\|h_i^{clean}\|_2} - \frac{h_i^{comp}}{\|h_i^{comp}\|_2} \right\|_2^2 = 2 \left( 1 - \cos(h_i^{clean}, h_i^{comp}) \right) = 2 \, \mathcal{L}_{cons}$$

This loss directly forces the view-attention aggregation head to weight surviving invariant features such that the final aggregated embedding $h \in \mathbb{R}^{1792}$ does not drift when social-media compression is applied.

### 3. Total Joint Optimization Objective:
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \cdot \mathcal{L}_{cons}$$

In accordance with user instructions, we will systematically investigate:
- **Run 1**: $\lambda = 0.05$ (mild invariance regularization, preserving clean discrimination).
- **Run 2**: $\lambda = 0.10$ (moderate invariance regularization).

---

## 4. Exact Augmentation & Degradation Distributions

For every training sample $x_i$ loaded from the E5 training manifest ($N=23,165$):
1. **Clean Branch ($x_i^{clean}$)**:
   - 100% original RGB image without any artificial compression or scaling.
   - Extracted into standard 5 views (1 center resize $224 \times 224$ + 4 corner crops $224 \times 224$).
2. **Compressed Branch ($x_i^{comp}$)**:
   Degradation applied at source image level before crop extraction, modeled on measured WhatsApp parameters from E10-A:
   - **50% Probability — Single WhatsApp Processing**:
     - Bilinear resize to long edge $L \sim \text{Uniform}(1080, 1600)\text{ px}$.
     - JPEG compression with Quality $Q \sim \text{Uniform}(70, 82)$ and $4:2:0$ chroma subsampling (`subsampling=2`).
   - **50% Probability — Double Social-Media Transmission**:
     - Bilinear resize to long edge $L \sim \text{Uniform}(1080, 1600)\text{ px}$.
     - First JPEG pass: $Q_1 \sim \text{Uniform}(75, 85)$, 4:2:0 subsampling.
     - Second JPEG pass: $Q_2 \sim \text{Uniform}(65, 75)$, 4:2:0 subsampling.
   - Decoded to RGB and extracted into 5 views using identical crop geometries as the clean branch.

---

## 5. Hardware, VRAM Footprint & Runtime Analysis (RTX 3050 Laptop GPU)

### GPU Specifications:
- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU
- **Dedicated VRAM**: 4,096 MB (4.0 GB)
- **Host RAM**: 16 GB
- **PyTorch Environment**: PyTorch 2.13.0+cu126, CUDA 12.6, Torch AMP (`float16` autocast).

### Memory Management Strategy:
- In E10-B, batch size $B=16$ with 5 crops ($16 \times 5 = 80$ crops) consumed **~2.2 GB VRAM**.
- In E12-A, each training step processes paired clean and compressed images. If batch size $B=8$ pairs (8 clean + 8 compressed = 16 images total = 80 crops), the VRAM footprint is **identical to E10-B (~2.2 GB)**.
- Crucially, the backbone `base_model` is evaluated strictly under `torch.no_grad()`. Backprop computational graphs and gradient buffers are allocated **only** for the 1.21M attention and classifier parameters, completely preventing VRAM out-of-memory errors on the 4GB card.

### Detailed Runtime Estimation:
| Operation | Batch Size / Details | Time per Step | Steps per Epoch | Total Time per Epoch |
| :--- | :--- | :--- | :---: | :---: |
| **Data Loading & Dual Crop Extraction** | 4 DataLoader workers, pin memory | ~0.15 sec | 2,896 | Concurrently loaded |
| **Frozen Backbone Forward Pass** | 80 crops (8 clean + 8 comp) in AMP | ~0.18 sec | 2,896 | ~520 sec (~8.6 min) |
| **Attention & Classifier Forward + Loss** | Trainable 1.21M parameters | ~0.04 sec | 2,896 | ~115 sec (~1.9 min) |
| **Backward Pass + Optimizer Step** | Gradient scaler for 1.21M parameters | ~0.05 sec | 2,896 | ~145 sec (~2.4 min) |
| **Total Training Loop per Epoch** | $N=23,165$ images | ~0.27 sec / step | 2,896 | **~780 sec (~13.0 min)** |
| **Clean E5 Validation ($N=5,775$)** | 100% clean, batch size 16, no grad | - | 361 | **~240 sec (~4.0 min)** |
| **Total Time per Epoch** | Training + Validation | - | - | **~17.0 min** |
| **3-Epoch Training Run** | Full experiment | - | - | **~51.0 minutes** |

### Comparison Against 4-Hour Limit:
- **Total Expected Runtime for $\lambda = 0.05$**: **~51 minutes** (21.2% of 4-hour budget).
- **Total Expected Runtime for $\lambda = 0.10$**: **~51 minutes** (21.2% of 4-hour budget).
- **Both sequential runs total**: **~102 minutes (~1.7 hours)**, well below the mandatory 4-hour threshold.
- No full backbone fine-tuning is conducted, ensuring strict adherence to resource constraints.

---

## 6. Model Selection Criteria (Strictly Clean E5 Validation, $N=5,775$)

In accordance with strict uncorrupted protocol, checkpoint selection is governed **exclusively** by the clean E5 validation set. The winning epoch is selected by the following lexicographic priority:
1. **Clean E5 ROC-AUC**: Must maintain $\ge 0.9900$ (target $\ge 0.9936$).
2. **Clean E5 PR-AUC**: Must maintain $\ge 0.9900$.
3. **Clean E5 FPR**: Must remain $\le 3.50\%$ (production E6-C is 2.97%, E10-B is 3.27%).
4. **Clean E5 F1-Score**: Maximized subject to FPR $\le 3.50\%$ (target $\ge 0.9650$).

The winning checkpoint is permanently saved as `e12_best_model.pt` and frozen prior to external testing:
```
E12 BEST MODEL FROZEN — BEGINNING SINGLE-SHOT EXTERNAL WHATSAPP TEST
```

---

## 7. Success Criteria on Final Frozen WhatsApp Benchmark ($N=67$)

After the winning model is selected and frozen, it is evaluated **single-shot** on the frozen WhatsApp benchmark ($N=67$: 34 Real, 33 AI) at the standard production threshold ($\tau=0.50$):

| Metric | E6-C Production | E8-B Augmented | E10-B Best | E12 Success Target |
| :--- | :---: | :---: | :---: | :---: |
| **Clean E5 Val ROC-AUC** | 0.9936 | 0.9541 | 0.9936 | **$\ge 0.9930$** (preserve clean ranking) |
| **Clean E5 Val FPR** | **2.97%** | 3.44% | 3.27% | **$\le 3.50\%$** (no clean regression) |
| **WhatsApp ROC-AUC** | 0.6546 | 0.6881 | **0.7001** | **$\ge 0.7000$** |
| **WhatsApp PR-AUC** | 0.7136 | 0.7209 | **0.7519** | **$\ge 0.7500$** |
| **WhatsApp AI Recall** | 27.27% (9/33) | 42.42% (14/33) | 42.42% (14/33) | **$\ge 42.42\%$** (retain or improve AI catch) |
| **WhatsApp Real FPR** | **5.88%** (2/34) | 14.71% (5/34) | 17.65% (6/34) | **$< 14.71\%$** (materially suppress false alarms) |
| **WhatsApp F1-Score** | 0.4091 | **0.5385** | 0.5283 | **$\ge 0.5500$** |
| **WhatsApp Accuracy** | 61.19% | 64.18% | 62.69% | **$\ge 65.00\%$** |

**Primary Scientific Goal**: Win on both recall AND false alarms simultaneously rather than trading one for the other.

---

## 8. Rollback Plan

1. If during clean E5 validation the model demonstrates loss of clean capability (Clean AUC $< 0.9900$ or Clean FPR $> 4.0\%$), training is terminated immediately.
2. If post-evaluation WhatsApp results fail to improve the recall/FPR Pareto frontier over E6-C and E10-B, E12 will remain documented strictly as a research investigation.
3. Production backend (`backend/inference.py`), frontend, and model artifacts will not be altered. The production system remains safely locked to **E6-C** (`e6c_checkpoint_epoch2.pt`) at threshold **0.50**.

---

## 9. Verification Plan

### Automated Implementation Checks (Pre-Training):
- Verify starting checkpoint `e6c_checkpoint_epoch2.pt` is readable and valid.
- Verify exact parameter count programmatically (total = 12,365,322; trainable = 1,214,466; frozen = 11,150,856).
- Test one paired synthetic batch through the consistency loss pipeline to verify loss calculation, zero VRAM leaks, and correct gradient backpropagation through attention and classifier.
- Verify that `base_model` parameters receive zero gradients (`param.grad is None`).
