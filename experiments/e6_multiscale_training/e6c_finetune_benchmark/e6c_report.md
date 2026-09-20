# E6-C Controlled Multi-View Backbone Fine-Tuning Benchmark Report

## Executive Summary

This report evaluates **E6-C**, a controlled **2-epoch fine-tuning benchmark** of the **MultiViewE5Model** architecture with unfrozen spatial (EfficientNet-B3) and frequency (4-Block CNN) backbones. Training was initialized directly from the best frozen checkpoint (`experiments/e6_multiscale_training/e6b_frozen_best.pt`), using differential learning rates (`1e-5` for backbones, `1e-4` for multi-view attention/classifier) with batch size 8 on CUDA AMP FP16.

---

## 1. Architecture & Setup

- **Initialization Checkpoint**: `experiments/e6_multiscale_training/e6b_frozen_best.pt` (E6-B Frozen Best)
- **Spatial Branch**: ImageNet-pretrained EfficientNet-B3 (1536-D embedding, `requires_grad=True`, lr=1e-5)
- **Frequency Branch**: 2D FFT Log-Magnitude + 4-Block Frequency CNN (256-D embedding, `requires_grad=True`, lr=1e-5)
- **Multi-View Attention Head**: 2-layer MLP (`requires_grad=True`, lr=1e-4)
- **Classification Head**: Linear layer (`requires_grad=True`, lr=1e-4)
- **Parameters**:
  - Backbone Parameters: **11,150,856** (lr=1e-5)
  - Multi-View Parameters: **1,214,466** (lr=1e-4)
  - Total Trainable Parameters: **12,365,322**

---

## 2. Benchmark Configuration

- **Dataset**: E5 External Manifest (`data/e5_external/manifests/e5_manifest.csv`)
  - Training Set: 23,165 images
  - Validation Set: 5,775 images
- **Batch Size**: 8 (Effective batch size = 8 x 5 views = 40 views/pass)
- **Epochs**: 2
- **Optimizer**: AdamW with differential learning rates (`1e-5` backbones / `1e-4` attention+head, weight decay `1e-4`)
- **Precision**: CUDA AMP FP16
- **Random Seed**: 42

---

## 3. Epoch-by-Epoch Validation Results

| Epoch | Train Loss | Train Acc | Val Loss | Val ROC-AUC | Val PR-AUC | Val Acc | Val F1 | Epoch Time | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
|  1 | 0.1815 | 92.98% | 0.1446 | 0.9916 | 0.9920 | 95.53% | 0.9535 | 7895.1s | `e6c_checkpoint_epoch1.pt` |
|  2 * | 0.1507 | 94.25% | 0.1034 | 0.9936 | 0.9937 | 96.16% | 0.9606 | 7875.0s | `e6c_checkpoint_epoch2.pt` |

*\* Best Epoch: 2 (Val ROC-AUC: 0.9936)*

---

## 4. Model Comparison: E5 Baseline vs E6-B Frozen vs E6-C Fine-Tuned

| Metric | E5 Baseline (Single View) | E6-B Frozen (5-View) | E6-C Fine-Tuned (Epoch 2) | Delta vs E5 | Delta vs E6-B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation ROC-AUC** | 0.9820 | 0.9890 | **0.9936** | **+0.0116** | **+0.0046** |
| **Validation PR-AUC** | 0.9817 | 0.9884 | **0.9937** | **+0.0120** | **+0.0053** |
| **Accuracy** | 92.76% | 94.96% | **96.16%** | **+3.40%** | **+1.20%** |
| **Precision** | -- | -- | **0.9689** | -- | -- |
| **Recall** | -- | -- | **0.9525** | -- | -- |
| **F1 Score** | 0.9244 | 0.9485 | **0.9606** | **+0.0362** | **+0.0121** |
| **FPR (False Positives)** | -- | 0.0440 | **0.0297** | -- | **-0.0143** |
| **FNR (False Negatives)** | -- | -- | **0.0475** | -- | -- |

---

## 5. Source-Level Validation Breakdown

| Source Category | Samples | Real / AI | ROC-AUC | PR-AUC | Accuracy | F1 | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GenImage** | 5000 | 2500 / 2500 | 0.9937 | 0.9938 | 96.06% | 0.9603 | 0.0324 | 0.0464 |
| **FLUX dev** | 72 | 0 / 72 | 0.5000 | 1.0000 | 93.06% | 0.9640 | 0.0000 | 0.0694 |
| **FLUX schnell** | 77 | 0 / 77 | 0.5000 | 1.0000 | 96.10% | 0.9801 | 0.0000 | 0.0390 |
| **SDXL** | 192 | 0 / 192 | 0.5000 | 1.0000 | 94.27% | 0.9705 | 0.0000 | 0.0573 |
| **VISION Apple** | 231 | 231 / 0 | 0.5000 | 0.5000 | 98.27% | 0.0000 | 0.0173 | 0.0000 |
| **VISION Android** | 203 | 203 / 0 | 0.5000 | 0.5000 | 99.01% | 0.0000 | 0.0099 | 0.0000 |

---

## 6. Diagnostic Hard Case Evaluation

Diagnostic evaluation on challenging hard cases (diagnostic only; no influence on training):

### File: `original_ai_edit.png`
- **E6-C Aggregated AI Probability**: **0.8638** (AI Generated (1))
- **Global View Probability (View 0)**: 0.8633
- **Local Corner View Probabilities (Views 1–4)**: `[0.4153, 0.9595, 0.7583, 0.7896]`
- **Learned View Attention Weights**: `[0.2148, 0.2041, 0.2061, 0.1983, 0.1767]`

### File: `whatsapp_download.jpeg`
- **E6-C Aggregated AI Probability**: **0.1128** (Real Photo (0))
- **Global View Probability (View 0)**: 0.0549
- **Local Corner View Probabilities (Views 1–4)**: `[0.0008, 0.7471, 0.0793, 0.2394]`
- **Learned View Attention Weights**: `[0.2175, 0.1988, 0.2205, 0.1878, 0.1754]`

---

## 7. Computational Cost & Scaling Metrics

- **Total Benchmark Duration**: **262.85 minutes** (15770.89 seconds)
- **Average Epoch Duration**: **7885.45 seconds** (131.42 minutes)
- **Processing Throughput**: **2.94 images/sec** (Effective: 14.70 views/sec)
- **Peak GPU VRAM Usage**: **4256.78 MB** (4.16 GB)
- **Extrapolated 10-Epoch Full Training Runtime**: **21.90 hours**

---

## 8. Benchmark Evaluation & Research Questions

### A. Does backbone fine-tuning improve validation ROC-AUC over E6-B Frozen?
**YES**

### B. Does it improve F1?
**YES**

### C. Does it preserve the low FPR?
**YES**

### D. Does it improve the original AI-edit hard case?
**Aggregated AI probability shifted from 9.53% (E6-B) to 86.38%**

### E. Does it improve the WhatsApp hard case?
**Aggregated AI probability shifted from 0.24% (E6-B) to 11.28%**

### F. What is the measured full-training runtime estimate?
**21.90 hours for full 10-epoch training (based on measured 7885.4s/epoch)**

### G. Is further fine-tuning computationally practical?
**Moderate Compute Cost**

---

## 9. Conclusion & Next Experiment Recommendation

1. **Backbone Adaptation**: Unfreezing the dual-domain backbone with differential learning rates allowed the spatial EfficientNet-B3 and Frequency CNN to jointly adapt to localized crop features while retaining global coherence.
2. **Computational Practicality**: With CUDA AMP FP16 and batch size 8, 2 full epochs over 23,165 images completed in **262.9 minutes** using **4256.8 MB** VRAM.
3. **Actionable Next Step**: Proceed based on whether the measured 2-epoch gains warrant full convergence training.
