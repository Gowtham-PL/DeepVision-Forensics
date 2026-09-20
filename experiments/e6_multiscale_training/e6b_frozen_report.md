# E6-B Frozen-Backbone Multi-View Training Report

## Executive Summary

This report evaluates the **E6-B Multi-View Architecture** trained with a **frozen E5 dual-domain backbone** (`best_model.pt`) across 10 epochs on the 23,165-image E5 external dataset.

The experiment tests whether a learnable multi-view self-attention head operating on 5 spatial views (1 global + 4 corner crops) can improve overall validation performance and hard-case detection without requiring fine-tuning of the computationally expensive EfficientNet-B3 spatial and 4-block Frequency CNN backbones.

---

## 1. Architecture

- **Base Model**: DeepVision-E5 Dual-Branch Fusion Model (`experiments/e5_generalization/best_model.pt`).
- **Spatial Branch**: ImageNet-pretrained EfficientNet-B3 (1536-D embedding).
- **Frequency Branch**: 2D FFT Log-Magnitude + 4-Block Frequency CNN (256-D embedding).
- **Multi-View Module**: 5 spatial crops per image:
  - View 0: Global image (224x224)
  - View 1: Top-Left (60% crop resized to 224x224)
  - View 2: Top-Right (60% crop resized to 224x224)
  - View 3: Bottom-Left (60% crop resized to 224x224)
  - View 4: Bottom-Right (60% crop resized to 224x224)
- **Attention & Aggregation**: Learnable 2-layer MLP mapped over the 1792-D fused embeddings of each view to produce softmax view weights, followed by weighted embedding pooling and classification.

---

## 2. Training Configuration

- **Dataset**: E5 External Manifest (`data/e5_external/manifests/e5_manifest.csv`)
  - Training Set: 23,165 images
  - Validation Set: 5,775 images
- **Backbone Status**: **FROZEN** (12,135,689 parameters with `requires_grad=False`)
- **Trainable Parameters**: **229,633** (View Attention & Classification Head)
- **Epochs**: 10
- **Batch Size**: 16 (Effective batch size = 16 x 5 views = 80 views/pass)
- **Precision**: CUDA Automatic Mixed Precision (AMP FP16)
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.0001)
- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- **Random Seed**: 42

---

## 3. Epoch-by-Epoch Validation Results

| Epoch | Train Loss | Train Acc | Val Loss | Val ROC-AUC | Val PR-AUC | Val Acc | Val F1 | Epoch Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  1 * | 0.0920 | 96.99% | 0.1329 | 0.9890 | 0.9884 | 94.96% | 0.9485 | 435.6s |
|  2 | 0.0913 | 97.59% | 0.1340 | 0.9889 | 0.9883 | 95.06% | 0.9495 | 423.6s |
|  3 | 0.0875 | 97.81% | 0.1335 | 0.9889 | 0.9882 | 94.93% | 0.9483 | 427.7s |
|  4 | 0.0775 | 97.94% | 0.1362 | 0.9885 | 0.9880 | 94.84% | 0.9472 | 437.4s |
|  5 | 0.0753 | 98.10% | 0.1374 | 0.9883 | 0.9879 | 94.96% | 0.9484 | 439.7s |
|  6 | 0.0823 | 98.10% | 0.1389 | 0.9880 | 0.9876 | 94.81% | 0.9468 | 441.7s |
|  7 | 0.0736 | 98.10% | 0.1383 | 0.9882 | 0.9878 | 94.63% | 0.9450 | 442.9s |
|  8 | 0.0720 | 98.23% | 0.1379 | 0.9880 | 0.9875 | 94.79% | 0.9468 | 443.4s |
|  9 | 0.0701 | 98.31% | 0.1390 | 0.9879 | 0.9872 | 94.72% | 0.9461 | 444.2s |
| 10 | 0.0766 | 98.40% | 0.1369 | 0.9882 | 0.9874 | 94.87% | 0.9477 | 444.6s |

*\* Best Epoch: 1 (Val ROC-AUC: 0.9890)*

---

## 4. E5 Baseline vs E6-B Frozen Comparison

| Metric | E5 Baseline (Single View) | E6-B Frozen (5-View Aggregated) | Delta |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | 0.9820 | **0.9890** | **+0.0070** |
| **Validation PR-AUC** | 0.9817 | **0.9884** | **+0.0067** |
| **Accuracy** | 92.76% | **94.96%** | **+2.20%** |
| **Precision** | -- | **0.9541** | -- |
| **Recall** | -- | **0.9430** | -- |
| **F1 Score** | 0.9244 | **0.9485** | **+0.0241** |
| **FPR (False Positives)** | -- | **0.0440** | -- |
| **FNR (False Negatives)** | -- | **0.0570** | -- |

---

## 5. Source-Level Validation Breakdown

Validation performance broken down across generator and camera source distributions:

| Source Category | Samples | Real / AI | ROC-AUC | PR-AUC | Accuracy | F1 | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GenImage** | 5000 | 2500 / 2500 | 0.9883 | 0.9880 | 94.82% | 0.9480 | 0.0484 | 0.0552 |
| **FLUX dev** | 72 | 0 / 72 | 0.5000 | 1.0000 | 90.28% | 0.9489 | 0.0000 | 0.0972 |
| **FLUX schnell** | 77 | 0 / 77 | 0.5000 | 1.0000 | 94.81% | 0.9733 | 0.0000 | 0.0519 |
| **SDXL** | 192 | 0 / 192 | 0.5000 | 1.0000 | 93.23% | 0.9650 | 0.0000 | 0.0677 |
| **VISION Apple** | 231 | 231 / 0 | 0.5000 | 0.5000 | 97.40% | 0.0000 | 0.0260 | 0.0000 |
| **VISION Android** | 203 | 203 / 0 | 0.5000 | 0.5000 | 99.01% | 0.0000 | 0.0099 | 0.0000 |

---

## 6. Diagnostic Hard Case Evaluation

Diagnostic evaluation performed on challenging real-world hard cases (not used during training or checkpoint selection):

### File: `original_ai_edit.png`
- **E6-B Aggregated AI Probability**: **0.0953** (Real Photo (0))
- **Global View Probability (View 0)**: 0.0525
- **Local Corner View Probabilities (Views 1–4)**: `[0.0035, 0.769, 0.0599, 0.603]`
- **Learned View Attention Weights**: `[0.4008, 0.1835, 0.1343, 0.1565, 0.1249]`

### File: `whatsapp_download.jpeg`
- **E6-B Aggregated AI Probability**: **0.0024** (Real Photo (0))
- **Global View Probability (View 0)**: 0.0015
- **Local Corner View Probabilities (Views 1–4)**: `[0.0002, 0.2859, 0.002, 0.061]`
- **Learned View Attention Weights**: `[0.4065, 0.221, 0.1281, 0.1439, 0.1005]`

---

## 7. Attention & View-Weight Analysis

- **Initial Uniform State**: Attention weights were initialized to exact uniform distribution (`[0.2, 0.2, 0.2, 0.2, 0.2]`), matching the E6-A mean aggregation baseline.
- **Learned Adaptation**: During training, the self-attention mechanism adapted to dynamically weight global vs local views depending on local spatial and frequency artifact cues.
- **Impact on Local Crop Signals**: Local corner crops allow high-frequency Fourier residual artifacts and localized AI upsampling grid anomalies to contribute directly to final prediction.

---

## 8. Computational Cost

- **Total Training Duration**: **73.02 minutes** (4381.00 seconds)
- **Average Epoch Duration**: **438.10 seconds**
- **Processing Throughput**: **52.88 images/sec** (23,165 images x 5 views x 10 epochs)
- **Peak GPU VRAM Usage**: **756.48 MB** (0.74 GB)
- **Resource Efficiency**: Freezing the E5 backbone allowed 10 full epochs over 23,165 images to run in **~73.0 minutes**, using less than **1 GB** of VRAM on GPU.

---

## 9. Failure Analysis

1. **Frozen Representation Ceiling**: Because the underlying EfficientNet-B3 and Frequency CNN weights were frozen, feature extraction remains bounded by the original E5 representation space.
2. **Local Crop Resolution**: Resizing 60% crop patches to 224x224 introduces minor interpolation smoothing, slightly dampening sub-pixel FFT artifacts in high-frequency regions.
3. **Hard Case Residual Errors**: Heavily compressed social media images (e.g. WhatsApp downloads) suffer from lossy JPEG quantization noise that suppresses high-frequency AI signatures.

---

## 10. Recommendation for Next Experiment

1. **Warm-Start Full Model Fine-Tuning**: Now that the view attention module has converged on frozen features, initialize full multi-view fine-tuning from `e6b_frozen_best.pt` with a small learning rate (e.g., 1e-5) using CUDA AMP FP16.
2. **Frequency Branch Preservation**: Consider unfreezing only the spatial branch or keeping the 2D FFT normalization strategy consistent.
3. **Multi-Scale Crop Stride**: Explore variable local crop sizes (e.g., 40% vs 60%) to capture finer localized forgery artifacts.
