# E20 Training & Validation Experiment Report

**Experiment**: E20 High-Generalization Forensic Detector Training  
**Date**: September 19, 2026  
**Status**: Completed, Frozen, and Audited  
**Base Architecture**: 5-View `MultiViewE5Model`  
**Starting Checkpoint**: `experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`  
**Best Checkpoint Selected**: `experiments/e20_training/checkpoints/e20_best_model.pt` (Epoch 4)  
**Evaluation Threshold**: Strictly 0.50  

---

## 1. Executive Summary & Objective

The objective of **Experiment 20 (E20)** was to train a high-generalization multi-scale forensic detector addressing the concrete failure patterns documented throughout E1–E19:
1. **Modern Flow-Matching & SOTA Generative Architectures**: Overcoming prior detector blind spots on FLUX.1 [dev], Gemini 1.5 Pro / Imagen 3, Stable Diffusion 3.5 / SDXL, DALL-E, and Midjourney.
2. **Camera Hardware & ISP Post-Processing**: Eliminating false alarms on modern smartphone computational photography (Apple iPhone X, Samsung Galaxy S9, Google Pixel 7/8, Vivo X90) and uncompressed DSLR optics (Nikon D7000/D90 via RAISE).
3. **Severe & Compound Compression Invariance**: Resolving the catastrophic high-frequency feature collapse observed in previous models under sequential recompression, thumbnail scaling, and WhatsApp degradation.

**Key Experimental Result**:
Starting from the frozen production baseline E6-C (which achieved **82.04% accuracy** and **0.8011 F1** on E20 Dev with 387 AI false negatives and 116 Real false alarms), the staged E20 fine-tuning pipeline achieved:
- **Dev Accuracy**: **98.54%** (+16.50 percentage points over E6-C)
- **Dev ROC-AUC**: **0.9979** (+0.0899 over E6-C)
- **Dev F1 Score**: **0.9853** (+0.1842 over E6-C)
- **Real False Positive Rate (FPR)**: **1.29%** (vs 8.29% for E6-C — an **84.5% reduction in false alarms**)
- **FLUX.1 [dev] Detection Recall**: **100.00%** (826/826 detected) vs 76.63% in E6-C
- **Midjourney & DALL-E Recall**: **100.00%**
- **Severe JPEG (Q40) F1 Score**: **98.75%** vs 66.02% in E6-C (+32.73 percentage points)
- **WhatsApp Simulation F1 Score**: **98.75%** vs 89.01% in E6-C

---

## 2. Model Architecture & Parameter Counts

The model strictly maintains the frozen production E6-C multi-view architecture:
- **Spatial Branch**: ImageNet-normalized EfficientNet-B3 extracting 1536-D feature representations.
- **Frequency Branch**: Standardized 2D FFT Log-Magnitude 4-block CNN extracting 256-D spectral representations.
- **Dual-Domain Concatenation**: Fuses spatial and frequency embeddings into a 1792-D vector per view.
- **Multi-View Attention**: 5 views processed simultaneously:
  1. Global View (224 $\times$ 224 bilinear resize)
  2. Top-Left View (60% crop resized to 224 $\times$ 224)
  3. Top-Right View (60% crop resized to 224 $\times$ 224)
  4. Bottom-Left View (60% crop resized to 224 $\times$ 224)
  5. Bottom-Right View (60% crop resized to 224 $\times$ 224)
  Learnable self-attention maps $1792 \to 128 \to 1$ followed by softmax view aggregation.
- **Classifier Head**: Reused deep MLP ($1792 \to 512 \to 128 \to 1$ with BatchNorm and Dropout).

### Parameter Breakdown

| Component | Trainable (Phase 1) | Trainable (Phase 2) | Total Parameters |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B3 Spatial Backbone** | Frozen (0) | 10,696,352 | 10,696,352 |
| **Frequency CNN Backbone** | Frozen (0) | 454,504 | 454,504 |
| **View Attention Module** | 229,761 | 229,761 | 229,761 |
| **Classifier Head** | 984,705 | 984,705 | 984,705 |
| **Total Model Parameters** | **1,214,466 (9.8%)** | **12,365,322 (100%)** | **12,365,322** |

---

## 3. Dataset Sizes & Distribution

The training and development datasets were audited and sealed prior to training:
- **Train Split**: 22,400 examples (3,200 clean originals $\times$ 7 symmetric variants)
- **Dev Split**: 2,800 examples (400 clean originals $\times$ 7 symmetric variants)
- **Label Balance**: Exactly 50.0% Real (11,200 Train / 1,400 Dev) and 50.0% AI (11,200 Train / 1,400 Dev)
- **Splitting Strategy**: Strictly partitioned by original source image (seed 42); zero cross-split derivative leakage.
- **Sealed Test Isolation**: Zero overlap against E14 (`FINAL_TEST_POOL`, `FINAL_TEST_DEGRADED`), E19 benchmark, or WhatsApp N=67 benchmark (verified against 41,372 historical hashes).

### Symmetric Compression Transformation Distribution

Each original source image in both splits has exactly 7 symmetric variants:
1. `clean`: Pristine source render (3,200 Train, 400 Dev)
2. `resize_jpeg`: 75% bilinear resize + JPEG Q80 4:2:0 (3,200 Train, 400 Dev)
3. `moderate_jpeg`: JPEG Q65 4:2:0 (3,200 Train, 400 Dev)
4. `severe_jpeg`: JPEG Q40 4:2:0 (3,200 Train, 400 Dev)
5. `sequential_jpeg`: JPEG Q85 recompressed to JPEG Q60 4:2:0 (3,200 Train, 400 Dev)
6. `resize_jpeg_resize`: 60% downscale + JPEG Q75 + upscale + JPEG Q85 (3,200 Train, 400 Dev)
7. `social_media_whatsapp`: WhatsApp simulation ($\max(w,h) \le 1280\text{px}$ + JPEG Q75 4:2:0) (3,200 Train, 400 Dev)

---

## 4. Training Hyperparameters & Staged Strategy

To ensure stable convergence from the pre-trained E6-C weights without catastrophic forgetting, training followed a two-phase staged fine-tuning schedule:

| Hyperparameter | Phase 1 (Epochs 1–2) | Phase 2 (Epochs 3–5) |
| :--- | :--- | :--- |
| **Backbone Status** | Frozen (`requires_grad = False`) | Unfrozen (`requires_grad = True`) |
| **Backbone Learning Rate** | N/A | $1.0 \times 10^{-5}$ |
| **Head Learning Rate** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **Optimizer** | AdamW | AdamW |
| **Weight Decay** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ |
| **Loss Function** | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) | Same |
| **Batch Size (Physical)** | 4 | 4 |
| **Gradient Accumulation**| 2 steps (Effective Batch Size = 8) | 2 steps (Effective Batch Size = 8) |
| **Mixed Precision** | PyTorch AMP FP16 (`torch.amp.autocast('cuda')`) | Same |
| **Data Workers** | 2 persistent workers with `pin_memory = True` | Same |
| **Device** | NVIDIA GeForce RTX 3050 Laptop GPU (4.29 GB VRAM) | Same |
| **Decision Threshold** | 0.50 (Frozen) | 0.50 (Frozen) |

---

## 5. Epoch-by-Epoch Training & Validation Progression

| Epoch | Phase | Train Loss | Dev ROC-AUC | Dev PR-AUC | Dev Accuracy | Dev F1 | Dev Precision | Dev Recall | Dev FPR | Dev FNR | TP | TN | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (E6-C)**| *Baseline* | — | 0.9080 | 0.9087 | 82.04% | 0.8011 | 89.73% | 72.36% | 8.29% | 27.64% | 1,013 | 1,284 | 116 | 387 |
| **1** | Phase 1 | 0.3594 | 0.9931 | 0.9922 | 96.21% | 0.9616 | 97.64% | 94.71% | 2.29% | 5.29% | 1,326 | 1,368 | 32 | 74 |
| **2** | Phase 1 | 0.2425 | 0.9949 | 0.9940 | 97.25% | 0.9723 | 97.83% | 96.64% | 2.14% | 3.36% | 1,353 | 1,370 | 30 | 47 |
| **3** | Phase 2 | 0.1791 | 0.9940 | 0.9847 | 97.86% | 0.9784 | 98.76% | 96.93% | 1.21% | 3.07% | 1,357 | 1,383 | 17 | 43 |
| **4\*** | **Phase 2** | **0.1161** | **0.9979** | **0.9973** | **98.54%** | **0.9853** | **98.71%** | **98.36%** | **1.29%** | **1.64%** | **1,377** | **1,382** | **18** | **23** |
| **5** | Phase 2 | 0.0808 | 0.9961 | 0.9963 | 97.89% | 0.9787 | 98.98% | 96.79% | 1.00% | 3.21% | 1,355 | 1,386 | 14 | 45 |

*\*Selected Best Checkpoint based on highest E20 Dev F1 (0.9853) and ROC-AUC (0.9979).*

---

## 6. Granular Subgroup Metrics Analysis

### 6.1 AI Generator Recall Progression

| Generator Family | Total Dev Count | E6-C Baseline (Epoch 0) | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 (Best) | Epoch 5 | Delta (vs E6-C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLUX.1 [dev]** | 826 | 76.63% | 98.06% | 99.64% | 99.88% | **100.00%** | 99.27% | **+23.37 pp** |
| **Stable Diffusion / SDXL** | 511 | 68.49% | 91.19% | 92.95% | 93.35% | **96.87%** | 94.13% | **+28.38 pp** |
| **Google Gemini / Imagen** | 28 | 50.00% | 60.71% | 71.43% | 75.00% | **75.00%** | 67.86% | **+25.00 pp** |
| **OpenAI DALL-E** | 28 | 42.86% | 92.86% | 100.00% | 96.43% | **100.00%** | 100.00% | **+57.14 pp** |
| **Midjourney** | 7 | 57.14% | 100.00% | 100.00% | 100.00% | **100.00%** | 100.00% | **+42.86 pp** |

### 6.2 Real Camera Hardware False Positive Rate (FPR) Progression

| Camera Hardware Family | Total Dev Count | E6-C Baseline (Epoch 0) | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 (Best) | Epoch 5 | Delta (vs E6-C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple iPhone (iPhone X)** | 427 | 3.28% | 1.41% | 1.64% | 0.00% | **0.00%** | 0.00% | **-3.28 pp (100% fixed)** |
| **Samsung Galaxy (S9)** | 427 | 6.32% | 0.00% | 0.00% | 0.23% | **0.00%** | 0.00% | **-6.32 pp (100% fixed)** |
| **Google Pixel (Pixel 7/8)** | 42 | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | 0.00% | **Maintained 0.00%** |
| **Vivo (Vivo X90 / Zeiss)** | 196 | 9.18% | 3.57% | 3.57% | 2.55% | **0.00%** | 0.00% | **-9.18 pp (100% fixed)** |
| **Nikon DSLR RAW (RAISE)** | 308 | 18.51% | 6.17% | 5.19% | 3.57% | **5.84%** | 4.55% | **-12.67 pp (68.4% reduction)**|

### 6.3 Compression Variant F1 Score Progression

| Compression Type | Dev Count | E6-C Baseline (Epoch 0) | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 (Best) | Epoch 5 | Delta (vs E6-C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | 400 | 86.36% | 96.74% | 97.01% | 97.73% | **98.50%** | 98.24% | **+12.14 pp** |
| `resize_jpeg` | 400 | 83.15% | 96.24% | 96.98% | 97.74% | **98.50%** | 97.74% | **+15.35 pp** |
| `moderate_jpeg` | 400 | 81.82% | 95.73% | 96.97% | 97.47% | **98.24%** | 97.73% | **+16.42 pp** |
| `severe_jpeg` (Q40) | 400 | 66.02% | 95.67% | 97.49% | 97.22% | **98.75%** | 98.49% | **+32.73 pp** |
| `sequential_jpeg` | 400 | 83.78% | 96.48% | 97.49% | 97.99% | **98.50%** | 97.98% | **+14.72 pp** |
| `resize_jpeg_resize` | 400 | 82.87% | 95.96% | 97.24% | 97.98% | **98.50%** | 97.98% | **+15.63 pp** |
| `social_media_whatsapp`| 400 | 89.01% | 96.73% | 97.74% | 97.98% | **98.75%** | 98.24% | **+9.74 pp** |

---

## 7. Operational & Hardware Performance

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (Dedicated VRAM: 4.29 GB)
- **Peak VRAM Allocated**: **2,395.5 MB** (55.8% of available physical memory; 0 CUDA out-of-memory events, 0 memory spikes).
- **Training Throughput**:
  - Phase 1 (Frozen backbones): **57.0 images/second** (~13.1 minutes per epoch)
  - Phase 2 (Unfrozen backbones): **21.5 images/second** (~17.4 minutes per epoch)
- **Total Training Duration**: **4,192.0 seconds** (**69.87 minutes** across all 5 epochs + 6 complete validation sweeps).

---

## 8. Checkpoint Selection & Artifacts

In strict compliance with the research protocol, model selection was governed **exclusively** by the E20 Dev split:
- **Best Model Selected**: **Epoch 4** (`experiments/e20_training/checkpoints/e20_best_model.pt`)
- **Primary Metric (F1)**: **0.9853**
- **Tie-Breaker 1 (ROC-AUC)**: **0.9979**
- **Tie-Breaker 2 (FPR)**: **0.0129**
- **Tie-Breaker 3 (Recall)**: **0.9836**

### Generated Artifacts
1. Best Model Checkpoint: `experiments/e20_training/checkpoints/e20_best_model.pt`
2. Per-Epoch Checkpoints: `experiments/e20_training/checkpoints/e20_checkpoint_epoch{1..5}.pt`
3. Training Metrics Table: `experiments/e20_training/e20_training_metrics.csv`
4. Best Model Predictions: `experiments/e20_training/predictions/e20_best_dev_predictions.csv`
5. Checkpoint Metadata: `experiments/e20_training/checkpoints/e20_best_meta.json`

---

## 9. Safety, Integrity, & Non-Contamination Confirmation

The following repository safeguards have been verified:
- **E14 Final Test Pools**: Untouched and sealed (`data/final_external_pool`).
- **E19 Smartphone Benchmark**: Untouched and sealed (`data/e19_smartphone_benchmark`).
- **WhatsApp N=67 Benchmark**: Untouched and sealed (`data/whatsapp_robustness_test`).
- **E6-C Checkpoint**: Untouched (`experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`).
- **Production Code**: Backend (`backend/`), frontend (`frontend/`), and default thresholds (0.50) remain completely unchanged.
- **Git State**: Clean; no commits or pushes executed.
