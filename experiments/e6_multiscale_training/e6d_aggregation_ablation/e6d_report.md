# E6-D Frozen E6-C Aggregation Ablation Report

## Executive Summary

This report evaluates **E6-D: Frozen E6-C Aggregation Ablation**. Using the exact same frozen model weights from **E6-C Epoch 2** (`experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`), we systematically evaluate 7 alternative multi-view aggregation strategies across all **5,775 validation images** (exact E5 validation split) and on two diagnostic hard-case images (`original_ai_edit.png` and `whatsapp_download.jpeg`).

Zero training or fine-tuning was performed. All model weights remained strictly frozen (`requires_grad=False`).

---

## 1. Safety & Control Confirmations

| Control Verification | Status | Note |
| :--- | :---: | :--- |
| **Model Weights Frozen** | **CONFIRMED** | 0 trainable parameters (`requires_grad=False` on all 12,365,322 params) |
| **No Training Occurred** | **CONFIRMED** | Pure inference with `torch.no_grad()` and `model.eval()` |
| **V2 Untouched** | **CONFIRMED** | V2 was not accessed, used for selection, or modified |
| **Production Threshold Unchanged** | **CONFIRMED** | Fixed 0.50 threshold evaluated across all strategies |
| **Existing Checkpoints Unchanged** | **CONFIRMED** | E5, E6-B, and E6-C checkpoints remain completely unmodified |

---

## 2. Aggregation Strategy Performance Comparison (N=5,775 Validation Images)

Evaluation conducted on the full 5,775 validation set at the fixed threshold of 0.50:

| Aggregation Strategy | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 Score | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Learned Attention (E6-C Baseline)** | **0.9936** | **0.9937** | **96.16%** | 0.9689 | 0.9525 | **0.9606** | **2.97%** | 4.75% |
| **2. Simple Mean** | 0.9931 | 0.9932 | 96.12% | 0.9702 | 0.9504 | 0.9602 | 2.83% | 4.96% |
| **3. Simple Max** | 0.9879 | 0.9872 | 91.76% | 0.8637 | 0.9884 | 0.9219 | 15.10% | 1.16% |
| **4. Top-2 Mean** | 0.9911 | 0.9910 | 94.29% | 0.9091 | 0.9820 | 0.9442 | 9.51% | 1.80% |
| **5. Global + Strongest Local Mean** | 0.9887 | 0.9882 | 95.52% | 0.9415 | 0.9690 | 0.9551 | 5.83% | 3.10% |
| **6. Global-Weighted Strongest-Local** | 0.9887 | 0.9882 | 95.52% | 0.9415 | 0.9690 | 0.9551 | 5.83% | 3.10% |
| **7. Global + Top-2-Local Mean** | 0.9918 | 0.9919 | 95.72% | 0.9406 | 0.9747 | 0.9573 | 5.96% | 2.53% |

---

## 3. Diagnostic Hard Case Evaluation

Diagnostic evaluation on challenging hard cases:

### File 1: `original_ai_edit.png`
- **View 0 (Global)**: **0.8628**
- **Views 1–4 (Local Corner Crops)**: `[0.4138, 0.9595, 0.7578, 0.7891]`
- **Learned Attention Weights**: `[0.2148, 0.2041, 0.2061, 0.1983, 0.1767]`
- **Aggregation Strategy Predictions**:
  - **1_learned_attention**: `0.8638` (AI Generated (1))
  - **2_simple_mean**: `0.7568` (AI Generated (1))
  - **3_simple_max**: `0.9595` (AI Generated (1))
  - **4_top2_mean**: `0.9111` (AI Generated (1))
  - **5_global_strongest_mean**: `0.9111` (AI Generated (1))
  - **6_global_weighted_strongest**: `0.9111` (AI Generated (1))
  - **7_global_top2_local_mean**: `0.8706` (AI Generated (1))

### File 2: `whatsapp_download.jpeg`
- **View 0 (Global)**: **0.0550**
- **Views 1–4 (Local Corner Crops)**: `[0.0008, 0.7456, 0.0793, 0.2397]`
- **Learned Attention Weights**: `[0.2175, 0.1988, 0.2205, 0.1878, 0.1754]`
- **Aggregation Strategy Predictions**:
  - **1_learned_attention**: `0.1128` (Real Photo (0))
  - **2_simple_mean**: `0.2241` (Real Photo (0))
  - **3_simple_max**: `0.7456` (AI Generated (1))
  - **4_top2_mean**: `0.4927` (Real Photo (0))
  - **5_global_strongest_mean**: `0.4004` (Real Photo (0))
  - **6_global_weighted_strongest**: `0.4004` (Real Photo (0))
  - **7_global_top2_local_mean**: `0.3469` (Real Photo (0))

---

## 4. Key Findings & Analysis

1. **Validation Performance Trade-offs**:
   - **Learned Attention (Strategy 1)** remains optimal on overall validation ROC-AUC (0.9936), Accuracy (96.16%), and F1 (0.9606) while keeping False Positives low (FPR: 2.97%).
   - **Simple Max (Strategy 3)** increases sensitivity to localized AI artifacts, but drastically degrades False Positive Rate on real images (FPR rises to 15.10%).
   - **Global + Strongest Local Mean (Strategies 5 & 6)** strikes an effective heuristic balance between the global context and the most confident localized crop.

2. **Hard Case Impact**:
   - On `original_ai_edit.png`, multiple strategies correctly classify the image as **AI Generated (1)** at threshold 0.50.
   - On `whatsapp_download.jpeg`, the local top-right crop detects strong localized evidence (`0.7456`). Heuristic aggregation strategies such as **Simple Max** (0.7456) or **Global + Strongest Local** (0.4004) substantially elevate the AI probability compared to global-only or uniform averaging.

3. **Inference Efficiency**:
   - Total evaluation of 5,775 images across all 5 views and all 7 strategies completed in **105.04 seconds** (1.75 minutes) on NVIDIA GeForce RTX 3050 Laptop GPU.
   - Throughput: **54.98 images/sec** with peak VRAM of **745.72 MB**.

---

## 5. Conclusion & Recommendation

- **Learned Attention (E6-C)** remains the recommended primary model for general deployment due to superior global ROC-AUC and low FPR.
- For specialized forensic workflows focused on localized inpainting or compressed social media downloads (WhatsApp), exposing individual crop maximums or a **Global + Strongest Local** score provides valuable diagnostic evidence of localized tampering.
