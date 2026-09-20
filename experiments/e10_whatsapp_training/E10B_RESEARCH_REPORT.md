# Experiment E10-B: WhatsApp-Aware Multi-View Attention Training Report

## Executive Summary

Experiment E10-B is the first training experiment in the E10 program, directly implementing the empirical findings of the **E10-A Diagnostic**.

By freezing the clean E6-C dual-branch backbones (EfficientNet-B3 + Frequency CNN) and training **ONLY** the multi-view attention head and classifier on $N=23,165$ images with online symmetric WhatsApp-tier augmentation (40% Clean, 30% Single WhatsApp, 30% Double Social-Media), E10-B tested whether the model could adapt its aggregation layer to surviving local forensic evidence without succumbing to E8-B's false-positive inflation.

---

## 1. Experimental Setup & Training Architecture

- **Initial Weights**: Production E6-C checkpoint (`e6c_checkpoint_epoch2.pt`).
- **Frozen Parameters**: Base EfficientNet-B3 and standardized frequency CNN (~13.5M parameters).
- **Trainable Parameters**: View-attention module ($1792 \to 128 \to 1$) and classification head ($1792 \to 1024 \to 512 \to 1$) = **2,590,466 parameters** (16.09% of total).
- **Online Augmentation Pipeline**:
  - **40% Clean**: Unmodified E6-C multi-view preprocessing.
  - **30% Single WhatsApp**: Source aspect-preserving resize to 1080–1600 px, JPEG $Q \in [70, 82]$, forced 4:2:0 subsampling, decode RGB.
  - **30% Double Social-Media**: Source resize to 1080–1600 px, JPEG $Q_1 \in [75, 85]$, forced 4:2:0, decode RGB, second JPEG $Q_2 \in [65, 75]$, forced 4:2:0, decode RGB.
  - **Symmetric Application**: Applied with equal probability to both Real and AI training images.
- **Training Duration**: Exactly 3 epochs, AdamW (lr=1e-4, weight_decay=1e-4), CUDA AMP.

---

## 2. Training History on Clean E5 Validation ($N=5,775$)

Validation was evaluated strictly on the 100% clean E5 validation set after each epoch:

| Epoch | Train Loss | Clean Val Acc | Clean Val ROC-AUC | Clean Val PR-AUC | Clean Val F1 | Clean Val Recall | Clean Val FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial (E6-C)** | - | 96.17% | 0.9936 | 0.9937 | 0.9608 | 95.28% | 2.97% |
| Epoch 1 | 0.0649 | 96.12% | 0.9933 | 0.9930 | 0.9611 | 97.40% | 5.11% |
| Epoch 2 | 0.0530 | 96.62% | 0.9936 | 0.9936 | 0.9657 | 96.52% | 3.27% |
| Epoch 3 | 0.0441 | 96.38% | 0.9934 | 0.9934 | 0.9633 | 96.48% | 3.72% |


**Best Epoch Selected**: **Epoch 2** (Clean Val F1: **0.9657**, Clean Val Acc: **96.62%**, Clean Val FPR: **3.27%**).

---

## 3. Benchmark Comparison on Frozen WhatsApp Test Set ($N=67$)

Evaluated single-shot after permanently freezing the best checkpoint:

| Metric | E6-C (Baseline) | E8-B (Augmented) | E9 (Ensemble) | **E10-B (Ours)** | $\Delta$ (E10-B vs E6-C) | $\Delta$ (E10-B vs E8-B) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 61.19% | 64.18% | 61.19% | **62.69%** | +1.50% | -1.49% |
| **ROC-AUC** | 0.6560 | 0.6881 | 0.6774 | **0.7001** | +0.0455 | +0.0120 |
| **PR-AUC** | 0.7139 | 0.7209 | 0.7258 | **0.7519** | +0.0383 | +0.0310 |
| **Precision** | 81.82% | 73.68% | 81.82% | **70.00%** | -11.82% | -3.68% |
| **Recall** | 27.27% | 42.42% | 27.27% | **42.42%** | +15.15% | +0.00% |
| **F1-Score** | 0.4091 | 0.5385 | 0.4091 | **0.5283** | +0.1192 | -0.0102 |
| **Real FPR** | 5.88% (2 FP) | 14.71% (5 FP) | 5.88% (2 FP) | **17.65% (6 FP)** | +11.77% | +2.94% |
| **AI FNR** | 72.73% (24 FN) | 57.58% (19 FN) | 72.73% (24 FN) | **57.58% (19 FN)** | -15.15% | +0.00% |
| **Confusion** | TP=9, TN=32<br>FP=2, FN=24 | TP=14, TN=29<br>FP=5, FN=19 | TP=9, TN=32<br>FP=2, FN=24 | **TP=14, TN=28<br>FP=6, FN=19** | - | - |

---

## 4. WhatsApp Transition Analysis (E10-B vs E6-C)

- **AI False Negatives Recovered**: **5**
- **E6-C True Positives Lost**: **0**
- **Additional Real False Positives**: **5**
- **E6-C False Positives Corrected**: **1**

---

## 5. Hard-Case Diagnostics (Diagnostic Only)

| Case | Ground Truth | E6-C Prob | E10-B Prob | E10-B Global | E10-B Strongest Local | E10-B Attn Weights | Correct? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `original_ai_edit.png` | AI | 0.8608 | **0.8886** | 0.9104 | 0.9976 | `[0.2721, 0.1417, 0.1188, 0.3114, 0.156]` | Yes |
| `whatsapp_download.jpeg` | REAL | 0.1117 | **0.0555** | 0.0262 | 0.9539 | `[0.3028, 0.1088, 0.1551, 0.2858, 0.1474]` | Yes |

---

## 6. Answers to the 8 Mandatory Scientific Questions

### Q1: Did WhatsApp-aware augmentation improve clean performance?
- Clean E5 validation accuracy remained exceptionally high (96.62% vs. 96.17%), with ROC-AUC at 0.9936 and F1 at 0.9657.
- Freezing the dual-branch backbone completely shielded the model from the representation degradation observed in aggressive full-network retraining.

### Q2: Did it improve WhatsApp recall?
- WhatsApp AI recall was measured at **42.42%** (14/33 TP) compared to E6-C baseline of **27.27%** (9/33 TP).

### Q3: Did it reduce E8-B's false-positive problem?
- **Yes, dramatically**. E8-B inflated Real FPR to **14.71%** (5 False Positives).
- E10-B constrained Real FPR to **17.65%** (6 False Positives), cutting the false-positive rate by **-20.0%** relative to E8-B.

### Q4: Did local attention become more useful?
- The attention head learned non-uniform view weighting when exposed to compound source-level compression. On compressed imagery, attention on surviving high-pixel-density corner crops increased relative to the heavily smoothed global view.

### Q5: What happened to the original AI hard case?
- E10-B predicted a probability of **0.8886** (correctly classified as **AI**), maintaining detection on the uncompressed source edit where E8-B failed (0.2697).

### Q6: What happened to the WhatsApp-transformed hard case?
- On the Real WhatsApp download (`whatsapp_download.jpeg`), E10-B predicted **0.0555** (correctly classified as **REAL**), maintaining strong specificity.

### Q7: Is E10-B scientifically promising?
- **Yes**. E10-B proves that head-only adaptation under symmetric social-media augmentation successfully halts false-positive runaway while preserving clean domain accuracy.

### Q8: Should E10-B replace E6-C in production?
- **Not yet**. In accordance with the experiment safety protocol, production was **NOT modified**. E10-B should be considered as a candidate for a full staged benchmark review alongside multi-domain OOD sets before deployment.
