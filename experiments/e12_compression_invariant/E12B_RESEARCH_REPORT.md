# Experiment E12-B Research Report: Clean-Data Domain-Conditioned Normalization

## Executive Summary

Experiment E12-B tests the hypothesis of whether introducing **lightweight domain-conditioned LayerNorm(1792)** on the aggregated multi-view representation prior to classification can reduce WhatsApp real-image false positives while preserving AI recall.

Crucially, in contrast to E10-B (which applied online social-media augmentations) and E12-A (which applied paired consistency regularization under synthetic compression), **E12-B trained exclusively on clean E5 data ($N=23,165$)**. The feature extraction backbones (EfficientNet-B3 + Frequency CNN, **11,150,856 parameters**) were kept **100% frozen**. Only `view_attention` (229,633 params), the new `LayerNorm` (3,584 params), and `classifier` (984,833 params) were trained (**1,218,050 trainable parameters**, 9.85% of model).

The 67-image WhatsApp benchmark remained completely frozen until final single-shot evaluation.

---

## 1. Clean E5 Validation Progression ($N=5,775$)

| Epoch | Train Loss | Val Loss | Val Accuracy | Val ROC-AUC | Val PR-AUC | Val F1 | Val Recall | Val FPR | Train Time | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 1 | 0.0425 | 0.1107 | 96.66% | 0.9945 | 0.9939 | 0.9665 | 98.13% | 4.77% | 308.8s | 0.79 GB |
| Epoch 2 | 0.0244 | 0.1070 | 97.18% | 0.9948 | 0.9948 | 0.9716 | 98.06% | 3.68% | 323.7s | 0.79 GB |
| Epoch 3 | 0.0151 | 0.1237 | 97.02% | 0.9946 | 0.9942 | 0.9701 | 98.06% | 3.99% | 319.6s | 0.79 GB |

**Winning Checkpoint**: **Epoch 2** selected strictly based on Clean E5 validation F1-score (0.9716) and saved as `checkpoints_e12b/e12b_best_model.pt`.

---

## 2. Comparison Across Models on Clean E5 Validation ($N=5,775$)

| Model | Checkpoint Status | Clean Val Acc | Clean Val ROC-AUC | Clean Val PR-AUC | Clean Val F1 | Clean Val FPR | Clean Val Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C (Baseline)** | Frozen | 96.17% | 0.9936 | 0.9937 | 0.9608 | **2.97%** | 95.28% |
| **E10-B (Augmented)** | Frozen | 96.62% | 0.9936 | 0.9936 | 0.9657 | 3.27% | 96.52% |
| **E12-A (Invariance Reg)** | Frozen | 96.48% | 0.9929 | 0.9920 | 0.9645 | 4.19% | **97.18%** |
| **E12-B (LayerNorm, Best)** | **Frozen** | **97.18%** | **0.9948** | **0.9948** | **0.9716** | **3.68%** | **98.06%** |

---

## 3. Hardware, Runtime, and VRAM Utilization

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **Peak VRAM**: Consistently **~0.79 GB** (zero OOM risk)
- **Total Training Duration**: **21.21 minutes**
- **Average Train Step Time**: ~0.20s per step (16 images = 80 crops per forward pass)

---

## 4. Single-Shot Evaluation on Frozen WhatsApp Benchmark ($N=67$ at Threshold 0.50)

```
E12-B CHECKPOINT FROZEN — BEGINNING EXTERNAL TEST
```

| Metric | E6-C (Baseline) | E8-B (Augmented) | E10-B (Standard) | E12-A (Invariance) | **E12-B (LayerNorm)** | $\Delta$ (E12-B vs E6-C) | $\Delta$ (E12-B vs E10-B) | $\Delta$ (E12-B vs E12-A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 61.19% | 64.18% | 62.69% | 61.19% | **62.69%** | +1.50% | +0.00% | +1.50% |
| **ROC-AUC** | 0.6546 | 0.6881 | 0.7001 | 0.6840 | **0.6386** | -0.0160 | -0.0615 | -0.0454 |
| **PR-AUC** | 0.7136 | 0.7209 | 0.7519 | 0.7418 | **0.7215** | +0.0079 | -0.0304 | -0.0203 |
| **Precision** | 81.82% | 73.68% | 70.00% | 68.42% | **75.00%** | -6.82% | +5.00% | +6.58% |
| **Recall** | 27.27% | 42.42% | 42.42% | 39.39% | **36.36%** | +9.09% | -6.06% | -3.03% |
| **F1-Score** | 0.4091 | 0.5385 | 0.5283 | 0.5000 | **0.4898** | +0.0807 | -0.0385 | -0.0102 |
| **Real FPR** | **5.88%** (2 FP) | 14.71% (5 FP) | 17.65% (6 FP) | 17.65% (6 FP) | **11.76% (4 FP)** | +5.88% | -5.89% | -5.89% |
| **AI FNR** | 72.73% (24 FN) | 57.58% (19 FN) | 57.58% (19 FN) | 60.61% (20 FN) | **63.64% (21 FN)** | -9.09% | +6.06% | +3.03% |
| **TP / TN** | 9 / 32 | 14 / 29 | 14 / 28 | 13 / 28 | **12 / 30** | +3 TP / -2 TN | -2 TP / +2 TN | -1 TP / +2 TN |
| **FP / FN** | 2 / 24 | 5 / 19 | 6 / 19 | 6 / 20 | **4 / 21** | +2 FP / -3 FN | -2 FP / +2 FN | -2 FP / +1 FN |

---

## 5. Transition Breakdown Relative to Prior Models on WhatsApp

### Flips vs E12-A (7 total flips):
- `WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg`: GT=REAL | E12-A=0.0133 (REAL) -> **E12-B=0.7510 (AI)**
- `WhatsApp Image 2026-09-17 at 3.40.06 PM (1).jpeg`: GT=REAL | E12-A=0.6914 (AI) -> **E12-B=0.0073 (REAL)**
- `WhatsApp Image 2026-09-17 at 3.44.10 PM (4).jpeg`: GT=REAL | E12-A=0.6841 (AI) -> **E12-B=0.0205 (REAL)**
- `WhatsApp Image 2026-09-17 at 3.44.11 PM (14).jpeg`: GT=REAL | E12-A=0.7886 (AI) -> **E12-B=0.3503 (REAL)**
- `WhatsApp Image 2026-09-17 at 3.56.02 PM.jpeg`: GT=AI | E12-A=0.6221 (AI) -> **E12-B=0.0792 (REAL)**
- `WhatsApp Image 2026-09-17 at 4.13.18 PM (2).jpeg`: GT=AI | E12-A=0.9912 (AI) -> **E12-B=0.4387 (REAL)**
- `WhatsApp Image 2026-09-17 at 4.28.11 PM (6).jpeg`: GT=AI | E12-A=0.2325 (REAL) -> **E12-B=0.5986 (AI)**

### Flips vs E10-B (6 total flips):
- `WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg`: GT=REAL | E10-B=0.0745 (REAL) -> **E12-B=0.7510 (AI)**
- `WhatsApp Image 2026-09-17 at 3.40.06 PM (1).jpeg`: GT=REAL | E10-B=0.7493 (AI) -> **E12-B=0.0073 (REAL)**
- `WhatsApp Image 2026-09-17 at 3.44.11 PM (11).jpeg`: GT=REAL | E10-B=0.5697 (AI) -> **E12-B=0.0276 (REAL)**
- `WhatsApp Image 2026-09-17 at 3.44.11 PM (14).jpeg`: GT=REAL | E10-B=0.7722 (AI) -> **E12-B=0.3503 (REAL)**
- `WhatsApp Image 2026-09-17 at 3.56.02 PM.jpeg`: GT=AI | E10-B=0.6093 (AI) -> **E12-B=0.0792 (REAL)**
- `WhatsApp Image 2026-09-17 at 4.13.18 PM (2).jpeg`: GT=AI | E10-B=0.9810 (AI) -> **E12-B=0.4387 (REAL)**

### Flips vs E6-C Baseline (5 total flips):
- `WhatsApp Image 2026-09-17 at 3.38.22 PM (1).jpeg`: GT=REAL | E6-C=0.3354 (REAL) -> **E12-B=0.8301 (AI)**
- `WhatsApp Image 2026-09-17 at 3.44.11 PM (12).jpeg`: GT=REAL | E6-C=0.4119 (REAL) -> **E12-B=0.9189 (AI)**
- `WhatsApp Image 2026-09-17 at 4.13.18 PM (3).jpeg`: GT=AI | E6-C=0.2781 (REAL) -> **E12-B=0.9468 (AI)**
- `WhatsApp Image 2026-09-17 at 4.13.18 PM.jpeg`: GT=AI | E6-C=0.4772 (REAL) -> **E12-B=0.6680 (AI)**
- `WhatsApp Image 2026-09-17 at 4.28.11 PM (6).jpeg`: GT=AI | E6-C=0.3284 (REAL) -> **E12-B=0.5986 (AI)**

---

## 6. Hard-Case Benchmark Reproduction

| Benchmark Image | Ground Truth | E10-B Prob | E12-A Prob | **E12-B Prob** | Global Prob | Strongest Local Prob | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit.png` | **AI** | 0.8886 | 0.9282 | **0.9985** | 0.9976 | 1.0000 | **True** |
| `whatsapp_download.jpeg` | **REAL** | 0.0555 | 0.0120 | **0.0015** | 0.0006 | 0.9258 | **True** |

---

## 7. Empirical Findings & Scientific Conclusions

1. **Clean E5 Performance**:
   - Clean Validation Accuracy: **97.18%**
   - Clean Validation ROC-AUC: **0.9948**
   - Clean Validation PR-AUC: **0.9948**
   - Clean Validation F1-Score: **0.9716**
   - Clean Validation FPR: **3.68%**
   - Clean Validation AI Recall: **98.06%**

2. **WhatsApp Robustness Hypothesis Evaluation**:
   - Real False Positives: **4 / 34** (11.76% FPR)
   - AI True Positives: **12 / 33** (36.36% Recall)
   - WhatsApp ROC-AUC: **0.6386**
   - WhatsApp F1: **0.4898**

*(Detailed analysis and next-step considerations documented after empirical review.)*
