# Experiment E13 Research Report: Compression-Aware Training + Domain Normalization

## Executive Summary

Experiment **E13** investigated whether combining realistic compression-aware augmentation (reproducing the validated **E10-B** distribution) with lightweight domain normalization (**`LayerNorm(1792)`**, validated in **E12-B**) on the aggregated multi-view representation can successfully unite their complementary advantages: retaining the high compressed-AI recall and strong ranking ability of E10-B while dampening the real-photo false-positive tendency via LayerNorm stabilization.

The feature extraction backbones (EfficientNet-B3 + Frequency CNN, **11,150,856 parameters**) were kept **100% frozen**. Only `view_attention` (229,633 params), the new `LayerNorm` (3,584 params), and `classifier` (984,833 params) were trained (**1,218,050 trainable parameters**, 9.85% of model).

Training ran for **exactly 3 epochs** on the official E5 training split ($N=23,165$) using AdamW (lr=1e-4, weight_decay=1e-4) with CUDA AMP FP16. Model selection strictly adhered to the gated Clean E5 validation protocol without any WhatsApp leakage.

---

## 1. Clean E5 Validation Progression ($N=5,775$)

| Epoch | Train Loss | Val Loss | Val Accuracy | Val ROC-AUC | Val PR-AUC | Val F1 | Val Recall | Val FPR | Gate Status | Train Time | Peak VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 1 | 0.0773 | 0.1244 | 95.77% | 0.9919 | 0.9913 | 0.9576 | 96.94% | 5.35% | **FAIL** | 410.5s | 0.79 GB |
| Epoch 2 | 0.0487 | 0.1160 | 96.61% | 0.9931 | 0.9927 | 0.9656 | 96.76% | 3.54% | **PASS** | 404.2s | 0.79 GB |
| Epoch 3 | 0.0344 | 0.1259 | 96.43% | 0.9926 | 0.9913 | 0.9638 | 96.44% | 3.58% | **FAIL** | 340.3s | 0.79 GB |

### Checkpoint Selection Audit:
- **Winning Checkpoint**: **Epoch 2** (`checkpoints/e13_best_model.pt`)
- **Selection Reason**: Epoch 2 passed both safety gates (ROC-AUC=0.9931 >= 0.9930, FPR=3.54% <= 4.00%) and achieved highest Clean F1 (0.9656) among 1 qualifying checkpoint(s).

---

## 2. Comparison Across Models on Clean E5 Validation ($N=5,775$)

| Model | Regimen | Clean Val Acc | Clean Val ROC-AUC | Clean Val PR-AUC | Clean Val F1 | Clean Val FPR | Clean Val Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C (Baseline)** | Multiscale FT | 96.17% | 0.9936 | 0.9937 | 0.9608 | **2.97%** | 95.28% |
| **E10-B (Augmented)** | Social-Media Aug | 96.62% | 0.9936 | 0.9936 | 0.9657 | 3.27% | 96.52% |
| **E12-A (Invariance Reg)** | Paired Cosine Consistency | 96.48% | 0.9929 | 0.9920 | 0.9645 | 4.19% | 97.18% |
| **E12-B (LayerNorm Only)** | Clean Data + LayerNorm | **97.18%** | **0.9948** | **0.9948** | **0.9716** | 3.68% | **98.06%** |
| **E13 (Comp Aug + LayerNorm)** | **E10-B Aug + LayerNorm** | **96.61%** | **0.9931** | **0.9927** | **0.9656** | **3.54%** | **96.76%** |

---

## 3. Hardware, Runtime, and VRAM Utilization

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM limit)
- **Peak VRAM**: Consistently **~0.79 GB** (leaves >2.8 GB headroom, zero OOM risk)
- **Total Training Duration**: **24.41 minutes** (Target: $\le 120$ minutes $\implies$ PASSED)
- **Average Train Step Time**: ~0.24s per step (16 images = 80 crops per forward pass in CUDA AMP FP16)

---

## 4. Single-Shot Evaluation on Frozen WhatsApp Benchmark ($N=67$ at Threshold 0.50)

```
E13 CHECKPOINT FROZEN — BEGINNING EXTERNAL TEST
```

| Metric | E6-C (Baseline) | E8-B (Augmented) | E10-B (Standard) | E12-A (Invariance) | E12-B (LayerNorm) | **E13 (Combined)** | $\Delta$ (E13 vs E10-B) | $\Delta$ (E13 vs E12-B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 61.19% | 64.18% | 62.69% | 61.19% | 62.69% | **59.70%** | -2.99% | -2.99% |
| **ROC-AUC** | 0.6546 | 0.6881 | **0.7001** | 0.6840 | 0.6386 | **0.6974** | -0.0027 | +0.0588 |
| **PR-AUC** | 0.7136 | 0.7209 | **0.7519** | 0.7418 | 0.7215 | **0.7514** | -0.0005 | +0.0299 |
| **Precision** | **81.82%** | 73.68% | 70.00% | 68.42% | 75.00% | **63.64%** | -6.36% | -11.36% |
| **Recall** | 27.27% | **42.42%** | **42.42%** | 39.39% | 36.36% | **42.42%** | +0.00% | +6.06% |
| **F1-Score** | 0.4091 | **0.5385** | 0.5283 | 0.5000 | 0.4898 | **0.5091** | -0.0192 | +0.0193 |
| **Real FPR** | **5.88% (2 FP)** | 14.71% (5 FP) | 17.65% (6 FP) | 17.65% (6 FP) | 11.76% (4 FP) | **23.53% (8 FP)** | +5.88% | +11.77% |
| **AI FNR** | 72.73% (24 FN) | 57.58% (19 FN) | **57.58% (19 FN)** | 60.61% (20 FN) | 63.64% (21 FN) | **57.58% (19 FN)** | +0.00% | -6.06% |
| **TP / TN** | 9 / 32 | 14 / 29 | 14 / 28 | 13 / 28 | 12 / 30 | **14 / 26** | +0 TP / -2 TN | +2 TP / -4 TN |
| **FP / FN** | 2 / 24 | 5 / 19 | 6 / 19 | 6 / 20 | 4 / 21 | **8 / 19** | +2 FP / +0 FN | +4 FP / -2 FN |

---

## 5. Transition Breakdown Relative to Prior Models on WhatsApp

### A. Relative to E10-B:
- **AI False Negatives Recovered (0):**
  - None
- **AI True Positives Lost (0):**
  - None
- **Real False Positives Corrected (0):**
  - None
- **New Real False Positives Introduced (2):**
  - `WhatsApp Image 2026-09-17 at 3.38.22 PM (3).jpeg`: E10-B=0.4992 -> **E13=0.5347 (AI)**
  - `WhatsApp Image 2026-09-17 at 3.44.10 PM (4).jpeg`: E10-B=0.4211 -> **E13=0.6201 (AI)**

### B. Relative to E12-B:
- **AI False Negatives Recovered (2):**
  - `WhatsApp Image 2026-09-17 at 3.56.02 PM.jpeg`: E12-B=0.0792 -> **E13=0.6064 (AI)**
  - `WhatsApp Image 2026-09-17 at 4.13.18 PM (2).jpeg`: E12-B=0.4387 -> **E13=0.9844 (AI)**
- **AI True Positives Lost (0):**
  - None
- **Real False Positives Corrected (1):**
  - `WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg`: E12-B=0.7510 -> **E13=0.0583 (REAL)**
- **New Real False Positives Introduced (5):**
  - `WhatsApp Image 2026-09-17 at 3.38.22 PM (3).jpeg`: E12-B=0.2216 -> **E13=0.5347 (AI)**
  - `WhatsApp Image 2026-09-17 at 3.40.06 PM (1).jpeg`: E12-B=0.0073 -> **E13=0.7104 (AI)**
  - `WhatsApp Image 2026-09-17 at 3.44.10 PM (4).jpeg`: E12-B=0.0205 -> **E13=0.6201 (AI)**
  - `WhatsApp Image 2026-09-17 at 3.44.11 PM (11).jpeg`: E12-B=0.0276 -> **E13=0.5483 (AI)**
  - `WhatsApp Image 2026-09-17 at 3.44.11 PM (14).jpeg`: E12-B=0.3503 -> **E13=0.6787 (AI)**

---

## 6. Hard-Case Benchmark Reproduction

| Benchmark Image | Ground Truth | E10-B Prob | E12-B Prob | **E13 Prob** | Global Prob | Strongest Local Prob | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit.png` | **AI** | 0.8886 | 0.9985 | **0.9961** | 0.9932 | 0.9995 | **True** |
| `whatsapp_download.jpeg` | **REAL** | 0.0555 | 0.0015 | **0.2246** | 0.1331 | 0.9902 | **True** |

---

## 7. Formal Success Criteria Evaluation

### Primary Success Criteria:
- **WhatsApp AI Recall $\ge 42.42\%$**: PASSED (Achieved: **42.42%**)
- **WhatsApp Real FPR $< 14.71\%$**: FAILED (Achieved: **23.53%**, 8 FP)
- **WhatsApp ROC-AUC $\ge 0.7000$**: FAILED (Achieved: **0.6974**)

### Clean-Data Safety Criteria:
- **Clean E5 ROC-AUC $\ge 0.9930$**: PASSED (Achieved: **0.9931**)
- **Clean E5 FPR $\le 4.00\%$**: PASSED (Achieved: **3.54%**)
- **Clean E5 F1 $\ge 0.9650$**: PASSED (Achieved: **0.9656**)
- **Clean E5 Accuracy $\ge 96.50\%$**: PASSED (Achieved: **96.61%**)

### Secondary Success Criterion:
- **WhatsApp F1 $\ge 0.5300$**: FAILED (Achieved: **0.5091**)

---

## 8. Final Conclusion: Does E13 Combine the Benefits of E10-B and E12-B?

*(Empirical verdict documented in final analysis.)*
