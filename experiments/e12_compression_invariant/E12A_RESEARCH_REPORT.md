# Experiment E12-A Research Report: Compression-Invariant Feature Regularization

## Executive Summary

Experiment E12-A investigates **Compression-Invariant Feature Regularization** ($\lambda = 0.05$) to target the root cause of WhatsApp degradation identified in E10 and E11: feature covariance drift causing compressed real-photo embeddings to overlap with AI cues.

The base backbones (EfficientNet-B3 + Frequency CNN, **11,150,856 parameters**) were kept **100% frozen**. Only `view_attention` (229,633 parameters) and `classifier` (984,833 parameters) were trained (**1,214,466 trainable parameters**, 9.82% of model).

Training processed paired clean/compressed views with a joint loss:
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + 0.05 \cdot \mathcal{L}_{cons}$$
where $\mathcal{L}_{cons} = 1 - \cos(h_{clean}, h_{comp})$.

---

## 1. Clean E5 Validation Progression ($N=5,775$)

| Epoch | Total Loss | Cls Loss | Cons Loss | Accuracy | ROC-AUC | PR-AUC | F1-Score | Recall | FPR | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 1 | 0.0658 | 0.0653 | 0.0100 | 95.43% | 0.9918 | 0.9900 | 0.9545 | 97.54% | 6.61% | 713.8s |
| Epoch 2 | 0.0447 | 0.0442 | 0.0102 | 96.48% | 0.9929 | 0.9920 | 0.9645 | 97.18% | 4.19% | 706.1s |
| Epoch 3 | 0.0324 | 0.0319 | 0.0104 | 96.26% | 0.9922 | 0.9900 | 0.9623 | 97.04% | 4.50% | 633.6s |


**Selected Winning Checkpoint**: **Epoch 2** was frozen as `e12a_best_model.pt` based strictly on clean E5 validation.

---

## 2. Comparison Across Models on Clean E5 Validation

| Model | Checkpoint Status | Clean Val Acc | Clean Val ROC-AUC | Clean Val PR-AUC | Clean Val F1 | Clean Val FPR | Clean Val Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C (Production Baseline)** | Frozen | 96.17% | 0.9936 | 0.9937 | 0.9608 | **2.97%** | 95.28% |
| **E10-B (Social-Media Aug)** | Frozen | 96.62% | 0.9936 | 0.9936 | 0.9657 | 3.27% | 96.52% |
| **E12-A (Invariance Reg, Best)**| **Frozen** | **96.48%** | **0.9929** | **0.9920** | **0.9645** | **4.19%** | **97.18%** |

---

## 3. Hardware, Runtime, and VRAM Utilization

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **VRAM Utilization**: Peak VRAM stayed consistently at **~2.2 GB** (no VRAM spike or OOM).
- **Total Training Duration**: **39.23 minutes** (well within the 4-hour budget).
- **Average Train Step Time**: ~0.26 sec per paired step (8 clean + 8 comp = 80 crops).

---

## 4. Single-Shot Evaluation on Frozen WhatsApp Benchmark ($N=67$)

```
E12-A CHECKPOINT FROZEN — BEGINNING EXTERNAL TEST
```

| Metric | E6-C (Baseline) | E8-B (Augmented) | E10-B (Standard) | **E12-A (Best Model)** | $\Delta$ (E12-A vs E6-C) | $\Delta$ (E12-A vs E10-B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 61.19% | 64.18% | 62.69% | **61.19%** | +0.00% | -1.50% |
| **ROC-AUC** | 0.6546 | 0.6881 | 0.7001 | **0.6840** | +0.0294 | -0.0161 |
| **PR-AUC** | 0.7136 | 0.7209 | 0.7519 | **0.7418** | +0.0282 | -0.0101 |
| **Precision** | 81.82% | 73.68% | 70.00% | **68.42%** | -13.40% | -1.58% |
| **Recall** | 27.27% | 42.42% | 42.42% | **39.39%** | +12.12% | -3.03% |
| **F1-Score** | 0.4091 | 0.5385 | 0.5283 | **0.5000** | +0.0909 | -0.0283 |
| **Real FPR** | **5.88%** (2 FP) | 14.71% (5 FP) | 17.65% (6 FP) | **17.65% (6 FP)** | +11.77% | -0.00% |
| **AI FNR** | 72.73% (24 FN) | 57.58% (19 FN) | 57.58% (19 FN) | **60.61% (20 FN)** | -12.12% | +3.03% |
| **TP / TN** | 9 / 32 | 14 / 29 | 14 / 28 | **13 / 28** | +4 TP / -4 TN | -1 TP / +0 TN |
| **FP / FN** | 2 / 24 | 5 / 19 | 6 / 19 | **6 / 20** | +4 FP / -4 FN | +0 FP / +1 FN |

---

## 5. Transition Breakdown Relative to E10-B on WhatsApp

### AI False Negatives Recovered (+0):
None

### AI True Positives Lost (-1):
- `WhatsApp Image 2026-09-17 at 4.28.11 PM (6).jpeg`: E10-B=0.5154 -> **E12-A=0.2325**

### Real False Positives Corrected (+1):
- `WhatsApp Image 2026-09-17 at 3.44.11 PM (11).jpeg`: E10-B=0.5697 -> **E12-A=0.4438**

### Real False Positives Introduced (+1):
- `WhatsApp Image 2026-09-17 at 3.44.10 PM (4).jpeg`: E10-B=0.4211 -> **E12-A=0.6841**

---

## 6. Hard-Case Benchmark Reproduction

| Benchmark Image | Ground Truth | E10-B Prob | E12-A Prob | E12-A Global | E12-A Strongest Local | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit.png` | **AI** | 0.8886 | **0.9282** | 0.9683 | 0.9995 | **True** |
| `whatsapp_download.jpeg` | **REAL** | 0.0555 | **0.0120** | 0.0051 | 0.9492 | **True** |

Both hard cases were classified with high confidence and deterministic accuracy.

---

## 7. Recommendation Regarding Lambda = 0.10

**NOT RECOMMENDED / STOP**: E12-A (lambda=0.05) did not satisfy all continuation criteria (Clean AUC >= 0.9930, Clean FPR <= 3.50%, WhatsApp Recall >= 42.42%). Additional training runs with higher regularization weights are unlikely to yield superior Pareto frontiers.

---

## 8. Protocol and Safety Confirmation
Zero modifications were made to production code, backend inference endpoints, models, or datasets. Production remains on **E6-C** at threshold **0.50**.
