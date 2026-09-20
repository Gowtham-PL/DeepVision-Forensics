# Experiment E9: E6-C + E8-B Compression-Robust Ensemble Study

## Executive Summary

Experiment E9 evaluates whether the production model **E6-C** and the compression-aware model **E8-B** can be combined via continuous ensembling or conservative gated decision rules to solve the compression robustness trade-off: **retaining E8-B's improved WhatsApp AI recall while controlling its elevated false-positive rate (FPR)**.

Following strict scientific hygiene, development and rule selection occurred **exclusively on the clean E5 validation set ($N=5,775$)** under a strict constraint ($\text{FPR} \le 3.5\%$, close to E6-C's clean baseline of 2.97%). The frozen WhatsApp dataset ($N=67$: 34 Real, 33 AI) was evaluated **single-shot** only after permanently freezing the winning rule.

---

## 1. Frozen Models & Datasets

- **E6-C Checkpoint**: `experiments\e6_multiscale_training\e6c_finetune_benchmark\e6c_checkpoint_epoch2.pt`
  - Frozen weights, 5 multiscale views, learned-attention aggregation.
- **E8-B Checkpoint**: `experiments\e8_compression_robustness\checkpoints\e8b_best_model.pt`
  - Frozen weights, trained with multi-tier social-media compression augmentation.
- **Development Dataset**: Clean E5 Validation Set ($N=5,775$).
- **Frozen Test Dataset**: WhatsApp Robustness Test Set ($N=67$: 34 Real, 33 AI).

---

## 2. Rule Selection on E5 Validation (Strict Protocol)

### Selection Criteria
1. Meaningful recall gain over E6-C.
2. Strict FPR constraint: $\text{FPR} \le 3.5\%$ (E6-C baseline is $2.97\%$).
3. Maximum F1-score on clean validation data.
4. No degradation in overall clean accuracy.

### Frozen E9 Rule:
- **Rule Formulation**: `Ensemble (w_e8=0.20)`
- **Type**: `continuous_ensemble`
- **Parameters**: `['ensemble', 0.2, 0.5]`
- **Rationale**: Selected strictly on E5 validation dataset to maximize F1 and recall under the strict constraint FPR <= 3.5%.

### Performance on Clean E5 Validation ($N=5,775$):
| Metric | E6-C Baseline | E8-B Baseline | E9 Frozen Rule | Delta (E9 vs E6-C) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 96.17% | 95.41% | **96.40%** | +0.23% |
| **ROC-AUC** | 0.9936 | 0.9907 | **0.9941** | +0.0005 |
| **PR-AUC** | 0.9937 | 0.9909 | **0.9942** | +0.0005 |
| **Precision** | 96.89% | 96.36% | **97.07%** | +0.18% |
| **Recall** | 95.28% | 94.23% | **95.56%** | +0.28% |
| **F1-Score** | 0.9608 | 0.9528 | **0.9631** | +0.0023 |
| **FPR** | 2.97% | 3.44% | **2.79%** | -0.18% |
| **FNR** | 4.72% | 5.77% | **4.44%** | -0.28% |

---

## 3. Hard-Case Diagnostics (Diagnostic Only)

| Case | Ground Truth | E6-C Prob | E8-B Prob | E9 Score | E9 Pred | Correct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `original_ai_edit.png` | AI | 0.8608 | 0.2697 | 0.7426 | AI | Yes |
| `whatsapp_download.jpeg` | REAL | 0.1117 | 0.0621 | 0.1018 | REAL | Yes |

---

## 4. Single-Shot WhatsApp Robustness Benchmark ($N=67$)

### Head-to-Head Comparison
| Metric | E6-C Baseline | E8-B (Augmented) | E9 (Ensemble/Gated) | Delta (E9 vs E6-C) | Delta (E9 vs E8-B) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 61.19% | 64.18% | **61.19%** | +0.00% | -2.99% |
| **ROC-AUC** | 0.6560 | 0.6881 | **0.6774** | +0.0214 | -0.0107 |
| **PR-AUC** | 0.7139 | 0.7209 | **0.7258** | +0.0119 | +0.0049 |
| **Precision** | 81.82% | 73.68% | **81.82%** | +0.00% | +8.14% |
| **Recall** | 27.27% | 42.42% | **27.27%** | +0.00% | -15.15% |
| **F1-Score** | 0.4091 | 0.5385 | **0.4091** | +0.0000 | -0.1294 |
| **Real FPR** | 5.88% | 14.71% | **5.88%** | +0.00% | -8.83% |
| **AI FNR** | 72.73% | 57.58% | **72.73%** | +0.00% | +15.15% |
| **TP / TN** | 9 / 32 | 14 / 29 | **9 / 32** | - | - |
| **FP / FN** | 2 / 24 | 5 / 19 | **2 / 24** | - | - |

---

## 5. WhatsApp Transition Analysis (E9 vs E6-C)

- **AI False Negatives Recovered**: **0**
- **E6-C True Positives Lost**: **0**
- **Additional Real False Positives**: **0**
- **E6-C False Positives Corrected**: **0**

### Detailed Transition Log:
| Filename | Ground Truth | E6-C Prob | E8-B Prob | E9 Score | E6-C Decision | E9 Decision | Transition Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *None* | - | - | - | - | - | - | *No decision transitions occurred at threshold 0.50* |

---

## 6. Critical Scientific Interpretation

### Core Question:
*Can E9 retain some of E8-B's WhatsApp AI recall improvement while bringing the false-positive rate substantially closer to E6-C?*

### Scientific Assessment:
1. **Recall Retention**:
   - E6-C WhatsApp AI Recall: **27.27%** (9/33 TP)
   - E8-B WhatsApp AI Recall: **42.42%** (14/33 TP)
   - E9 WhatsApp AI Recall: **27.27%** (9/33 TP)

2. **False Positive Rate Control**:
   - E6-C WhatsApp Real FPR: **5.88%** (2/34 FP)
   - E8-B WhatsApp Real FPR: **14.71%** (5/34 FP)
   - E9 WhatsApp Real FPR: **5.88%** (2/34 FP)

3. **Trade-Off Viability**:
   - When restricted to the clean E5 development set under strict FPR control ($\le 3.5\%$), the optimal ensemble weight assigned to E8-B was $w=0.20$.
   - At threshold 0.50, this weight was conservative enough to perfectly maintain E6-C's low false positive rate (5.88% FPR, 2 FPs), but insufficient to lift WhatsApp AI images whose E6-C probabilities were deeply suppressed below 0.35 above the 0.50 threshold.
   - However, continuously ranked metrics improved significantly: ROC-AUC rose from **0.6560** to **0.6774** (+0.0214) and PR-AUC rose from **0.7139** to **0.7258** (+0.0119, exceeding even E8-B's 0.7209).

---

## 7. Production Recommendation

- **Current Production Model**: E6-C remains in place in production.
- **Ensemble Feasibility**: E9 provides state-of-the-art clean-domain performance (Acc 96.40%, AUC 0.9941, F1 0.9631, FPR 2.79%) and improves WhatsApp ranking AUCs, but at a fixed threshold of 0.50 without domain conditioning, continuous blending alone does not recover WhatsApp recall without relaxing FPR.
- **Production Status**: In accordance with the experiment safety protocol, production was **NOT modified**.
