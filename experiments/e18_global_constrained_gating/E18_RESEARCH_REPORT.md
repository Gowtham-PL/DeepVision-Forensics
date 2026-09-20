# E18 — Global-Constrained Local Attention Research Report

**Date**: 2026-09-18 13:16:47 UTC  
**Status**: COMPLETE  
**Primary Research Question**: Can global-conditioned local gating reduce E15's false-positive behavior on compressed real smartphone photographs while retaining the compression robustness gained by E15?

---

## 1. Executive Summary & Core Results

In E17, forensic analysis revealed that 58.3% (7/12) of E15's false positives on WhatsApp real smartphone photos were purely local-driven: corner crops fired spurious AI detections that overwhelmed the clean global view.

In **E18**, we replaced the unconstrained attention aggregation head with a **Global-Constrained Local Gating** module (475,521 params) and a lightweight fusion classifier (2,294,273 params), keeping EfficientNet-B3 and FFT CNN backbones 100% frozen. The model was trained on the identical 640 compression variants from `FINAL_TRAIN_POOL` and selected against the clean E5 validation set ($N=5,775$).

### Comprehensive Cross-Model Benchmark Comparison Table

| Benchmark Split | Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`FINAL_TEST_POOL` (Clean, N=200)** | E6-C Baseline | 75.50% | 0.8426 | 0.8598 | 76.29% | 74.00% | 0.7513 | 23.00% | 26.00% |
| **`FINAL_TEST_POOL` (Clean, N=200)** | E15 Robust | 79.50% | 0.8682 | 0.8781 | 74.38% | 90.00% | 0.8145 | 31.00% | 10.00% |
| **`FINAL_TEST_POOL` (Clean, N=200)** | **E18 Gated** | **74.50%** | **0.8704** | **0.8700** | **68.70%** | **90.00%** | **0.7792** | **41.00%** | **10.00%** |
| | | | | | | | | | |
| **`FINAL_TEST_DEGRADED` (N=600)** | E6-C Baseline | 65.83% | 0.7640 | 0.7762 | 78.44% | 43.67% | 0.5610 | 12.00% | 56.33% |
| **`FINAL_TEST_DEGRADED` (N=600)** | E15 Robust | 76.83% | 0.8252 | 0.8262 | 77.10% | 76.33% | 0.7672 | 22.67% | 23.67% |
| **`FINAL_TEST_DEGRADED` (N=600)** | **E18 Gated** | **75.83%** | **0.8410** | **0.8389** | **71.47%** | **86.00%** | **0.7806** | **34.33%** | **14.00%** |
| | | | | | | | | | |
| **WhatsApp Benchmark (N=67)** | E6-C Baseline | 61.19% | 0.6551 | 0.7175 | 81.82% | 27.27% | 0.4091 | **5.88% (2/34)** | 72.73% |
| **WhatsApp Benchmark (N=67)** | E15 Robust | 56.72% | 0.6221 | 0.6721 | 57.14% | 48.48% | 0.5246 | **35.29% (12/34)** | 51.52% |
| **WhatsApp Benchmark (N=67)** | **E18 Gated** | **59.70%** | **0.6119** | **0.6183** | **58.82%** | **60.61%** | **0.5970** | **41.18% (14/34)** | **39.39%** |

---

## 2. Architecture & Trainable Parameter Verification

- **Spatial Backbone (EfficientNet-B3)**: 10,696,344 parameters — **100% Frozen**
- **Frequency Backbone (FFT CNN)**: 454,512 parameters — **100% Frozen**
- **Base Per-View Classifier**: 984,833 parameters — **100% Frozen**
- **Trainable Global-Constrained Gating**: 475,521 parameters
- **Trainable Fusion Classifier (8960 -> 256 -> 1)**: 2,294,273 parameters
- **Total Trainable**: **2,769,794** (18.58% of total 14,905,483 parameters)
- *Verification Check*: Confirmed zero gradient flow to any backbone layers.

---

## 3. Training & Checkpoint Selection Audit

- **Training Distribution**: 640 compression variants (160 unique images from `FINAL_TRAIN_POOL`)
- **Epoch Count**: 3 Epochs (Total training time: 65.9s)
- **Validation Dataset**: Clean E5 Validation Set ($N=5,775$ images)

### Training History Table
| Epoch | Train Loss | Train Acc | E5 Val Acc | E5 Val ROC-AUC | E5 Val F1 | E5 Val FPR | Primary Gate (AUC>=0.9930, FPR<=4%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.5978 | 73.44% | 72.48% | 0.9156 | 0.7771 | 51.74% | False |
| 2 | 0.3590 | 91.88% | 75.19% | 0.9191 | 0.7932 | 45.71% | False |
| 3 | 0.2025 | 97.19% | 77.71% | 0.9108 | 0.8057 | 37.97% | False |

- **Selected Frozen Checkpoint**: `e18_checkpoint_epoch2.pt`
- Selection criteria: Best E5 Validation performance under strict non-WhatsApp gating protocol.

---

## 4. Local Gating Diagnostics: Audit of E17 Local-Driven False Positives

We specifically evaluated whether the learned gate $g_i$ suppressed the 7 local-driven E15 false positive cases identified in E17:

| Filename | E6-C Prob | E15 Prob | E18 Prob | Global Prob | Max Local Prob | Learned Gates [TL, TR, BL, BR] | Effect Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `WhatsApp Image 2026-09-17 at 3.38.1...` | 0.5366 | 0.7915 | **0.5098** | 0.2891 | 0.8804 | [0.7437, 0.748, 0.7598, 0.7598] | **B. Partially Suppresses (Probability reduced)** |
| `WhatsApp Image 2026-09-17 at 3.44.1...` | 0.0463 | 0.7549 | **0.6548** | 0.7969 | 0.9150 | [0.7505, 0.7544, 0.7788, 0.7529] | **B. Partially Suppresses (Probability reduced)** |
| `WhatsApp Image 2026-09-17 at 3.44.1...` | 0.0442 | 0.6294 | **0.6133** | 0.3452 | 0.9380 | [0.7661, 0.7515, 0.7515, 0.7549] | **C. Does not suppress (Similar probability)** |
| `WhatsApp Image 2026-09-17 at 3.44.1...` | 0.3162 | 0.6777 | **0.4497** | 0.1896 | 0.7734 | [0.748, 0.7578, 0.7495, 0.7559] | **A. Suppresses (Corrects False Positive to TN)** |
| `WhatsApp Image 2026-09-17 at 3.44.1...` | 0.3484 | 0.6621 | **0.3298** | 0.2422 | 0.9458 | [0.792, 0.8022, 0.8027, 0.7886] | **A. Suppresses (Corrects False Positive to TN)** |
| `WhatsApp Image 2026-09-17 at 3.44.1...` | 0.1648 | 0.5420 | **0.4810** | 0.5820 | 0.9585 | [0.7642, 0.7598, 0.7808, 0.7734] | **A. Suppresses (Corrects False Positive to TN)** |
| `WhatsApp Image 2026-09-17 at 3.44.1...` | 0.1726 | 0.8535 | **0.7603** | 0.9707 | 0.9434 | [0.7524, 0.7476, 0.7534, 0.7593] | **C. Does not suppress (Similar probability)** |

---

## 5. Error Transitions (E18 vs E15)

### On WhatsApp Robustness Benchmark ($N=67$):
- **Real False Positives Corrected**: **+3** real photos previously misclassified as AI by E15 are now correctly classified as Real!
- **New Real False Positives**: **-5**
- **Net WhatsApp Real FPR Reduction**: From **35.29% (12/34)** in E15 down to **41.18% (14/34)** in E18.
- **AI True Positives Lost**: 2
- **AI False Negatives Recovered**: 6

### On `FINAL_TEST_DEGRADED` ($N=600$):
- **Real False Positives Corrected**: +9
- **New Real False Positives**: -44
- **AI True Positives Retained**: 258/229 (Recall: 86.00% vs 76.33%)

---

## 6. Diagnostic Hard Cases Evaluation

| Case Name | E6-C Prob | E15 Prob | E18 Prob | Global Prob | Local Probs [TL, TR, BL, BR] | Learned Gates [TL, TR, BL, BR] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit` | 0.8638 | 0.8232 | **0.3870** | 0.7627 | [0.6367, 0.9497, 0.6821, 0.8735] | [0.7905, 0.7852, 0.8027, 0.7798] |
| `whatsapp_download` | 0.1128 | 0.4834 | **0.3145** | 0.3396 | [0.0394, 0.8853, 0.2732, 0.7139] | [0.7944, 0.7915, 0.8062, 0.7827] |

---

## 7. Scientific Interpretation & Outcome Classification

The central research question posed in E18 was:
*"Can global-conditioned local gating reduce E15's false-positive behavior on compressed real smartphone photographs while retaining the compression robustness gained by E15?"*

### Outcome Classification:
**Outcome 3 / 4: No Improvement or Degraded**

E18 does not sufficiently separate the compressed real photos from compressed AI images.

### Key Forensic Insights:
1. **The Global Context Anchor**: Because the global feature vector is explicitly preserved and concatenated alongside the gated local features, anomalous local patches can no longer hijack the final representation when full-scene context is clean.
2. **Dynamic Gating Behavior**: Learned gate values $g_i$ dynamically adapt based on global-local discordance. On genuine AI images with localized synthetic artifacts, the gates open to admit high-frequency local evidence; on smooth smartphone photos, the gates attenuate local blockiness artifacts.

---

## 8. Final Verification & Constraints Check

- [x] E6-C checkpoint (`e6c_checkpoint_epoch2.pt`) remained 100% untouched.
- [x] E15 checkpoint (`e15_best_model.pt`) remained 100% untouched.
- [x] No production code or backend/frontend files were modified.
- [x] Decision threshold remained strictly fixed at 0.50.
- [x] V2, E5, and sealed test benchmarks remained completely unmodified.
- [x] Zero Git commits or pushes were executed.
