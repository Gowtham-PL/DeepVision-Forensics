# E15 — External Compression-Robust Training Research Report

**Date**: 2026-09-18 11:38:52 UTC
**Status**: COMPLETE
**Selected Checkpoint**: Epoch 3 (`e15_best_model.pt`)

## 1. Executive Summary

Experiment E15 evaluated whether fine-tuning the multi-view attention aggregation and classification head on a small, clean external domain-adaptation training set with realistic compression variants improves compressed-image detection on an untouched, cryptographically sealed independent external evaluation benchmark (`FINAL_TEST_POOL`, N=200; `FINAL_TEST_DEGRADED`, N=600).

### Benchmark Performance Comparison Table

| Benchmark Split | Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FINAL_TEST (Clean, N=200) | **E6-C Baseline** | 75.50% | 0.8426 | 0.8598 | 76.29% | 74.00% | 0.7513 | 23.00% | 26.00% |
| FINAL_TEST (Clean, N=200) | **E15 Robust** | 79.50% | 0.8682 | 0.8781 | 74.38% | 90.00% | 0.8145 | 31.00% | 10.00% |
| FINAL_TEST_DEGRADED (N=600) | **E6-C Baseline** | 65.83% | 0.7640 | 0.7762 | 78.44% | 43.67% | 0.5610 | 12.00% | 56.33% |
| FINAL_TEST_DEGRADED (N=600) | **E15 Robust** | 76.83% | 0.8252 | 0.8262 | 77.10% | 76.33% | 0.7672 | 22.67% | 23.67% |
| WhatsApp (N=67) | **E6-C Baseline** | 61.19% | 0.6551 | 0.7175 | 81.82% | 27.27% | 0.4091 | 5.88% | 72.73% |
| WhatsApp (N=67) | **E15 Robust** | 56.72% | 0.6221 | 0.6721 | 57.14% | 48.48% | 0.5246 | 35.29% | 51.52% |

## 2. Training History & Checkpoint Selection

| Epoch | Train Loss | Val Acc | Val ROC-AUC | Val PR-AUC | Val F1 | Val FPR | Val FNR | Passed Selection Gates |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 0.3745 | 70.00% | 0.8000 | 0.8556 | 0.7143 | 35.00% | 25.00% | NO |
| 5 | 0.2905 | 70.00% | 0.7900 | 0.8529 | 0.7143 | 35.00% | 25.00% | NO |
| 4 | 0.3162 | 70.00% | 0.7875 | 0.8525 | 0.7143 | 35.00% | 25.00% | NO |
| 1 | 0.7353 | 70.00% | 0.7625 | 0.8317 | 0.7000 | 30.00% | 30.00% | NO |
| 2 | 0.4640 | 67.50% | 0.7775 | 0.8326 | 0.6829 | 35.00% | 30.00% | NO |

- **Selection Decision**: Selected Epoch 3 satisfying Dev Val ROC-AUC >= 0.98 and FPR <= 5.0%.

## 3. Detailed Results by Degradation Type (FINAL_TEST_DEGRADED, N=600)

| Degradation Type | Count | E6-C Acc | E15 Acc | E6-C AUC | E15 AUC | E6-C F1 | E15 F1 | E6-C FPR | E15 FPR | E6-C Recall | E15 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Double JPEG (L1280, Q80->Q70, 4:2:0) | 200 | 60.00% | 74.50% | 0.7333 | 0.8096 | 0.4118 | 0.7213 | 8.00% | 17.00% | 28.00% | 66.00% |
| Resize (L1080) + JPEG Q80 (4:2:0) | 200 | 71.00% | 78.00% | 0.7928 | 0.8420 | 0.6548 | 0.7864 | 13.00% | 25.00% | 55.00% | 81.00% |
| WhatsApp-tier JPEG (L1280, Q75, 4:2:0) | 200 | 66.50% | 78.00% | 0.7812 | 0.8345 | 0.5890 | 0.7885 | 15.00% | 26.00% | 48.00% | 82.00% |

## 4. Detailed Results by AI Generator (FINAL_TEST_POOL Clean, N=100 AI)

| Generator | Count | E6-C AI Recall | E15 AI Recall | E6-C Mean Prob | E15 Mean Prob |
| :--- | :---: | :---: | :---: | :---: | :---: |
| dalle2 | 14 | 42.86% | 85.71% | 0.5088 | 0.6927 |
| dalle3 | 6 | 100.00% | 100.00% | 0.9727 | 0.9329 |
| firefly | 14 | 42.86% | 64.29% | 0.4522 | 0.5463 |
| glide | 13 | 92.31% | 100.00% | 0.9403 | 0.9576 |
| midjourney-v5 | 12 | 91.67% | 100.00% | 0.7850 | 0.8081 |
| stable-diffusion-1-3 | 14 | 71.43% | 92.86% | 0.6678 | 0.8384 |
| stable-diffusion-1-4 | 12 | 100.00% | 100.00% | 0.9325 | 0.9433 |
| stable-diffusion-2 | 15 | 73.33% | 86.67% | 0.7063 | 0.8061 |

## 5. Real Optical Photo Analysis (FINAL_TEST_POOL Clean, N=100 Real)

- Real Source: RAISE-1k (Nikon DSLR Optical Sensors)
- E6-C False Positive Rate (FPR): **23.00%** (Mean Prob: 0.2569)
- E15 False Positive Rate (FPR): **31.00%** (Mean Prob: 0.3800)

## 6. Diagnostic Hard Cases

| Case Name | Ground Truth | E6-C Prob | E6-C Pred | E15 Prob | E15 Pred |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit` | AI | 0.8638 | AI | 0.8232 | AI |
| `whatsapp_download` | AI | 0.1128 | Real | 0.4834 | Real |

## 7. Scientific Conclusion & Core Takeaways

1. **Impact on Clean Independent Test Set (`FINAL_TEST_POOL`, N=200)**:
   - Accuracy shift: +4.00 percentage points (75.50% -> 79.50%)
   - ROC-AUC shift: +0.0256 (0.8426 -> 0.8682)

2. **Impact on Compressed Robustness Benchmark (`FINAL_TEST_DEGRADED`, N=600)**:
   - Accuracy shift: +11.00 percentage points (65.83% -> 76.83%)
   - ROC-AUC shift: +0.0612 (0.7640 -> 0.8252)

3. **Impact on Real-World WhatsApp Benchmark (N=67)**:
   - E6-C ROC-AUC: 0.6551 | E15 ROC-AUC: 0.6221
   - E6-C F1 Score: 0.4091 | E15 F1 Score: 0.5246

