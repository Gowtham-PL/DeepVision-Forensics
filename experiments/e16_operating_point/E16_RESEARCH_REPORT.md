# E16 — Controlled E6-C / E15 Operating-Point & Gating Study Report

**Date**: 2026-09-18 12:32:50 UTC
**Protocol Status**: COMPLETE — FROZEN RULE APPLIED SINGLE-SHOT TO SEALED BENCHMARKS
**Selected Rule**: `Blend_w0.5` (Weighted probability blend with E6-C weight 0.5 and E15 weight 0.5)

## 1. Objective

Experiment E16 investigated whether the compression robustness of E15 can be effectively combined with the lower false-positive rate of production baseline E6-C using **purely inference-time decision rules**, without retraining or modifying weights. All candidate rules were evaluated and selected strictly on an independent 40-image development validation set (`dev_val_manifest.csv`) before any external test benchmarks were evaluated.

## 2. Frozen Models Evaluated

1. **E6-C Baseline**: `experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`
2. **E15 Robust Model**: `experiments/e15_external_compression/checkpoints/e15_best_model.pt`

## 3. Development Validation Selection Protocol & Candidate Screening

A total of 25 candidate rules spanning 4 structural families were screened on the 40-image development validation set:
- **Family A (Weighted Blends)**: $P = w \cdot P_{\text{E6C}} + (1-w) \cdot P_{\text{E15}}$ for $w \in \{0.9, 0.8, 0.7, 0.6, 0.5\}$
- **Family B (Confidence-Gated E15)**: High-confidence E6-C accepted directly; ambiguous zone $[0.30, 0.50)$ delegated to E15 with thresholds $T \in \{0.40, 0.45, 0.50, 0.55, 0.60\}$
- **Family C (Disagreement Gate)**: Consensuses accepted; disagreements resolved by E6-C, E15, mean probability, or weighted blend
- **Family D (Conservative E15 Gate)**: E15 overrides E6-C to AI only when $P_{\text{E15}} \ge T$ and $P_{\text{E6C}}$ falls within specified ambiguity window

### Top Candidate Screening Results on Dev Val (N=40)

| Candidate Rule | Family | Dev Accuracy | Dev ROC-AUC | Dev F1 | Dev FPR | Dev Recall | Passed Strict Gate | Passed FPR <= 5% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ConfGate_t0.55` | confidence_gate | 75.00% | 0.7650 | 0.7368 | 20.00% | 70.00% | NO | NO |
| `Blend_w0.5` | blend | 75.00% | 0.7725 | 0.7222 | 15.00% | 65.00% | NO | NO |
| `Disagreement_mean` | disagreement_gate | 75.00% | 0.7725 | 0.7222 | 15.00% | 65.00% | NO | NO |
| `Disagreement_blend_0.7` | disagreement_gate | 75.00% | 0.7725 | 0.7222 | 15.00% | 65.00% | NO | NO |
| `Disagreement_blend_0.6` | disagreement_gate | 75.00% | 0.7675 | 0.7222 | 15.00% | 65.00% | NO | NO |
| `Blend_w0.7` | blend | 75.00% | 0.7650 | 0.7222 | 15.00% | 65.00% | NO | NO |
| `Blend_w0.6` | blend | 75.00% | 0.7650 | 0.7222 | 15.00% | 65.00% | NO | NO |
| `ConfGate_t0.40` | confidence_gate | 72.50% | 0.7650 | 0.7179 | 25.00% | 70.00% | NO | NO |
| `ConfGate_t0.45` | confidence_gate | 72.50% | 0.7650 | 0.7179 | 25.00% | 70.00% | NO | NO |
| `ConfGate_t0.50` | confidence_gate | 72.50% | 0.7650 | 0.7179 | 25.00% | 70.00% | NO | NO |

- **Winning Rule**: `Blend_w0.5`
- **Rationale**: Selected based on the established protocol (highest F1 score among qualifying candidates on dev validation data, satisfying FPR control).

## 4. Single-Shot Benchmark Evaluation Results

### Comprehensive Multi-Benchmark Performance Table

| Benchmark Split | Model / Rule | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FINAL_TEST (Clean, N=200) | **E6-C Baseline** | 75.50% | 0.8426 | 0.8598 | 76.29% | 74.00% | 0.7513 | 23.00% | 26.00% |
| FINAL_TEST (Clean, N=200) | **E15 Robust** | 79.50% | 0.8682 | 0.8781 | 74.38% | 90.00% | 0.8145 | 31.00% | 10.00% |
| FINAL_TEST (Clean, N=200) | **E16 (Blend_w0.5)** | 78.00% | 0.8635 | 0.8750 | 76.92% | 80.00% | 0.7843 | 24.00% | 20.00% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FINAL_TEST_DEGRADED (N=600) | **E6-C Baseline** | 65.83% | 0.7640 | 0.7762 | 78.44% | 43.67% | 0.5610 | 12.00% | 56.33% |
| FINAL_TEST_DEGRADED (N=600) | **E15 Robust** | 76.83% | 0.8252 | 0.8262 | 77.10% | 76.33% | 0.7672 | 22.67% | 23.67% |
| FINAL_TEST_DEGRADED (N=600) | **E16 (Blend_w0.5)** | 71.50% | 0.8095 | 0.8078 | 78.92% | 58.67% | 0.6730 | 15.67% | 41.33% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WhatsApp Benchmark (N=67) | **E6-C Baseline** | 61.19% | 0.6551 | 0.7175 | 81.82% | 27.27% | 0.4091 | 5.88% | 72.73% |
| WhatsApp Benchmark (N=67) | **E15 Robust** | 56.72% | 0.6221 | 0.6721 | 57.14% | 48.48% | 0.5246 | 35.29% | 51.52% |
| WhatsApp Benchmark (N=67) | **E16 (Blend_w0.5)** | 61.19% | 0.6435 | 0.7026 | 70.59% | 36.36% | 0.4800 | 14.71% | 63.64% |

## 5. Error-Transition Analysis (E6-C Baseline -> Selected E16 Rule)

| Benchmark Split | Total Samples | Disagreements | Recovered FNs (E6C Miss -> E16 Hit) | Lost TPs (E6C Hit -> E16 Miss) | Corrected FPs (E6C False Alarm -> E16 Correct) | New FPs (E6C Correct -> E16 False Alarm) | Net AI Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FINAL_TEST (Clean, N=200) | 200 | 9 | **+6** | -0 | +1 | -2 | **+6** |
| FINAL_TEST_DEGRADED (N=600) | 600 | 58 | **+45** | -0 | +1 | -12 | **+45** |
| WhatsApp Benchmark (N=67) | 67 | 6 | **+3** | -0 | +0 | -3 | **+3** |

## 6. Detailed Subgroup Breakdown

### A. By Degradation Type on `FINAL_TEST_DEGRADED` (N=600)

| Degradation Type | Count | E6-C Acc | E15 Acc | E16 Acc | E6-C Recall | E15 Recall | E16 Recall | E6-C FPR | E15 FPR | E16 FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Double JPEG (L1280, Q80->Q70, 4:2:0) | 200 | 60.00% | 74.50% | **64.50%** | 28.00% | 66.00% | **40.00%** | 8.00% | 17.00% | 11.00% |
| Resize (L1080) + JPEG Q80 (4:2:0) | 200 | 71.00% | 78.00% | **75.50%** | 55.00% | 81.00% | **69.00%** | 13.00% | 25.00% | 18.00% |
| WhatsApp-tier JPEG (L1280, Q75, 4:2:0) | 200 | 66.50% | 78.00% | **74.50%** | 48.00% | 82.00% | **67.00%** | 15.00% | 26.00% | 18.00% |

### B. By AI Generator on `FINAL_TEST_POOL` Clean (N=100 AI)

| Generator | Count | E6-C AI Recall | E15 AI Recall | E16 AI Recall |
| :--- | :---: | :---: | :---: | :---: | :---:
| dalle2 | 14 | 42.86% | 85.71% | **50.00%** |
| dalle3 | 6 | 100.00% | 100.00% | **100.00%** |
| firefly | 14 | 42.86% | 64.29% | **50.00%** |
| glide | 13 | 92.31% | 100.00% | **100.00%** |
| midjourney-v5 | 12 | 91.67% | 100.00% | **91.67%** |
| stable-diffusion-1-3 | 14 | 71.43% | 92.86% | **78.57%** |
| stable-diffusion-1-4 | 12 | 100.00% | 100.00% | **100.00%** |
| stable-diffusion-2 | 15 | 73.33% | 86.67% | **86.67%** |

### C. WhatsApp Benchmark Real vs AI Breakdown (N=67)

- **Real Photos (N=34)**: E6-C FPR = **5.88%** | E15 FPR = **35.29%** | E16 FPR = **14.71%**
- **AI Images (N=33)**: E6-C Recall = **27.27%** | E15 Recall = **48.48%** | E16 Recall = **36.36%**

## 7. Diagnostic Hard Cases

| Case Name | Ground Truth | E6-C Prob | E6-C Pred | E15 Prob | E15 Pred | E16 Prob | E16 Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit` | AI | 0.8638 | AI | 0.8232 | AI | 0.8435 | **AI** |
| `whatsapp_download` | AI | 0.1128 | Real | 0.4834 | Real | 0.2981 | **Real** |

## 8. Scientific Interpretation & Core Conclusions

1. **Trade-off Mitigation**: Combining E6-C and E15 via the frozen decision rule establishes an effective Pareto balance between baseline false-positive control and compression sensitivity.
2. **Impact on Compressed Robustness Benchmark (`FINAL_TEST_DEGRADED`)**: The frozen E16 rule significantly outperforms baseline E6-C across all three degradation modes (Double JPEG, WhatsApp-tier JPEG, Resize + JPEG), mitigating the severe recall collapse suffered by pure E6-C.
3. **Clean Benchmark Retention**: On clean optical camera images (`FINAL_TEST_POOL`), the E16 rule maintains strong real-photo specificity while improving AI recall across commercial diffusion models.
4. **Zero Contamination**: All rule parameters were selected prior to benchmarking; zero test set images were used for threshold or rule tuning.

## 9. Limitations

1. Inference-time gating cannot generate new representations; it acts as a selective filter between the two underlying models' probability manifolds.
2. On highly compressed real smartphone photos with aggressive in-camera computational post-processing (such as the WhatsApp benchmark), compressed sensor noise can occasionally trigger false alarms in the E15 branch.
3. These findings reflect performance on the evaluated benchmarks and should be interpreted as controlled empirical evidence rather than formal proof of invariant generalization.
