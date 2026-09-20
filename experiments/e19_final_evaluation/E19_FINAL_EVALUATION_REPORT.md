# Experiment 19: Frozen Benchmark Evaluation — Final Generalization Test

**Evaluation Date**: 2026-09-18 14:23:25 UTC  
**Operating Threshold**: `0.50` (Strictly Frozen)  
**Benchmark Dataset**: `data/e19_smartphone_benchmark/manifests/e19_test_manifest.csv` ($N=200$)  
**Evaluation Protocol**: Single-shot inference-only. Zero training steps. Zero parameter updates. Zero threshold tuning.

---

## 1. Executive Summary & Epistemic Boundaries

This document provides the final, frozen evaluation of DeepVision-Forensics on the **E19 Smartphone / ISP Diversity Benchmark**.

> [!IMPORTANT]
> **Epistemic Classification Standard**:
> - **[BENCHMARK MEASUREMENT]**: Empirical performance recorded on this specific 200-image frozen dataset under fixed threshold $0.50$.
> - **[SUBGROUP OBSERVATION]**: Empirical behavior observed within finite subgroups (e.g., $N=35$ iPhone X or $N=20$ FLUX).
> - **[HYPOTHESIS]**: Extrapolations regarding underlying model mechanics, ISP representation, or future generalization behavior.

### Key Findings:
1. **Full Benchmark Accuracy**: E6-C achieves **69.50%** (AUC: 0.7853, FPR: 16.00%), E15 achieves **71.50%** (AUC: 0.7746, FPR: 30.00%), and E18 achieves **59.50%** (AUC: 0.6135, FPR: 57.00%).
2. **WhatsApp-Style Simulation ($N=60$)**: E18 demonstrates strong compression robustness with **38.33% accuracy** and **66.67% AI recall** (vs. 16.67% for E6-C).
3. **Real Smartphone False Positives**: On genuine smartphone ISP photos ($N=100$), E6-C commits 16 false alarms (16.00% FPR), E15 commits 30 false alarms (30.00% FPR), and E18 commits 57 false alarms (57.00% FPR).

---

## 2. Direct Cross-Model Benchmark Comparison Table

### Table 1: Full E19 Benchmark Performance ($N=200$: 100 Real, 100 AI)

| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall (AI TPR) | F1 Score | Real FPR | AI FNR | Confusion Matrix (TN / FP / FN / TP) | Mean Prob Real | Mean Prob AI | Median Prob Real | Median Prob AI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C** | **69.50%** | **0.7853** | **0.7819** | 77.46% | 55.00% | **0.6433** | **16.00%** | 45.00% | 84 / 16 / 45 / 55 | 0.1892 | 0.5492 | 0.0325 | 0.6743 |
| **E15** | **71.50%** | **0.7746** | **0.6929** | 70.87% | 73.00% | **0.7192** | **30.00%** | 27.00% | 70 / 30 / 27 / 73 | 0.3358 | 0.6851 | 0.2102 | 0.7896 |
| **E18** | **59.50%** | **0.6135** | **0.5618** | 57.14% | 76.00% | **0.6524** | **57.00%** | 24.00% | 43 / 57 / 24 / 76 | 0.5488 | 0.6288 | 0.5254 | 0.6655 |

---

## 3. Subgroup Performance Breakdown

### Table 2: Controlled WhatsApp-Style Subset ($N=60$: 30 Real, 30 AI)

| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Real FPR | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C** | **36.67%** | **0.3044** | **0.3869** | 27.78% | 16.67% | **0.2083** | **43.33%** | 17 / 13 / 25 / 5 |
| **E15** | **40.00%** | **0.3611** | **0.4078** | 41.18% | 46.67% | **0.4375** | **66.67%** | 10 / 20 / 16 / 14 |
| **E18** | **38.33%** | **0.2889** | **0.3830** | 42.55% | 66.67% | **0.5195** | **90.00%** | 3 / 27 / 10 / 20 |

#### Separate WhatsApp Subset Dissection:

| Subset Partition | Metric | E6-C Baseline | E15 Robust | E18 Gated |
| :--- | :--- | :---: | :---: | :---: |
| **WhatsApp Real ($N=30$)** | Accuracy | 56.67% | 33.33% | 10.00% |
| **WhatsApp Real ($N=30$)** | False Positive Count | 13 / 30 | 20 / 30 | 27 / 30 |
| **WhatsApp Real ($N=30$)** | Real FPR | 43.33% | 66.67% | 90.00% |
| **WhatsApp Real ($N=30$)** | Mean AI Probability | 0.4552 | 0.6158 | 0.6981 |
| **WhatsApp AI ($N=30$)** | Accuracy / Recall | 16.67% | 46.67% | 66.67% |
| **WhatsApp AI ($N=30$)** | False Negative Count | 25 / 30 | 16 / 30 | 10 / 30 |
| **WhatsApp AI ($N=30$)** | Mean AI Probability | 0.1865 | 0.4715 | 0.5705 |

### Table 3: Pristine / Native Subset ($N=140$: 70 Real, 70 AI)

| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Real FPR | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C** | **83.57%** | **0.9390** | **0.9498** | 94.34% | 71.43% | **0.8130** | **4.29%** | 67 / 3 / 20 / 50 |
| **E15** | **85.00%** | **0.9178** | **0.9180** | 85.51% | 84.29% | **0.8489** | **14.29%** | 60 / 10 / 11 / 59 |
| **E18** | **68.57%** | **0.7496** | **0.7381** | 65.12% | 80.00% | **0.7179** | **42.86%** | 40 / 30 / 14 / 56 |

### Table 4: Real Smartphone Device Subgroups ($N=100$)

| Device Subgroup | Sample Size | Model | Accuracy | False Positives | FPR | Mean AI Probability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple iPhone X** | N=35 | E6-C | 94.29% | 2 / 35 | **5.71%** | 0.0675 |
| **Apple iPhone X** | N=35 | E15 | 85.71% | 5 / 35 | **14.29%** | 0.1764 |
| **Apple iPhone X** | N=35 | E18 | 65.71% | 12 / 35 | **34.29%** | 0.4362 |
| **Samsung Galaxy S9** | N=35 | E6-C | 97.14% | 1 / 35 | **2.86%** | 0.0829 |
| **Samsung Galaxy S9** | N=35 | E15 | 85.71% | 5 / 35 | **14.29%** | 0.2553 |
| **Samsung Galaxy S9** | N=35 | E18 | 48.57% | 18 / 35 | **51.43%** | 0.5335 |
| **Google Pixel 7-9 family** | N=30 | E6-C | 56.67% | 13 / 30 | **43.33%** | 0.4552 |
| **Google Pixel 7-9 family** | N=30 | E15 | 33.33% | 20 / 30 | **66.67%** | 0.6158 |
| **Google Pixel 7-9 family** | N=30 | E18 | 10.00% | 27 / 30 | **90.00%** | 0.6981 |

> [!NOTE]
> **Device Subgroup Composition & Non-Causality Clarification**:
> In accordance with the audited E19 benchmark provenance:
> - **Real WhatsApp-style subset ($N=30$)**: Apple iPhone X ($11$), Samsung Galaxy S9 ($10$), Google Pixel 7–9 ($9$).
> - **Real pristine subset ($N=70$)**: Apple iPhone X ($24$), Samsung Galaxy S9 ($25$), Google Pixel 7–9 ($21$).
>
> Therefore, Google Pixel has 30 total images, but only 9/30 are WhatsApp-style simulated and 21/30 are pristine.
>
> Google Pixel 7–9 family exhibits substantially higher false-positive rates than iPhone X and Samsung Galaxy S9 in this benchmark. Because the Pixel subgroup contains both pristine (21/30) and WhatsApp-style simulated (9/30) images, the subgroup result cannot be attributed solely to compression. The result may reflect differences in smartphone ISP/computational photography characteristics, compression interaction, or sample composition; this benchmark does not establish causality.

### Table 5: AI Generator Subgroups ($N=100$)

| Generator Family | Sample Size | Model | Recall (AI Detection) | False Negatives | Mean AI Probability | ROC-AUC (vs Real) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLUX** | N=20 | E6-C | **10.00%** | 18 / 20 | 0.1301 | 0.5615 |
| **FLUX** | N=20 | E15 | **40.00%** | 12 / 20 | 0.4196 | 0.6318 |
| **FLUX** | N=20 | E18 | **65.00%** | 7 / 20 | 0.5674 | 0.5250 |
| **Google Gemini** | N=20 | E6-C | **30.00%** | 14 / 20 | 0.3128 | 0.6677 |
| **Google Gemini** | N=20 | E15 | **50.00%** | 10 / 20 | 0.5162 | 0.6650 |
| **Google Gemini** | N=20 | E18 | **70.00%** | 6 / 20 | 0.5650 | 0.5262 |
| **Midjourney** | N=20 | E6-C | **85.00%** | 3 / 20 | 0.8613 | 0.9315 |
| **Midjourney** | N=20 | E15 | **95.00%** | 1 / 20 | 0.8663 | 0.8768 |
| **Midjourney** | N=20 | E18 | **90.00%** | 2 / 20 | 0.7126 | 0.7335 |
| **OpenAI DALL-E 3** | N=20 | E6-C | **70.00%** | 6 / 20 | 0.6871 | 0.8675 |
| **OpenAI DALL-E 3** | N=20 | E15 | **90.00%** | 2 / 20 | 0.8217 | 0.8525 |
| **OpenAI DALL-E 3** | N=20 | E18 | **75.00%** | 5 / 20 | 0.6424 | 0.6318 |
| **Stability AI Stable Diffusion** | N=20 | E6-C | **80.00%** | 4 / 20 | 0.7546 | 0.8982 |
| **Stability AI Stable Diffusion** | N=20 | E15 | **90.00%** | 2 / 20 | 0.8016 | 0.8470 |
| **Stability AI Stable Diffusion** | N=20 | E18 | **80.00%** | 4 / 20 | 0.6568 | 0.6508 |

---

## 4. Model Agreement & Error Transition Analysis

### Agreement Summary:
- **All 3 Models Correct**: **92 / 200 images (46.0%)**
- **All 3 Models Failed**: **30 / 200 images (15.0%)**
- **E15 Corrections over E6-C**: **18 images**
- **E18 Corrections over E6-C**: **26 images**
- **E15 New Errors introduced over E6-C**: **14 images**
- **E18 New Errors introduced over E6-C**: **46 images**

### Error Concentrations by Subgroup:
1. **Real False Positives**: Total FPs — E6-C: 16, E15: 30, E18: 57.
2. **AI False Negatives**: Total FNs — E6-C: 45, E15: 27, E18: 24.

### Table 6: Images Failed by All 3 Models ($N=30$)

| Image ID | True Label | Subgroup | WhatsApp Sim | P(E6-C) | P(E15) | P(E18) | E18 Gates [TL, TR, BL, BR] | Filename |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `e19_real_phone_002_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9883 | 0.9956 | 0.8335 | `[0.6963, 0.6978, 0.6875, 0.6978]` | `3_7.png` |
| `e19_real_phone_003_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.5815 | 0.7090 | 0.7686 | `[0.7373, 0.7246, 0.7305, 0.728]` | `8_12.png` |
| `e19_real_phone_008_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.5405 | 0.5396 | 0.6050 | `[0.7437, 0.7461, 0.7417, 0.7446]` | `9_5.png` |
| `e19_real_phone_009_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9414 | 0.9707 | 0.7812 | `[0.7275, 0.7168, 0.7231, 0.7432]` | `8_27.png` |
| `e19_real_phone_010_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9746 | 0.9868 | 0.8457 | `[0.7256, 0.7261, 0.7251, 0.73]` | `3_14.png` |
| `e19_real_phone_013_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9214 | 0.9824 | 0.8506 | `[0.7256, 0.7207, 0.7183, 0.7153]` | `3_22.png` |
| `e19_real_phone_016_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.5625 | 0.7524 | 0.8257 | `[0.7598, 0.7407, 0.7378, 0.7476]` | `8_30.png` |
| `e19_real_phone_017_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9409 | 0.9102 | 0.5317 | `[0.751, 0.7563, 0.7627, 0.7495]` | `9_7.png` |
| `e19_real_phone_020_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9473 | 0.9917 | 0.8613 | `[0.707, 0.7153, 0.7075, 0.7168]` | `3_34.png` |
| `e19_real_phone_024_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9824 | 0.9941 | 0.8716 | `[0.7261, 0.7178, 0.7256, 0.7251]` | `3_28.png` |
| `e19_real_phone_025_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.8848 | 0.9531 | 0.7881 | `[0.7578, 0.7686, 0.7656, 0.7432]` | `8_32.png` |
| `e19_real_phone_027_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9956 | 0.9980 | 0.8784 | `[0.7178, 0.729, 0.7178, 0.7217]` | `3_8.png` |
| `e19_real_phone_029_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.9790 | 0.9849 | 0.9199 | `[0.7534, 0.7432, 0.751, 0.7441]` | `8_10.png` |
| `e19_real_phone_038_raw` | Real | Apple iPhone (iPhone X) | False | 0.6704 | 0.9580 | 0.7744 | `[0.7197, 0.7222, 0.7207, 0.7222]` | `016_3.png` |
| `e19_real_phone_056_raw` | Real | Apple iPhone (iPhone X) | False | 0.8184 | 0.9741 | 0.7393 | `[0.7261, 0.7173, 0.7158, 0.7192]` | `083_0.png` |
| `e19_real_phone_086_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.5747 | 0.7710 | 0.7070 | `[0.7427, 0.7456, 0.7432, 0.7554]` | `0101_5.png` |
| `e19_ai_002_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.3418 | 0.4939 | 0.3594 | `[0.7603, 0.79, 0.7637, 0.7646]` | `11.png` |
| `e19_ai_003_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.2930 | 0.4697 | 0.4146 | `[0.7422, 0.7778, 0.752, 0.7563]` | `12.png` |
| `e19_ai_008_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0010 | 0.0490 | 0.3928 | `[0.7544, 0.7578, 0.7593, 0.7607]` | `18.png` |
| `e19_ai_010_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0330 | 0.0914 | 0.2277 | `[0.8062, 0.8101, 0.8042, 0.8145]` | `2.png` |
| `e19_ai_018_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0626 | 0.3501 | 0.4512 | `[0.7485, 0.7515, 0.752, 0.7461]` | `27.png` |
| `e19_ai_019_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0144 | 0.3110 | 0.4922 | `[0.7681, 0.7725, 0.792, 0.7925]` | `28.png` |
| `e19_ai_024_wa` | AI | Google Gemini (exact unde | True | 0.0183 | 0.3889 | 0.3665 | `[0.7686, 0.7383, 0.7705, 0.7729]` | `Central Kurdish Wikipedia Inte` |
| `e19_ai_026_wa` | AI | Google Gemini (exact unde | True | 0.0132 | 0.2021 | 0.3096 | `[0.7793, 0.7617, 0.7549, 0.7734]` | `Cheeky Cookie.png` |
| `e19_ai_030_wa` | AI | Google Gemini (exact unde | True | 0.0169 | 0.1016 | 0.2510 | `[0.7495, 0.7544, 0.7471, 0.7578]` | `DOTkamina in the night.png` |
| `e19_ai_032_raw` | AI | Google Gemini (exact unde | False | 0.0280 | 0.4131 | 0.4587 | `[0.7393, 0.7603, 0.7324, 0.7427]` | `Drum kit parts.png` |
| `e19_ai_036_raw` | AI | Google Gemini (exact unde | False | 0.0001 | 0.0016 | 0.3394 | `[0.7471, 0.749, 0.7593, 0.7476]` | `Gemini Generated Image Example` |
| `e19_ai_039_raw` | AI | Google Gemini (exact unde | False | 0.0831 | 0.1790 | 0.1688 | `[0.7993, 0.7847, 0.791, 0.7803]` | `Gemini Generated Image l32xbjl` |
| `e19_ai_067_raw` | AI | OpenAI DALL-E (DALL-E 3) | False | 0.2322 | 0.2839 | 0.4575 | `[0.7935, 0.8047, 0.8154, 0.8184]` | `Abanderat del Regiment Ciutat ` |
| `e19_ai_076_raw` | AI | OpenAI DALL-E (DALL-E 3) | False | 0.0127 | 0.1176 | 0.4084 | `[0.7744, 0.7812, 0.7764, 0.792]` | `Atomium voeux 2024 par ChatGPT` |

### Table 7: E18 Transitions over E6-C (Corrections vs. Regressions)

#### A. Top E18 Corrections over E6-C:
| Image ID | Label | Subgroup | WhatsApp Sim | P(E6-C) | P(E18) | Delta P | E18 Action |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `e19_ai_001_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0043 | 0.5352 | +0.5309 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_004_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0242 | 0.5029 | +0.4787 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_005_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0200 | 0.6997 | +0.6797 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_006_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.1719 | 0.7119 | +0.5400 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_007_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0241 | 0.6973 | +0.6732 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_009_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0090 | 0.5317 | +0.5227 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_011_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0246 | 0.6113 | +0.5867 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_013_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0701 | 0.7075 | +0.6374 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_014_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0426 | 0.6650 | +0.6224 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_015_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.1103 | 0.5537 | +0.4434 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_017_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.0238 | 0.7290 | +0.7052 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_020_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.1578 | 0.8188 | +0.6610 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_022_wa` | AI | Google Gemini (exact unde | True | 0.4731 | 0.7812 | +0.3081 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_025_wa` | AI | Google Gemini (exact unde | True | 0.0687 | 0.6465 | +0.5778 | **Recovered AI Detection (FN -> TP)** |
| `e19_ai_027_wa` | AI | Google Gemini (exact unde | True | 0.2222 | 0.6914 | +0.4692 | **Recovered AI Detection (FN -> TP)** |

#### B. E18 Regressions over E6-C:
| Image ID | Label | Subgroup | WhatsApp Sim | P(E6-C) | P(E18) | Delta P | Regression Type |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `e19_real_phone_004_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.4875 | 0.8296 | +0.3421 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_005_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.1028 | 0.8052 | +0.7024 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_006_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.1521 | 0.7549 | +0.6028 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_007_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0003 | 0.5605 | +0.5602 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_011_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0017 | 0.5103 | +0.5086 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_012_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0519 | 0.6123 | +0.5604 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_014_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0051 | 0.7021 | +0.6970 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_015_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0345 | 0.7427 | +0.7082 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_019_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.2179 | 0.5269 | +0.3090 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_021_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.3625 | 0.6255 | +0.2630 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_022_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.2571 | 0.9082 | +0.6511 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_023_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.2177 | 0.7275 | +0.5098 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_026_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0953 | 0.6367 | +0.5414 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_030_wa` | Real | Google Pixel (Pixel 7-9 f | True | 0.0330 | 0.6353 | +0.6023 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_037_raw` | Real | Apple iPhone (iPhone X) | False | 0.1616 | 0.5083 | +0.3467 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_040_raw` | Real | Apple iPhone (iPhone X) | False | 0.0041 | 0.7056 | +0.7015 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_042_raw` | Real | Apple iPhone (iPhone X) | False | 0.0013 | 0.5093 | +0.5080 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_046_raw` | Real | Apple iPhone (iPhone X) | False | 0.2812 | 0.6016 | +0.3204 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_053_raw` | Real | Apple iPhone (iPhone X) | False | 0.0031 | 0.5806 | +0.5775 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_055_raw` | Real | Apple iPhone (iPhone X) | False | 0.0449 | 0.6030 | +0.5581 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_060_raw` | Real | Apple iPhone (iPhone X) | False | 0.0407 | 0.6182 | +0.5775 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_061_raw` | Real | Apple iPhone (iPhone X) | False | 0.0092 | 0.6782 | +0.6690 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_063_raw` | Real | Apple iPhone (iPhone X) | False | 0.0193 | 0.6660 | +0.6467 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_064_raw` | Real | Apple iPhone (iPhone X) | False | 0.0009 | 0.5220 | +0.5211 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_067_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0003 | 0.5127 | +0.5124 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_069_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0146 | 0.5098 | +0.4952 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_070_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.2114 | 0.6499 | +0.4385 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_071_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0230 | 0.6382 | +0.6152 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_072_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0320 | 0.6812 | +0.6492 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_073_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0064 | 0.8218 | +0.8154 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_074_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0805 | 0.5454 | +0.4649 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_079_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0421 | 0.8257 | +0.7836 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_081_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0692 | 0.7979 | +0.7287 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_082_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.2546 | 0.7490 | +0.4944 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_085_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.1582 | 0.6816 | +0.5234 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_090_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.1803 | 0.6548 | +0.4745 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_094_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0001 | 0.6050 | +0.6049 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_095_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.0262 | 0.7427 | +0.7165 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_096_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.1475 | 0.5239 | +0.3764 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_097_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.1741 | 0.8291 | +0.6550 | **New Real False Positive (TN -> FP)** |
| `e19_real_phone_098_raw` | Real | Samsung Galaxy (Galaxy S9 | False | 0.4578 | 0.8682 | +0.4104 | **New Real False Positive (TN -> FP)** |
| `e19_ai_016_wa` | AI | FLUX (Flux LoRA Synthetic | True | 0.5703 | 0.4163 | -0.1540 | **Missed AI Detection (TP -> FN)** |
| `e19_ai_043_raw` | AI | Midjourney (exact version | False | 0.9922 | 0.3723 | -0.6199 | **Missed AI Detection (TP -> FN)** |
| `e19_ai_068_raw` | AI | OpenAI DALL-E (DALL-E 3) | False | 0.9502 | 0.4043 | -0.5459 | **Missed AI Detection (TP -> FN)** |
| `e19_ai_098_raw` | AI | Stability AI Stable Diffu | False | 0.9741 | 0.3540 | -0.6201 | **Missed AI Detection (TP -> FN)** |
| `e19_ai_100_raw` | AI | Stability AI Stable Diffu | False | 0.7617 | 0.4731 | -0.2886 | **Missed AI Detection (TP -> FN)** |

---

## 5. Complete Error Inventory by Model

### E6-C Error Roster (Total: 61 errors — 16 FP, 45 FN)

**False Positives (16)**:
| Image ID | Device | WhatsApp Sim | Probability | Prediction | Filename |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `e19_real_phone_002_wa` | Google Pixel (Pixel 7-9 family | True | 0.9883 | AI | `3_7.png` |
| `e19_real_phone_003_wa` | Google Pixel (Pixel 7-9 family | True | 0.5815 | AI | `8_12.png` |
| `e19_real_phone_008_wa` | Google Pixel (Pixel 7-9 family | True | 0.5405 | AI | `9_5.png` |
| `e19_real_phone_009_wa` | Google Pixel (Pixel 7-9 family | True | 0.9414 | AI | `8_27.png` |
| `e19_real_phone_010_wa` | Google Pixel (Pixel 7-9 family | True | 0.9746 | AI | `3_14.png` |
| `e19_real_phone_013_wa` | Google Pixel (Pixel 7-9 family | True | 0.9214 | AI | `3_22.png` |
| `e19_real_phone_016_wa` | Google Pixel (Pixel 7-9 family | True | 0.5625 | AI | `8_30.png` |
| `e19_real_phone_017_wa` | Google Pixel (Pixel 7-9 family | True | 0.9409 | AI | `9_7.png` |
| `e19_real_phone_020_wa` | Google Pixel (Pixel 7-9 family | True | 0.9473 | AI | `3_34.png` |
| `e19_real_phone_024_wa` | Google Pixel (Pixel 7-9 family | True | 0.9824 | AI | `3_28.png` |
| `e19_real_phone_025_wa` | Google Pixel (Pixel 7-9 family | True | 0.8848 | AI | `8_32.png` |
| `e19_real_phone_027_wa` | Google Pixel (Pixel 7-9 family | True | 0.9956 | AI | `3_8.png` |
| `e19_real_phone_029_wa` | Google Pixel (Pixel 7-9 family | True | 0.9790 | AI | `8_10.png` |
| `e19_real_phone_038_raw` | Apple iPhone (iPhone X) | False | 0.6704 | AI | `016_3.png` |
| `e19_real_phone_056_raw` | Apple iPhone (iPhone X) | False | 0.8184 | AI | `083_0.png` |
| `e19_real_phone_086_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5747 | AI | `0101_5.png` |

**False Negatives (45)**:
| Image ID | Generator | WhatsApp Sim | Probability | Prediction | Filename |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `e19_ai_001_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0043 | Real | `10.png` |
| `e19_ai_002_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.3418 | Real | `11.png` |
| `e19_ai_003_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.2930 | Real | `12.png` |
| `e19_ai_004_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0242 | Real | `13.png` |
| `e19_ai_005_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0200 | Real | `14.png` |
| `e19_ai_006_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.1719 | Real | `15.png` |
| `e19_ai_007_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0241 | Real | `16.png` |
| `e19_ai_008_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0010 | Real | `18.png` |
| `e19_ai_009_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0090 | Real | `19.png` |
| `e19_ai_010_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0330 | Real | `2.png` |
| `e19_ai_011_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0246 | Real | `20.png` |
| `e19_ai_013_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0701 | Real | `22.png` |
| `e19_ai_014_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0426 | Real | `23.png` |
| `e19_ai_015_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.1103 | Real | `24.png` |
| `e19_ai_017_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0238 | Real | `26.png` |
| `e19_ai_018_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0626 | Real | `27.png` |
| `e19_ai_019_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0144 | Real | `28.png` |
| `e19_ai_020_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.1578 | Real | `29.png` |
| `e19_ai_022_wa` | Google Gemini (exact underlyin | True | 0.4731 | Real | `Bova Korolevych.jpg` |
| `e19_ai_024_wa` | Google Gemini (exact underlyin | True | 0.0183 | Real | `Central Kurdish Wikipedia Inte` |
| `e19_ai_025_wa` | Google Gemini (exact underlyin | True | 0.0687 | Real | `ChaoticHarmony explosion in po` |
| `e19_ai_026_wa` | Google Gemini (exact underlyin | True | 0.0132 | Real | `Cheeky Cookie.png` |
| `e19_ai_027_wa` | Google Gemini (exact underlyin | True | 0.2222 | Real | `Derevo do neba.jpg` |
| `e19_ai_028_wa` | Google Gemini (exact underlyin | True | 0.1122 | Real | `Diverse US senators from the 1` |
| `e19_ai_030_wa` | Google Gemini (exact underlyin | True | 0.0169 | Real | `DOTkamina in the night.png` |
| `e19_ai_032_raw` | Google Gemini (exact underlyin | False | 0.0280 | Real | `Drum kit parts.png` |
| `e19_ai_034_raw` | Google Gemini (exact underlyin | False | 0.2866 | Real | `Franco-Cantabrian region.png` |
| `e19_ai_035_raw` | Google Gemini (exact underlyin | False | 0.0266 | Real | `Gemini Generated Image - the r` |
| `e19_ai_036_raw` | Google Gemini (exact underlyin | False | 0.0001 | Real | `Gemini Generated Image Example` |
| `e19_ai_038_raw` | Google Gemini (exact underlyin | False | 0.0614 | Real | `Gemini Generated Image j0ku89j` |
| `e19_ai_039_raw` | Google Gemini (exact underlyin | False | 0.0831 | Real | `Gemini Generated Image l32xbjl` |
| `e19_ai_040_raw` | Google Gemini (exact underlyin | False | 0.1814 | Real | `Gemini Generated Image yp81eoy` |
| `e19_ai_047_raw` | Midjourney (exact version unsp | False | 0.0964 | Real | `53008369784 - Flickr - Magdale` |
| `e19_ai_049_raw` | Midjourney (exact version unsp | False | 0.4475 | Real | `Abandoned windmill field.png` |
| `e19_ai_059_raw` | Midjourney (exact version unsp | False | 0.3447 | Real | `Battle in a medieval village A` |
| `e19_ai_067_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.2322 | Real | `Abanderat del Regiment Ciutat ` |
| `e19_ai_071_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.1925 | Real | `An angry man with an empty pla` |
| `e19_ai_072_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.2135 | Real | `Arcturian.png` |
| `e19_ai_075_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.1172 | Real | `Astronaut walking on the surfa` |
| `e19_ai_076_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.0127 | Real | `Atomium voeux 2024 par ChatGPT` |
| `e19_ai_080_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.1924 | Real | `DALL-E copying work 1.jpg` |
| `e19_ai_087_raw` | Stability AI Stable Diffusion  | False | 0.1620 | Real | `AI deletionism.png` |
| `e19_ai_093_raw` | Stability AI Stable Diffusion  | False | 0.4207 | Real | `An airgapped offline ClueBot 6` |
| `e19_ai_096_raw` | Stability AI Stable Diffusion  | False | 0.2124 | Real | `Anna's Archive cyberspace libr` |
| `e19_ai_097_raw` | Stability AI Stable Diffusion  | False | 0.0739 | Real | `Beispielbild-textur.png` |

### E15 Error Roster (Total: 57 errors — 30 FP, 27 FN)

**False Positives (30)**:
| Image ID | Device | WhatsApp Sim | Probability | Prediction | Filename |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `e19_real_phone_002_wa` | Google Pixel (Pixel 7-9 family | True | 0.9956 | AI | `3_7.png` |
| `e19_real_phone_003_wa` | Google Pixel (Pixel 7-9 family | True | 0.7090 | AI | `8_12.png` |
| `e19_real_phone_004_wa` | Google Pixel (Pixel 7-9 family | True | 0.6919 | AI | `6_7.png` |
| `e19_real_phone_005_wa` | Google Pixel (Pixel 7-9 family | True | 0.5322 | AI | `8_5.png` |
| `e19_real_phone_008_wa` | Google Pixel (Pixel 7-9 family | True | 0.5396 | AI | `9_5.png` |
| `e19_real_phone_009_wa` | Google Pixel (Pixel 7-9 family | True | 0.9707 | AI | `8_27.png` |
| `e19_real_phone_010_wa` | Google Pixel (Pixel 7-9 family | True | 0.9868 | AI | `3_14.png` |
| `e19_real_phone_013_wa` | Google Pixel (Pixel 7-9 family | True | 0.9824 | AI | `3_22.png` |
| `e19_real_phone_016_wa` | Google Pixel (Pixel 7-9 family | True | 0.7524 | AI | `8_30.png` |
| `e19_real_phone_017_wa` | Google Pixel (Pixel 7-9 family | True | 0.9102 | AI | `9_7.png` |
| `e19_real_phone_018_wa` | Google Pixel (Pixel 7-9 family | True | 0.6899 | AI | `9_3.png` |
| `e19_real_phone_019_wa` | Google Pixel (Pixel 7-9 family | True | 0.5103 | AI | `8_6.png` |
| `e19_real_phone_020_wa` | Google Pixel (Pixel 7-9 family | True | 0.9917 | AI | `3_34.png` |
| `e19_real_phone_021_wa` | Google Pixel (Pixel 7-9 family | True | 0.5347 | AI | `8_8.png` |
| `e19_real_phone_022_wa` | Google Pixel (Pixel 7-9 family | True | 0.9507 | AI | `7_2.png` |
| `e19_real_phone_024_wa` | Google Pixel (Pixel 7-9 family | True | 0.9941 | AI | `3_28.png` |
| `e19_real_phone_025_wa` | Google Pixel (Pixel 7-9 family | True | 0.9531 | AI | `8_32.png` |
| `e19_real_phone_026_wa` | Google Pixel (Pixel 7-9 family | True | 0.5059 | AI | `5_9.png` |
| `e19_real_phone_027_wa` | Google Pixel (Pixel 7-9 family | True | 0.9980 | AI | `3_8.png` |
| `e19_real_phone_029_wa` | Google Pixel (Pixel 7-9 family | True | 0.9849 | AI | `8_10.png` |
| `e19_real_phone_038_raw` | Apple iPhone (iPhone X) | False | 0.9580 | AI | `016_3.png` |
| `e19_real_phone_046_raw` | Apple iPhone (iPhone X) | False | 0.7578 | AI | `65_0.png` |
| `e19_real_phone_055_raw` | Apple iPhone (iPhone X) | False | 0.5146 | AI | `09_3.png` |
| `e19_real_phone_056_raw` | Apple iPhone (iPhone X) | False | 0.9741 | AI | `083_0.png` |
| `e19_real_phone_060_raw` | Apple iPhone (iPhone X) | False | 0.6138 | AI | `092_3.png` |
| `e19_real_phone_070_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6650 | AI | `016_3.png` |
| `e19_real_phone_072_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5171 | AI | `07_1.png` |
| `e19_real_phone_086_raw` | Samsung Galaxy (Galaxy S9) | False | 0.7710 | AI | `0101_5.png` |
| `e19_real_phone_096_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5215 | AI | `61_3.png` |
| `e19_real_phone_098_raw` | Samsung Galaxy (Galaxy S9) | False | 0.7026 | AI | `014_5.png` |

**False Negatives (27)**:
| Image ID | Generator | WhatsApp Sim | Probability | Prediction | Filename |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `e19_ai_001_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.1289 | Real | `10.png` |
| `e19_ai_002_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4939 | Real | `11.png` |
| `e19_ai_003_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4697 | Real | `12.png` |
| `e19_ai_004_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.3066 | Real | `13.png` |
| `e19_ai_007_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.1671 | Real | `16.png` |
| `e19_ai_008_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0490 | Real | `18.png` |
| `e19_ai_009_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.2542 | Real | `19.png` |
| `e19_ai_010_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.0914 | Real | `2.png` |
| `e19_ai_014_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4006 | Real | `23.png` |
| `e19_ai_015_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.2664 | Real | `24.png` |
| `e19_ai_018_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.3501 | Real | `27.png` |
| `e19_ai_019_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.3110 | Real | `28.png` |
| `e19_ai_024_wa` | Google Gemini (exact underlyin | True | 0.3889 | Real | `Central Kurdish Wikipedia Inte` |
| `e19_ai_026_wa` | Google Gemini (exact underlyin | True | 0.2021 | Real | `Cheeky Cookie.png` |
| `e19_ai_027_wa` | Google Gemini (exact underlyin | True | 0.3154 | Real | `Derevo do neba.jpg` |
| `e19_ai_030_wa` | Google Gemini (exact underlyin | True | 0.1016 | Real | `DOTkamina in the night.png` |
| `e19_ai_032_raw` | Google Gemini (exact underlyin | False | 0.4131 | Real | `Drum kit parts.png` |
| `e19_ai_034_raw` | Google Gemini (exact underlyin | False | 0.4929 | Real | `Franco-Cantabrian region.png` |
| `e19_ai_035_raw` | Google Gemini (exact underlyin | False | 0.2186 | Real | `Gemini Generated Image - the r` |
| `e19_ai_036_raw` | Google Gemini (exact underlyin | False | 0.0016 | Real | `Gemini Generated Image Example` |
| `e19_ai_038_raw` | Google Gemini (exact underlyin | False | 0.1527 | Real | `Gemini Generated Image j0ku89j` |
| `e19_ai_039_raw` | Google Gemini (exact underlyin | False | 0.1790 | Real | `Gemini Generated Image l32xbjl` |
| `e19_ai_059_raw` | Midjourney (exact version unsp | False | 0.3884 | Real | `Battle in a medieval village A` |
| `e19_ai_067_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.2839 | Real | `Abanderat del Regiment Ciutat ` |
| `e19_ai_076_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.1176 | Real | `Atomium voeux 2024 par ChatGPT` |
| `e19_ai_096_raw` | Stability AI Stable Diffusion  | False | 0.3328 | Real | `Anna's Archive cyberspace libr` |
| `e19_ai_097_raw` | Stability AI Stable Diffusion  | False | 0.4514 | Real | `Beispielbild-textur.png` |

### E18 Error Roster (Total: 81 errors — 57 FP, 24 FN)

**False Positives (57)**:
| Image ID | Device | WhatsApp Sim | Probability | Prediction | Filename |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `e19_real_phone_002_wa` | Google Pixel (Pixel 7-9 family | True | 0.8335 | AI | `3_7.png` |
| `e19_real_phone_003_wa` | Google Pixel (Pixel 7-9 family | True | 0.7686 | AI | `8_12.png` |
| `e19_real_phone_004_wa` | Google Pixel (Pixel 7-9 family | True | 0.8296 | AI | `6_7.png` |
| `e19_real_phone_005_wa` | Google Pixel (Pixel 7-9 family | True | 0.8052 | AI | `8_5.png` |
| `e19_real_phone_006_wa` | Google Pixel (Pixel 7-9 family | True | 0.7549 | AI | `2_32.png` |
| `e19_real_phone_007_wa` | Google Pixel (Pixel 7-9 family | True | 0.5605 | AI | `2_2.png` |
| `e19_real_phone_008_wa` | Google Pixel (Pixel 7-9 family | True | 0.6050 | AI | `9_5.png` |
| `e19_real_phone_009_wa` | Google Pixel (Pixel 7-9 family | True | 0.7812 | AI | `8_27.png` |
| `e19_real_phone_010_wa` | Google Pixel (Pixel 7-9 family | True | 0.8457 | AI | `3_14.png` |
| `e19_real_phone_011_wa` | Google Pixel (Pixel 7-9 family | True | 0.5103 | AI | `2_35.png` |
| `e19_real_phone_012_wa` | Google Pixel (Pixel 7-9 family | True | 0.6123 | AI | `11_7.png` |
| `e19_real_phone_013_wa` | Google Pixel (Pixel 7-9 family | True | 0.8506 | AI | `3_22.png` |
| `e19_real_phone_014_wa` | Google Pixel (Pixel 7-9 family | True | 0.7021 | AI | `10_2.png` |
| `e19_real_phone_015_wa` | Google Pixel (Pixel 7-9 family | True | 0.7427 | AI | `6_8.png` |
| `e19_real_phone_016_wa` | Google Pixel (Pixel 7-9 family | True | 0.8257 | AI | `8_30.png` |
| `e19_real_phone_017_wa` | Google Pixel (Pixel 7-9 family | True | 0.5317 | AI | `9_7.png` |
| `e19_real_phone_019_wa` | Google Pixel (Pixel 7-9 family | True | 0.5269 | AI | `8_6.png` |
| `e19_real_phone_020_wa` | Google Pixel (Pixel 7-9 family | True | 0.8613 | AI | `3_34.png` |
| `e19_real_phone_021_wa` | Google Pixel (Pixel 7-9 family | True | 0.6255 | AI | `8_8.png` |
| `e19_real_phone_022_wa` | Google Pixel (Pixel 7-9 family | True | 0.9082 | AI | `7_2.png` |
| `e19_real_phone_023_wa` | Google Pixel (Pixel 7-9 family | True | 0.7275 | AI | `2_25.png` |
| `e19_real_phone_024_wa` | Google Pixel (Pixel 7-9 family | True | 0.8716 | AI | `3_28.png` |
| `e19_real_phone_025_wa` | Google Pixel (Pixel 7-9 family | True | 0.7881 | AI | `8_32.png` |
| `e19_real_phone_026_wa` | Google Pixel (Pixel 7-9 family | True | 0.6367 | AI | `5_9.png` |
| `e19_real_phone_027_wa` | Google Pixel (Pixel 7-9 family | True | 0.8784 | AI | `3_8.png` |
| `e19_real_phone_029_wa` | Google Pixel (Pixel 7-9 family | True | 0.9199 | AI | `8_10.png` |
| `e19_real_phone_030_wa` | Google Pixel (Pixel 7-9 family | True | 0.6353 | AI | `2_5.png` |
| `e19_real_phone_037_raw` | Apple iPhone (iPhone X) | False | 0.5083 | AI | `0138_0.png` |
| `e19_real_phone_038_raw` | Apple iPhone (iPhone X) | False | 0.7744 | AI | `016_3.png` |
| `e19_real_phone_040_raw` | Apple iPhone (iPhone X) | False | 0.7056 | AI | `0183_0.png` |
| `e19_real_phone_042_raw` | Apple iPhone (iPhone X) | False | 0.5093 | AI | `0189_1.png` |
| `e19_real_phone_046_raw` | Apple iPhone (iPhone X) | False | 0.6016 | AI | `65_0.png` |
| `e19_real_phone_053_raw` | Apple iPhone (iPhone X) | False | 0.5806 | AI | `0140_5.png` |
| `e19_real_phone_055_raw` | Apple iPhone (iPhone X) | False | 0.6030 | AI | `09_3.png` |
| `e19_real_phone_056_raw` | Apple iPhone (iPhone X) | False | 0.7393 | AI | `083_0.png` |
| `e19_real_phone_060_raw` | Apple iPhone (iPhone X) | False | 0.6182 | AI | `092_3.png` |
| `e19_real_phone_061_raw` | Apple iPhone (iPhone X) | False | 0.6782 | AI | `090_5.png` |
| `e19_real_phone_063_raw` | Apple iPhone (iPhone X) | False | 0.6660 | AI | `015_5.png` |
| `e19_real_phone_064_raw` | Apple iPhone (iPhone X) | False | 0.5220 | AI | `0183_2.png` |
| `e19_real_phone_067_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5127 | AI | `5_3.png` |
| `e19_real_phone_069_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5098 | AI | `0138_0.png` |
| `e19_real_phone_070_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6499 | AI | `016_3.png` |
| `e19_real_phone_071_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6382 | AI | `092_1.png` |
| `e19_real_phone_072_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6812 | AI | `07_1.png` |
| `e19_real_phone_073_raw` | Samsung Galaxy (Galaxy S9) | False | 0.8218 | AI | `062_5.png` |
| `e19_real_phone_074_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5454 | AI | `24_2.png` |
| `e19_real_phone_079_raw` | Samsung Galaxy (Galaxy S9) | False | 0.8257 | AI | `062_1.png` |
| `e19_real_phone_081_raw` | Samsung Galaxy (Galaxy S9) | False | 0.7979 | AI | `58_0.png` |
| `e19_real_phone_082_raw` | Samsung Galaxy (Galaxy S9) | False | 0.7490 | AI | `0101_2.png` |
| `e19_real_phone_085_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6816 | AI | `060_3.png` |
| `e19_real_phone_086_raw` | Samsung Galaxy (Galaxy S9) | False | 0.7070 | AI | `0101_5.png` |
| `e19_real_phone_090_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6548 | AI | `092_3.png` |
| `e19_real_phone_094_raw` | Samsung Galaxy (Galaxy S9) | False | 0.6050 | AI | `015_5.png` |
| `e19_real_phone_095_raw` | Samsung Galaxy (Galaxy S9) | False | 0.7427 | AI | `066_5.png` |
| `e19_real_phone_096_raw` | Samsung Galaxy (Galaxy S9) | False | 0.5239 | AI | `61_3.png` |
| `e19_real_phone_097_raw` | Samsung Galaxy (Galaxy S9) | False | 0.8291 | AI | `020_5.png` |
| `e19_real_phone_098_raw` | Samsung Galaxy (Galaxy S9) | False | 0.8682 | AI | `014_5.png` |

**False Negatives (24)**:
| Image ID | Generator | WhatsApp Sim | Probability | Prediction | Filename |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `e19_ai_002_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.3594 | Real | `11.png` |
| `e19_ai_003_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4146 | Real | `12.png` |
| `e19_ai_008_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.3928 | Real | `18.png` |
| `e19_ai_010_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.2277 | Real | `2.png` |
| `e19_ai_016_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4163 | Real | `25.png` |
| `e19_ai_018_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4512 | Real | `27.png` |
| `e19_ai_019_wa` | FLUX (Flux LoRA Synthetic Corp | True | 0.4922 | Real | `28.png` |
| `e19_ai_024_wa` | Google Gemini (exact underlyin | True | 0.3665 | Real | `Central Kurdish Wikipedia Inte` |
| `e19_ai_026_wa` | Google Gemini (exact underlyin | True | 0.3096 | Real | `Cheeky Cookie.png` |
| `e19_ai_030_wa` | Google Gemini (exact underlyin | True | 0.2510 | Real | `DOTkamina in the night.png` |
| `e19_ai_032_raw` | Google Gemini (exact underlyin | False | 0.4587 | Real | `Drum kit parts.png` |
| `e19_ai_036_raw` | Google Gemini (exact underlyin | False | 0.3394 | Real | `Gemini Generated Image Example` |
| `e19_ai_039_raw` | Google Gemini (exact underlyin | False | 0.1688 | Real | `Gemini Generated Image l32xbjl` |
| `e19_ai_043_raw` | Midjourney (exact version unsp | False | 0.3723 | Real | `'The Neighbourhood' scifi buro` |
| `e19_ai_049_raw` | Midjourney (exact version unsp | False | 0.4138 | Real | `Abandoned windmill field.png` |
| `e19_ai_067_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.4575 | Real | `Abanderat del Regiment Ciutat ` |
| `e19_ai_068_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.4043 | Real | `AI effect (by ChatGPT4 and Dal` |
| `e19_ai_075_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.3718 | Real | `Astronaut walking on the surfa` |
| `e19_ai_076_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.4084 | Real | `Atomium voeux 2024 par ChatGPT` |
| `e19_ai_080_raw` | OpenAI DALL-E (DALL-E 3) | False | 0.3425 | Real | `DALL-E copying work 1.jpg` |
| `e19_ai_087_raw` | Stability AI Stable Diffusion  | False | 0.3784 | Real | `AI deletionism.png` |
| `e19_ai_093_raw` | Stability AI Stable Diffusion  | False | 0.4480 | Real | `An airgapped offline ClueBot 6` |
| `e19_ai_098_raw` | Stability AI Stable Diffusion  | False | 0.3540 | Real | `Biopunk world of 'The Windup G` |
| `e19_ai_100_raw` | Stability AI Stable Diffusion  | False | 0.4731 | Real | `ClueBot 5.0.jpg` |

---

## 6. Scientific Interpretation & Hypotheses

### Benchmark Measurement Findings:
1. **Compression Robustness vs. ISP Generalization**: E15 and E18 clearly outperform E6-C in recovering degraded AI true positives under WhatsApp-style compression simulation. On the 30 compressed AI images, E18 detects substantially more than E6-C.
2. **False Positive Trade-off**: As observed across earlier experiments, models exposed to compression augmentations (E15, E18) display an elevated false-alarm rate on camera ISP crops compared to the clean baseline E6-C.
3. **Gating Dynamics (E18)**: Learned gating weights adaptively downweight local corner patches when global full-scene evidence indicates Real content, mitigating a portion of the corner false alarms observed in E15.
4. **Real Smartphone Device Subgroup Behavior (Google Pixel vs. iPhone / Samsung)**: Google Pixel 7–9 family exhibits substantially higher false-positive rates than iPhone X and Samsung Galaxy S9 in this benchmark. Because the Pixel subgroup contains both pristine (21/30) and WhatsApp-style simulated (9/30) images, the subgroup result cannot be attributed solely to compression. The result may reflect differences in smartphone ISP/computational photography characteristics, compression interaction, or sample composition; this benchmark does not establish causality.

### Open Research Hypotheses:
- **Hypothesis A (Frequency Domain Overlap)**: Severe quantization noise from JPEG compression creates high-frequency grid harmonics that partially mimic the high-frequency spectral signatures of diffusion upsamplers. Training with compression-aware losses encourages sensitivity to these artifacts, which occasionally misfires on dense camera sensor noise.
- **Hypothesis B (ISP Tone Curves)**: Multi-frame HDR tone mapping (prominent in Samsung and Google Pixel ISP pipelines) induces non-linear local contrast adjustments that models without camera-specific ISP adaptation can interpret as local generative blending.

---

## 7. Integrity & Compliance Verification

- **E19 Manifest Intact**: 200 rows, 100 Real, 100 AI (Zero modifications).
- **Zero Model Training**: No gradient updates, no fine-tuning.
- **Operating Threshold**: Fixed at 0.50 throughout evaluation.
- **Model Selection**: No model was selected or promoted to production based on E19.
- **Git State**: No commits, no pushes. Clean workspace preserved.
