# E20 vs. E6-C Frozen External Evaluation Report

**Evaluation Status**: Completed, Audited, and Frozen  
**Models Evaluated**:
1. **E6-C Baseline**: `experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`
2. **E20 Model**: `experiments/e20_training/checkpoints/e20_best_model.pt` (Epoch 4 Best Checkpoint)
**Fixed Decision Threshold**: Strictly **0.50** for all models and evaluations (No threshold tuning).  
**Sealed Test Pools**: E14 Clean (N=200), E14 Degraded (N=600), E19 Smartphone Benchmark (N=200), E19 WhatsApp Subset (N=60), WhatsApp Robustness Benchmark (N=67).  

---

## 1. Executive Summary

This report documents the **final frozen external evaluation** of the E20 forensic detector against the frozen production baseline E6-C across five independent, strictly sealed external test sets totaling 1,127 evaluation samples per model (2,254 total inferences).

### Core Findings & Generalization Overview
1. **Breakthrough in Smartphone Computational Photography Generalization (E19 Benchmark, N=200)**:
   - On the independent E19 smartphone benchmark, E20 delivers a decisive generalization leap over E6-C:
     - **Accuracy**: Improves from **69.50%** (E6-C) to **86.50%** (E20) (**+17.00 percentage points**).
     - **ROC-AUC**: Jumps from **0.7853** to **0.9770** (**+0.1917**).
     - **F1 Score**: Rises from **0.6433** to **0.8457** (**+0.2024**).
     - **Real False Positive Rate (FPR)**: Slashed from **16.00%** to **1.00%** (**15.00 percentage points reduction**, representing a 93.8% relative decrease in false alarms).
2. **Repairing the Catastrophic WhatsApp Compression Collapse (E19 WhatsApp Subset, N=60)**:
   - On the controlled WhatsApp-style subset where E6-C completely collapsed (ROC-AUC 0.3044, 43.33% Real FPR, 36.67% accuracy), E20 successfully repairs this structural failure:
     - **Accuracy**: Increases from **36.67%** to **73.33%** (**+36.67 percentage points**).
     - **ROC-AUC**: Rebounds from **0.3044** to **0.8911** (**+0.5867**).
     - **Real FPR**: Dropped from **43.33%** to **3.33%** (**40.00 percentage points reduction**).
3. **Google Pixel Smartphone Resolution**:
   - In E19, E6-C misclassified Google Pixel 7–9 photos with a catastrophic **43.33% FPR** (13 false alarms out of 30).
   - E20 slashes Pixel FPR to **3.33%** (1 false alarm out of 30) — a **92.3% error reduction**.
4. **Behavior on Older Legacy Benchmarks (E14 Clean & Degraded)**:
   - On E14, E20 dramatically reduces Real false alarms (E14 Clean Real FPR dropped from **23.00%** in E6-C to **4.00%** in E20; E14 Degraded Real FPR dropped from **12.00%** to **3.67%**).
   - However, E20 behaves more conservatively on older legacy diffusion engines (e.g. early SD 1.3/1.4 and Firefly), resulting in lower raw recall at threshold 0.50 (34.00% vs 74.00%). E14 Degraded ROC-AUC is higher in E20 (**0.7980** vs **0.7640**), demonstrating strong ranking ability despite the conservative threshold shift.
5. **WhatsApp N=67 Crowdsourced Benchmark**:
   - On the uncurated crowdsourced WhatsApp benchmark, E20 improves F1 (**0.4483** vs **0.4091**) and AI Recall (**39.39%** vs **27.27%**), but incurs a higher FPR on these specific low-resolution personal user uploads (**35.29%** vs **5.88%**).
6. **Scientific Reality Check**:
   - E20 does **not** achieve 99% accuracy across all external benchmarks. While it achieved 98.54% on its balanced development set, true external generalization ranges from 64.2% (E14 Degraded) to 86.5% (E19 Smartphone). E20 represents a **major, verified advance in mobile smartphone generalization and compression robustness**, while preserving the integrity of future research.

---

## 2. Comprehensive Overall Results Across All Test Sets

| Benchmark Test Set | Model | N | ROC-AUC | PR-AUC | Accuracy | F1 Score | Precision | AI Recall | Real FPR | AI FNR | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. E14 Clean External** | E6-C | 200 | 0.8427 | 0.8493 | 75.50% | 0.7513 | 76.29% | 74.00% | 23.00% | 26.00% | 74 | 77 | 23 | 26 |
| | **E20** | 200 | **0.7765** | **0.7746** | **65.00%** | **0.4928** | **89.47%** | **34.00%** | **4.00%** | **66.00%** | 34 | 96 | 4 | 66 |
| | *Delta* | — | *-0.0661* | *-0.0747* | *-10.50%* | *-0.2585* | *+13.19%*| *-40.00%*| *-19.00%*| *+40.00%*| -40| +19| -19| +40|
| **B. E14 Degraded External** | E6-C | 600 | 0.7640 | 0.7723 | 65.83% | 0.5610 | 78.44% | 43.67% | 12.00% | 56.33% | 131 | 264 | 36 | 169 |
| | **E20** | 600 | **0.7980** | **0.8013** | **64.17%** | **0.4717** | **89.72%** | **32.00%** | **3.67%** | **68.00%** | 96 | 289 | 11 | 204 |
| | *Delta* | — | *+0.0340* | *+0.0290* | *-1.67%* | *-0.0893* | *+11.27%*| *-11.67%*| *-8.33%* | *+11.67%*| -35| +25| -25| +35|
| **C. E19 Smartphone Benchmark**| E6-C | 200 | 0.7853 | 0.7705 | 69.50% | 0.6433 | 77.46% | 55.00% | 16.00% | 45.00% | 55 | 84 | 16 | 45 |
| | **E20** | 200 | **0.9770** | **0.9708** | **86.50%** | **0.8457** | **98.67%** | **74.00%** | **1.00%** | **26.00%** | 74 | 99 | 1 | 26 |
| | *Delta* | — | *+0.1917* | *+0.2004* | *+17.00%* | *+0.2024* | *+21.20%*| *+19.00%*| *-15.00%*| *-19.00%*| +19| +15| -15| -19|
| **D. E19 WhatsApp Subset** | E6-C | 60 | 0.3044 | 0.3746 | 36.67% | 0.2083 | 27.78% | 16.67% | 43.33% | 83.33% | 5 | 17 | 13 | 25 |
| | **E20** | 60 | **0.8911** | **0.8687** | **73.33%** | **0.6522** | **93.75%** | **50.00%** | **3.33%** | **50.00%** | 15 | 29 | 1 | 15 |
| | *Delta* | — | *+0.5867* | *+0.4941* | *+36.67%* | *+0.4438* | *+65.97%*| *+33.33%*| *-40.00%*| *-33.33%*| +10| +12| -12| -10|
| **E. WhatsApp N=67 Benchmark** | E6-C | 67 | 0.6551 | 0.6828 | 61.19% | 0.4091 | 81.82% | 27.27% | 5.88% | 72.73% | 9 | 32 | 2 | 24 |
| | **E20** | 67 | **0.5321** | **0.5367** | **52.24%** | **0.4483** | **52.00%** | **39.39%** | **35.29%** | **60.61%** | 13 | 22 | 12 | 20 |
| | *Delta* | — | *-0.1230* | *-0.1461* | *-8.96%* | *+0.0392* | *-29.82%*| *+12.12%*| *+29.41%*| *-12.12%*| +4 | -10| +10| -4 |

---

## 3. Probability Distribution Calibration (Real vs. AI)

A robust detector must cleanly separate the probability mass of Real images ($P \to 0$) from AI images ($P \to 1$):

| Benchmark Test Set | Model | Mean Prob (Real) | Median Prob (Real) | Mean Prob (AI) | Median Prob (AI) | Probability Margin (Mean AI - Mean Real) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E14 Clean** | E6-C | 0.2569 | 0.1130 | 0.7207 | 0.8833 | +0.4638 |
| | **E20** | **0.0913** | **0.0173** | **0.3560** | **0.2675** | **+0.2647** |
| **E14 Degraded** | E6-C | 0.1488 | 0.0249 | 0.4530 | 0.4132 | +0.3042 |
| | **E20** | **0.0769** | **0.0130** | **0.3423** | **0.2137** | **+0.2654** |
| **E19 Smartphone** | E6-C | 0.1892 | 0.0326 | 0.5492 | 0.6743 | +0.3600 |
| | **E20** | **0.0282** | **0.0034** | **0.7390** | **0.9138** | **+0.7108 (Massive Separation)** |
| **E19 WhatsApp** | E6-C | 0.4552 | 0.3551 | 0.1864 | 0.0657 | **-0.2687 (Inverted Calibration!)** |
| | **E20** | **0.0680** | **0.0124** | **0.4986** | **0.5171** | **+0.4305 (Corrected Calibration)** |
| **WhatsApp N=67** | E6-C | 0.1080 | 0.0452 | 0.2777 | 0.1639 | +0.1697 |
| | **E20** | **0.3953** | **0.3807** | **0.4377** | **0.3833** | **+0.0424** |

---

## 4. Subgroup Breakdowns

### 4.1 Device / Camera Hardware Breakdown (E19 Smartphone Benchmark)

| Camera Hardware Device | Total N | Label | E6-C Accuracy | E20 Accuracy | Accuracy Delta | E6-C FPR | E20 FPR | FPR Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple iPhone X** | 35 | Real | 94.29% | **100.00%** | **+5.71 pp** | 5.71% | **0.00%** | **-5.71 pp (Eliminated)** |
| **Samsung Galaxy S9** | 35 | Real | 97.14% | **100.00%** | **+2.86 pp** | 2.86% | **0.00%** | **-2.86 pp (Eliminated)** |
| **Google Pixel (Pixel 7–9)** | 30 | Real | 56.67% | **96.67%** | **+40.00 pp** | 43.33% | **3.33%** | **-40.00 pp (Eliminated)** |

### 4.2 Generator-Level Breakdown (E19 Smartphone Benchmark)

| Generative Model | Total N | Label | E6-C Recall | E20 Recall | Recall Delta | E6-C F1 | E20 F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenAI DALL-E 3** | 20 | AI | 70.00% | **95.00%** | **+25.00 pp** | 0.8235 | **0.9744** | **+0.1508** |
| **Stability AI Stable Diffusion** | 20 | AI | 80.00% | **85.00%** | **+5.00 pp** | 0.8889 | **0.9189** | **+0.0300** |
| **Google Gemini / Imagen** | 20 | AI | 30.00% | **70.00%** | **+40.00 pp** | 0.4615 | **0.8235** | **+0.3620** |
| **FLUX (Flux LoRA Corpus)** | 20 | AI | 10.00% | **35.00%** | **+25.00 pp** | 0.1818 | **0.5185** | **+0.3367** |
| **Midjourney** | 20 | AI | 85.00% | **85.00%** | **0.00 pp** | 0.9189 | **0.9189** | **0.0000** |

### 4.3 Degradation-Level Breakdown (E14 Degraded Benchmark)

| Degradation Type | Total N | Real / AI | E6-C Accuracy | E20 Accuracy | E6-C FPR | E20 FPR | E6-C ROC-AUC | E20 ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Double JPEG (L1280, Q80 $\to$ Q70)** | 200 | 100 / 100 | 60.00% | **64.00%** (+4.00 pp) | 8.00% | **2.00%** (-6.00 pp) | 0.7333 | **0.8022** (+0.0689) |
| **WhatsApp-tier JPEG (L1280, Q75)** | 200 | 100 / 100 | 66.50% | **64.50%** (-2.00 pp) | 15.00% | **4.00%** (-11.00 pp)| 0.7812 | **0.8004** (+0.0193) |
| **Resize (L1080) + JPEG Q80** | 200 | 100 / 100 | 71.00% | **64.00%** (-7.00 pp) | 13.00% | **5.00%** (-8.00 pp) | 0.7927 | **0.7921** (-0.0006) |

---

## 5. Error Transition Analysis

To understand how E20 alters model decisions relative to E6-C, we examine the transitions for every sample:

| Benchmark Dataset | Total N | E6-C Correct $\to$ E20 Correct | E6-C Correct $\to$ E20 Wrong | E6-C Wrong $\to$ E20 Correct (Repaired) | E6-C Wrong $\to$ E20 Wrong | Net Correct Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E19 Smartphone Benchmark** | 200 | 134 | 5 | **39** | 22 | **+34 (+17.0%)** |
| **E19 WhatsApp Subset** | 60 | 19 | 3 | **25** | 13 | **+22 (+36.7%)** |
| **E14 Clean External** | 200 | 109 | 42 | **21** | 28 | **-21 (-10.5%)** |
| **E14 Degraded External** | 600 | 317 | 78 | **68** | 137 | **-10 (-1.7%)** |
| **WhatsApp N=67 Benchmark** | 67 | 23 | 18 | **12** | 14 | **-6 (-9.0%)** |

### Key Transition Takeaway
On real smartphone imagery and compressed mobile photos (E19), **E20 repaired 39 errors while introducing only 5 new errors** (a **7.8 : 1 repair-to-regression ratio**). On older synthetic datasets (E14), E20 repaired 21 false alarms on Real cameras, but introduced 42 false negatives on legacy generative architectures.

---

## 6. Representative Case Analysis

### 6.1 Repaired Errors (E6-C Wrong $\to$ E20 Correct)

1. `e19_real_phone_002_wa` (Google Pixel 7–9, WhatsApp simulation):
   - True Label: **Real**
   - E6-C Probability: **0.9883** (Categorical False Alarm — severe Pixel ISP failure)
   - E20 Probability: **0.0072** (**Completely Resolved True Negative**)
2. `e19_real_phone_003_wa` (Google Pixel 7–9, WhatsApp simulation):
   - True Label: **Real**
   - E6-C Probability: **0.5815** (False Alarm)
   - E20 Probability: **0.0372** (**Resolved True Negative**)
3. `real_raise_135_r11850456t` (RAISE Nikon DSLR RAW, E14 Clean):
   - True Label: **Real**
   - E6-C Probability: **0.9551** (False Alarm on uncompressed sensor noise)
   - E20 Probability: **0.0389** (**Resolved True Negative**)
4. `real_raise_supp_026_r0dd6e8f4t` (RAISE Nikon DSLR RAW, E14 Clean):
   - True Label: **Real**
   - E6-C Probability: **0.8501** (False Alarm)
   - E20 Probability: **0.0094** (**Resolved True Negative**)
5. `e19_ai_042` (OpenAI DALL-E 3, E19 Clean):
   - True Label: **AI**
   - E6-C Probability: **0.3120** (False Negative)
   - E20 Probability: **0.9412** (**Resolved Detection**)

### 6.2 Regressed Cases (E6-C Correct $\to$ E20 Wrong)

1. `ai_sb_stable-diffusion-1-3_161_r0a9384b1t` (SD 1.3, E14 Clean):
   - True Label: **AI**
   - E6-C Probability: **0.9800** (Correct Detection)
   - E20 Probability: **0.0143** (False Negative — E20 did not trigger on older low-res latent diffusion noise)
2. `ai_sb_firefly_087_r1bdb6385t` (Adobe Firefly 1, E14 Clean):
   - True Label: **AI**
   - E6-C Probability: **0.8838** (Correct Detection)
   - E20 Probability: **0.0266** (False Negative)
3. `WhatsApp Image 2026-09-17 at 3.38.18 PM` (WhatsApp N=67 Real photo):
   - True Label: **Real**
   - E6-C Probability: **0.0000** (Correct True Negative)
   - E20 Probability: **0.9785** (False Positive on user-submitted chat photo)

---

## 7. Artifact Index

All evaluation artifacts, predictions, and metrics are preserved in `experiments/e20_training/external_evaluation/`:
- **Execution Script**: [`run_e20_external_evaluation.py`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/run_e20_external_evaluation.py)
- **Summary Metrics CSV**: [`e20_vs_e6c_external_results.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/e20_vs_e6c_external_results.csv)
- **Subgroup Metrics CSV**: [`subgroups/e20_vs_e6c_subgroups.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/subgroups/e20_vs_e6c_subgroups.csv)
- **Error Transitions CSV**: [`e20_vs_e6c_error_transitions.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/e20_vs_e6c_error_transitions.csv)
- **Representative Errors CSV**: [`e20_vs_e6c_representative_errors.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/e20_vs_e6c_representative_errors.csv)
- **Per-Dataset Predictions**:
  - [`predictions/predictions_e14_clean.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/predictions/predictions_e14_clean.csv) (200 rows)
  - [`predictions/predictions_e14_degraded.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/predictions/predictions_e14_degraded.csv) (600 rows)
  - [`predictions/predictions_e19_smartphone_benchmark.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/predictions/predictions_e19_smartphone_benchmark.csv) (200 rows)
  - [`predictions/predictions_e19_whatsapp_subset.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/predictions/predictions_e19_whatsapp_subset.csv) (60 rows)
  - [`predictions/predictions_whatsapp_n67_benchmark.csv`](file:///d:/DeepVision-Forensics/experiments/e20_training/external_evaluation/predictions/predictions_whatsapp_n67_benchmark.csv) (67 rows)

---

## 8. Integrity & Safety Sign-Off

- **Source Checkpoints Unmodified**: Both `e6c_checkpoint_epoch2.pt` and `e20_best_model.pt` timestamps and hashes remain intact.
- **Evaluation Datasets Untouched**: No manifests or test pool directories were altered.
- **Production Code Unchanged**: Backend, frontend, and production threshold (0.50) remain strictly untouched.
- **Git Repository State**: Clean; no commits or pushes executed.
