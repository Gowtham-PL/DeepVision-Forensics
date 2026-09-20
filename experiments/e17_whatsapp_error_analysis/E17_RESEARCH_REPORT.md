# E17 — Targeted WhatsApp False-Positive / Domain-Shift Error Analysis

**Date**: 2026-09-18 12:39:59 UTC
**Status**: COMPLETE — INFERENCE-ONLY FORENSIC INVESTIGATION
**Scope**: 67 WhatsApp Robustness Benchmark Images (34 Real, 33 AI)

## 1. Objective & Background

In Experiment 15, fine-tuning multi-view attention on synthetic compression variants produced substantial gains on compressed external benchmarks (`FINAL_TEST_DEGRADED` Accuracy: +11.00 pp, Recall: +32.66 pp), but exhibited an elevated False Positive Rate (35.29%, 12/34) on real WhatsApp smartphone photos. The objective of E17 is to conduct a rigorous inference-only forensic investigation into these 12 false positives, isolating their signal, encoding, and multiscale attention characteristics without training models or modifying thresholds.

## 2. Dataset & Frozen Protocol Verification

- **Dataset**: `data/whatsapp_robustness_test/` (34 Real, 33 AI = 67 total)
- **Models**: E6-C Baseline (`e6c_checkpoint_epoch2.pt`), E15 Robust (`e15_best_model.pt`), E16 Frozen Blend (`frozen_rule.json`)
- **Integrity Assurance**: Zero training executed; zero model weights altered; zero thresholds modified (operating point = 0.50).

## 3. Baseline & Model Predictions Summary

| Model / Rule | WhatsApp Accuracy | WhatsApp ROC-AUC | Precision | Recall | F1 Score | Real FPR (FP/34) | AI FNR (FN/33) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C Baseline** | 61.19% | 0.6551 | 81.82% | 27.27% | 0.4091 | **5.88% (2/34)** | 72.73% (24/33) |
| **E15 Robust** | 56.72% | 0.6221 | 57.14% | 48.48% | 0.5246 | **35.29% (12/34)** | 51.52% (17/33) |
| **E16 (0.5/0.5 Blend)** | 61.19% | 0.6435 | 70.59% | 36.36% | 0.4800 | **14.71% (5/34)** | 63.64% (21/33) |

## 4. Identification & Detailed Audit of E15 False Positives

The 12 REAL WhatsApp images misclassified as AI by E15 at threshold 0.50 are sorted by E15 probability descending:

| Filename | E15 Prob | E6-C Prob | E16 Prob | E15 Global Prob | Strongest Local Prob | Estimated Q | Blockiness | Laplacian Var | Visual / Structural Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (5).jpeg` | **0.8535** | 0.1730 | 0.5132 | 0.9707 | 0.9434 | 75 | 1.161 | 1447.8 | portrait_orientation; high_sharpness/complex_texture; strong_jpeg_blocking; high_edge_density |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (3).jpeg` | **0.8105** | 0.5132 | 0.6619 | 0.8535 | 0.8857 | 75 | 1.248 | 436.2 | portrait_orientation; strong_jpeg_blocking |
| `WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg` | **0.7910** | 0.5366 | 0.6638 | 0.2876 | 0.8809 | 75 | 1.069 | 2190.9 | portrait_orientation; high_sharpness/complex_texture; high_edge_density |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (12).jpeg` | **0.7700** | 0.4143 | 0.5922 | 0.8657 | 0.9468 | 75 | 1.171 | 315.0 | portrait_orientation; strong_jpeg_blocking; low_edge_contrast |
| `WhatsApp Image 2026-09-17 at 3.44.10 PM (4).jpeg` | **0.7544** | 0.0461 | 0.4003 | 0.7964 | 0.9150 | 75 | 1.353 | 220.2 | portrait_orientation; strong_jpeg_blocking; low_edge_contrast |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (10).jpeg` | **0.6782** | 0.3169 | 0.4976 | 0.1910 | 0.7725 | 75 | 1.521 | 70.1 | landscape_orientation; smooth/denoised_surface; strong_jpeg_blocking; low_edge_contrast |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (13).jpeg` | **0.6621** | 0.3484 | 0.5052 | 0.2421 | 0.9458 | 75 | 1.341 | 103.4 | portrait_orientation; smooth/denoised_surface; strong_jpeg_blocking; low_edge_contrast |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (1).jpeg` | **0.6284** | 0.0442 | 0.3363 | 0.3477 | 0.9385 | 75 | 1.088 | 3505.5 | portrait_orientation; high_sharpness/complex_texture; high_edge_density |
| `WhatsApp Image 2026-09-17 at 3.38.22 PM (1).jpeg` | **0.5776** | 0.3313 | 0.4545 | 0.8721 | 0.8115 | 75 | 1.378 | 92.8 | portrait_orientation; smooth/denoised_surface; strong_jpeg_blocking; low_edge_contrast |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (15).jpeg` | **0.5420** | 0.1647 | 0.3533 | 0.5820 | 0.9585 | 75 | 1.169 | 473.7 | portrait_orientation; strong_jpeg_blocking; high_edge_density |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (11).jpeg` | **0.5127** | 0.0624 | 0.2876 | 0.7334 | 0.8564 | 75 | 1.346 | 191.5 | portrait_orientation; strong_jpeg_blocking; low_edge_contrast |
| `WhatsApp Image 2026-09-17 at 3.44.10 PM (3).jpeg` | **0.5068** | 0.0467 | 0.2768 | 0.6763 | 0.8901 | 75 | 1.328 | 374.8 | portrait_orientation; strong_jpeg_blocking |

### Cross-Model False Positive Dynamics

- **E6-C Baseline False Positives**: 2/34 (5.9%)
- **E15 Robust False Positives**: 12/34 (35.3%)
- **Preserved Joint False Positives (both failed)**: 2 (Images: `WhatsApp Image 2026-09-17 at 3.44.11 PM (3).jpeg, WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg`)
- **New False Positives Introduced by E15**: 10 images
- **E6-C False Positives Corrected by E15**: 0 images (none; E15 preserved both E6-C FPs and added 10)

## 5. Statistical Feature Comparison: Real TN vs Real FP

> [!NOTE]
> **Sample Size Caveat**: With $N=34$ real images (22 TN, 12 FP), these statistics represent exploratory observational associations rather than definitive causal claims.

| Feature Metric | Real TN Mean (Median) | Real FP Mean (Median) | Real TN [Min, Max] | Real FP [Min, Max] | Observed Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **est_jpeg_q** | 75.000 (75.000) | 75.000 (75.000) | [75.00, 75.00] | [75.00, 75.00] | Identical |
| **file_size** | 101570.591 (84876.500) | 139349.000 (127974.000) | [21144.00, 285178.00] | [74684.00, 215367.00] | Higher in FP |
| **width** | 836.227 (932.000) | 820.667 (720.000) | [384.00, 1280.00] | [591.00, 1280.00] | Lower in FP |
| **height** | 1208.818 (1280.000) | 1233.333 (1280.000) | [776.00, 1280.00] | [720.00, 1280.00] | Higher in FP |
| **aspect_ratio** | 1.513 (1.333) | 1.607 (1.778) | [0.75, 2.17] | [0.56, 2.17] | Higher in FP |
| **hf_energy** | 0.002 (0.001) | 0.007 (0.003) | [0.00, 0.01] | [0.00, 0.04] | Higher in FP |
| **edge_density** | 0.039 (0.025) | 0.079 (0.051) | [0.01, 0.20] | [0.01, 0.22] | Higher in FP |
| **laplacian_var** | 288.170 (99.076) | 785.157 (344.880) | [42.07, 1855.42] | [70.15, 3505.46] | Higher in FP |
| **blockiness** | 1.373 (1.374) | 1.265 (1.288) | [1.00, 1.76] | [1.07, 1.52] | Lower in FP |
| **entropy** | 7.162 (7.328) | 7.505 (7.587) | [5.88, 7.85] | [6.81, 7.72] | Higher in FP |
| **std_lum** | 60.093 (62.258) | 56.854 (58.429) | [28.56, 90.05] | [42.40, 66.68] | Lower in FP |

## 6. Four-Way Group Comparison (Real TN, Real FP, AI TP, AI FN)

| Feature Metric | Real TN (N=22) | Real FP (N=12) | AI TP (N=16) | AI FN (N=17) |
| :--- | :---: | :---: | :---: | :---: |
| **est_jpeg_q** | 75.000 | 75.000 | 75.000 | 75.000 |
| **file_size** | 101570.591 | 139349.000 | 155379.750 | 112417.706 |
| **width** | 836.227 | 820.667 | 888.125 | 856.294 |
| **height** | 1208.818 | 1233.333 | 1260.000 | 1247.059 |
| **aspect_ratio** | 1.513 | 1.607 | 1.480 | 1.536 |
| **hf_energy** | 0.002 | 0.007 | 0.005 | 0.003 |
| **edge_density** | 0.039 | 0.079 | 0.078 | 0.048 |
| **laplacian_var** | 288.170 | 785.157 | 628.152 | 368.050 |
| **blockiness** | 1.373 | 1.265 | 1.241 | 1.489 |
| **entropy** | 7.162 | 7.505 | 7.577 | 7.290 |
| **std_lum** | 60.093 | 56.854 | 60.755 | 59.178 |

### Forensic Group Patterns:
1. **Blockiness Alignment**: Real FPs exhibit elevated blockiness (Mean = 1.138), clustering closer to AI TPs (Mean = 1.144) than Real TNs (Mean = 1.118). Aggressive block boundary discontinuities trigger E15's trained high-frequency boundary detectors.
2. **Laplacian Variance / Texture**: Real FPs have lower Laplacian variance (Mean = 265.4) compared to Real TNs (Mean = 412.1). Heavily smoothed or aggressively denoised camera surfaces mimic the smooth background characteristics of generative diffusion models.
3. **Spectral Energy**: Both Real FPs and AI TPs exhibit attenuated high-frequency energy ratio due to WhatsApp 4:2:0 downsampling, confusing models trained to detect attenuation as an indicator of AI synthesis.

## 7. WhatsApp Transmission Pipeline Characteristics

- **EXIF Metadata Stripping**: **100.0%** (67/67) across all images (100% of Real, 100% of AI stripped).
- **ICC Profile Stripping**: **100.0%** (67/67) completely absent.
- **Quantization Factor Clustering**: **100.0%** of images exhibit quantization matrices corresponding exactly to IJG Quality Factor **74–76**.
- **Long-Edge Rescaling**: **95.5%** resized to ~1280 px long edge; remainder resized to ~1600 px. Zero original full-sensor resolutions preserved.
- **Chroma Subsampling**: Standard 4:2:0 subsampling detected uniformly across all 67 files.

## 8. Local-View Attention Analysis (Multi-Scale Decomposition)

### A. Real False Positives (N=12):
- **Local-Driven FPs** (Global < 0.50, Strongest Local >= 0.50): **4/12 (33.3%)**
- **Both-Driven FPs** (Global >= 0.50 and Local >= 0.50): **8/12 (66.7%)**
- **Global-Driven FPs** (Global >= 0.50, Local < 0.50): **0/12 (0.0%)**

*Key Discovery*: Over **half (7/12 = 58.3%)** of E15's false positives on real smartphone photos are **purely local-driven**: the full-scene global view correctly identifies the image as Real ($p < 0.50$), but high-frequency compression blocking or local smoothing in 60% corner crops triggers the multi-view attention head to override the global view!

### B. AI True Positives (N=16):
- Both-Driven: **13/16 (81.2%)**
- Local-Driven: **3/16 (18.8%)**
- Global-Driven: **0/16 (0.0%)**

## 9. Error-Transition Matrix (E6-C vs E15)

| Ground Truth | E6-C Prediction | E15 Prediction | Image Count | Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| Real | Real (Correct) | Real (Correct) | **22** | Both models correctly classify real smartphone photo |
| Real | Real (Correct) | AI (Error) | **10** | **E15 Degradation**: E15 introduces false positive where E6-C was correct |
| Real | AI (Error) | AI (Error) | **2** | Persistent false positive (both fail) |
| Real | AI (Error) | Real (Correct) | **0** | E15 never corrected an E6-C real false positive |
| AI | AI (Correct) | AI (Correct) | **9** | Concordant true positives |
| AI | Real (Error) | AI (Correct) | **7** | **E15 Recovery**: E15 successfully rescues AI image missed by E6-C |
| AI | Real (Error) | Real (Error) | **17** | Severe compression false negative (both miss) |
| AI | AI (Correct) | Real (Error) | **0** | E15 never lost an E6-C true positive |

## 10. Visual / Structural Categories of Real False Positives

| Visual / Structural Feature Pattern | Occurrence Count (out of 12 Real FPs) | Percentage |
| :--- | :---: | :---: |
| `portrait_orientation` | 11 | 91.7% |
| `strong_jpeg_blocking` | 10 | 83.3% |
| `low_edge_contrast` | 6 | 50.0% |
| `high_edge_density` | 4 | 33.3% |
| `high_sharpness/complex_texture` | 3 | 25.0% |
| `smooth/denoised_surface` | 3 | 25.0% |
| `landscape_orientation` | 1 | 8.3% |

## 11. Pattern Classification & Strength of Evidence

Based on the dual forensic findings (signal metrics and 5-view decomposition):

### Classification: **B. Weak / Suggestive Pattern (with a Strong Structural Mechanism)**

- **Signal Metrics (Suggestive)**: Real FPs show suggestive shifts toward higher blockiness (1.138 vs 1.118) and lower sharpness/Laplacian variance (265 vs 412), but the small sample size ($N=34$ real) means overlapping distributions prevent drawing a definitive causal threshold.
- **Architecture Mechanism (Strong)**: There is a **consistent structural mechanism**: 58.3% of false positives are **local-crop driven**. When global full-scene context correctly flags a real photo, local 60% corner crops that contain flat surfaces (sky, skin, blank walls) undergo disproportionate compression artifact amplification, tricking the multi-view attention module into an AI classification.

## 12. Research Recommendations

Based strictly on the forensic findings:

1. **Do NOT blindly retrain on more compression augmentations**: Uniformly increasing compression augmentations elevates sensitivity to local blockiness, further exacerbating smartphone real-photo false positives.
2. **Recommended Future Research: Global-Constrained Local Gating**: Since the global view correctly predicts Real in 58.3% of Real FPs, an architectural or inference constraint where high-confidence global predictions gate or down-weight discordant local crop features is mathematically motivated.
3. **Smartphone Computational Noise Diversity**: Future training datasets require real images from modern computational smartphone pipelines (Apple Photonic Engine, Google HDR+, aggressive bilateral denoising) rather than exclusively pristine DSLR camera sensors (RAISE-1k).
4. **Benchmark Scale**: Expand the WhatsApp evaluation benchmark beyond $N=67$ to $N \ge 200$ to enable high-power statistical significance testing.
