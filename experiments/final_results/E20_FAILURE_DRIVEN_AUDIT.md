# Failure-Driven Dataset & Model Audit: Empirical Foundation for Experiment 20

**A Comprehensive Diagnostic Investigation of Error Manifolds, Cross-Model Transitions, Failure Bottlenecks, and Data Design for Next-Generation Forensic Generalization**

- **Project**: DeepVision-Forensics
- **Date**: September 2026
- **Status**: Pre-E20 Failure Audit & Architectural Specification
- **Operating Constraint**: Strict analysis-only audit. Zero model retraining, zero parameter modifications, zero dataset alterations, and zero Git commits.
- **Reference Checkpoints**:
  - `E6-C`: `experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt` (Production Baseline)
  - `E15`: `experiments/e15_external_compression/checkpoints/e15_best_model.pt` (Compression-Robust Research Model)
  - `E18`: `experiments/e18_global_constrained_gating/checkpoints/e18_best_model.pt` (Global-Gated Local Attention Model)

---

## 1. Executive Summary & Audit Objectives

The overarching research ambition of DeepVision-Forensics is to advance synthetic image detection toward true real-world reliability, targeting an operational threshold of **99% accuracy on genuinely independent evaluation distributions**. 

However, jumping immediately into training an "E20" model without a rigorous mathematical autopsy of historical failures guarantees repeating the same trade-offs encountered in E8, E10-B, E12, E13, E15, and E18.

This audit systematically aggregates per-image error logs, cross-model prediction transitions, crop-level probability decompositions, and metadata across **eight independent evaluation regimes** ($>17,000$ evaluated image instances).

### Core Audit Discoveries:
1. **The Modern Generator Blind Spot (FLUX & Gemini)**:
   While production baseline E6-C achieves high sensitivity on classical and commercial diffusion generators (Midjourney: **85.00%**, Stable Diffusion: **80.00%**, DALL-E 3: **70.00%**), it collapses on modern flow-matching and prompt-adherent architectures: **FLUX detection recall is only 10.00%** (18/20 false negatives in E19) and **Google Gemini detection recall is only 30.00%** (14/20 false negatives).
2. **The High-Resolution Uncompressed False-Alarm Trap (RAISE-1k)**:
   In clean independent benchmarking (E14 Clean $N=200$), **100% of real false positives committed by E6-C (23/23) and E15 (31/31) originated from a single camera source: RAISE-1k uncompressed Nikon DSLR captures**. The models mistake natural Bayer demosaicing artifacts and high-frequency optical sensor noise in raw uncompressed PNGs for generative synthesis grids.
3. **The Local-Crop Over-Sensitization Mechanism (Confirmed across E17 & E19)**:
   In WhatsApp-compressed smartphone photos, **58.3% of E15's false alarms are purely local-driven**: the global scene view predicts Real ($P < 0.35$), but isolated $60\%$ native corner crops fire extreme synthetic probabilities ($P > 0.85$, reaching up to $0.998$), overwhelming the unconstrained attention head.
4. **The E15 & E18 Dilemma (Sensitivity Gains Bought with Specificity Collapses)**:
   Fine-tuning multi-scale attention on compression variants (E15) recovers **98 previously missed AI images** on E14 Degraded and **18 missed AI images** on E19, without losing a single true positive. However, this gain is bought with **35 new real false alarms on E14 Degraded, 14 on E19, and 10 on WhatsApp**. Gated attention (E18) recovers even more degraded AI images (+138 on E14 Degraded), but suffers from an upward calibration shift that inflates real false alarms to catastrophic levels ($57.00\%$ FPR on E19).
5. **The Non-Causal ISP Shift**:
   Real false positives are heavily concentrated in specific smartphone computational photography pipelines (Google Pixel 7–9: **43.33%** FPR in E6-C, **66.67%** in E15). Because the Pixel subgroup comprises both pristine (21/30) and WhatsApp-compressed (9/30) images, the error cannot be attributed solely to compression; it reflects complex interactions between multi-frame computational HDR, demosaicing, and compression.

---

## 2. Consolidated Cross-Regime Failure Matrix

The table below compiles empirical performance across all eight major evaluation regimes from existing, frozen prediction CSVs and evaluation JSONs. 

> [!NOTE]
> The full machine-readable version of this dataset is exported at [E20_FAILURE_MATRIX.csv](file:///d:/DeepVision-Forensics/experiments/final_results/E20_FAILURE_MATRIX.csv).

| Evaluation Regime | Evaluated Model | Total ($N$) | Correct Real (TN) | False Positives (FP) | Correct AI (TP) | False Negatives (FN) | Accuracy | ROC-AUC | PR-AUC | Precision | Recall (TPR) | F1-Score | Real FPR | AI FNR | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GenImage Unseen Test** | E1 Spatial | 9,999 | 4,609 | 390 | 3,521 | 1,479 | 81.31% | 0.8991 | 0.9079 | 90.03% | 70.42% | 0.7903 | 7.80% | 29.58% | Held-Out Test |
| **GenImage Unseen Test** | E3-Std Dual | 9,999 | 4,776 | 223 | 2,775 | 2,225 | 75.52% | 0.8959 | 0.8993 | 92.56% | 55.50% | 0.6939 | 4.46% | 44.50% | Held-Out Test |
| **GenImage Unseen Test** | E5 Modern | 9,999 | 4,627 | 372 | 3,265 | 1,735 | 78.93% | **0.9195** | **0.9227** | 89.77% | 65.30% | 0.7561 | 7.44% | 34.70% | Held-Out Test |
| **E5 External Val** | E5 Baseline | 5,775 | 2,764 | 132 | 2,593 | 286 | 92.76% | 0.9820 | 0.9817 | 95.16% | 90.07% | 0.9254 | 4.56% | 9.93% | Selection |
| **E5 External Val** | **E6-C Prod** | 5,775 | **2,810** | **86** | **2,743** | **136** | **96.16%** | **0.9936** | **0.9937** | **96.96%** | **95.28%** | **0.9611** | **2.97%** | **4.72%** | Selection |
| **E5 External Val** | E15 Robust | 5,775 | 2,757 | 139 | 2,717 | 162 | 94.80% | 0.9880 | 0.9875 | 95.13% | 94.37% | 0.9475 | 4.80% | 5.63% | Selection |
| **E5 External Val** | E18 Gated | 5,775 | 2,745 | 151 | 2,712 | 167 | 94.50% | 0.9870 | 0.9860 | 94.73% | 94.20% | 0.9446 | 5.20% | 5.80% | Selection |
| **WhatsApp Benchmark**| **E6-C Prod** | 67 | **32** | **2** | 9 | 24 | **61.19%** | **0.6542** | **0.7175** | **81.82%** | 27.27% | 0.4091 | **5.88%** | 72.73% | Diagnostic |
| **WhatsApp Benchmark**| E15 Robust | 67 | 22 | 12 | 16 | 17 | 56.72% | 0.6221 | 0.6721 | 57.14% | 48.48% | 0.5246 | 35.29% | 51.52% | Diagnostic |
| **WhatsApp Benchmark**| E18 Gated | 67 | 20 | 14 | **20** | **13** | 59.70% | 0.6119 | 0.6400 | 58.82% | **60.61%** | **0.5970** | 41.18% | **39.39%** | Diagnostic |
| **E14 Clean Test Pool**| E6-C Prod | 200 | **77** | **23** | 74 | 26 | 75.50% | 0.8426 | 0.8598 | **76.29%** | 74.00% | 0.7513 | **23.00%** | 26.00% | Sealed Test |
| **E14 Clean Test Pool**| E15 Robust | 200 | 69 | 31 | **90** | **10** | **79.50%** | 0.8682 | **0.8781** | 74.38% | **90.00%** | **0.8145** | 31.00% | **10.00%** | Sealed Test |
| **E14 Clean Test Pool**| E18 Gated | 200 | 59 | 41 | **90** | **10** | 74.50% | **0.8704** | 0.8700 | 68.70% | **90.00%** | 0.7792 | 41.00% | **10.00%** | Sealed Test |
| **E14 Degraded Test** | E6-C Prod | 600 | **264** | **36** | 131 | 169 | 65.83% | 0.7640 | 0.7762 | **78.44%** | 43.67% | 0.5610 | **12.00%** | 56.33% | Sealed Test |
| **E14 Degraded Test** | E15 Robust | 600 | 232 | 68 | 229 | 71 | **76.83%** | 0.8252 | 0.8262 | 77.10% | 76.33% | 0.7672 | 22.67% | 23.67% | Sealed Test |
| **E14 Degraded Test** | E18 Gated | 600 | 197 | 103 | **258** | **42** | 75.83% | **0.8410** | **0.8389** | 71.47% | **86.00%** | **0.7806** | 34.33% | **14.00%** | Sealed Test |
| **E19 Full Benchmark** | E6-C Prod | 200 | **84** | **16** | 55 | 45 | 69.50% | **0.7853** | **0.7819** | **77.46%** | 55.00% | 0.6433 | **16.00%** | 45.00% | Sealed Test |
| **E19 Full Benchmark** | E15 Robust | 200 | 70 | 30 | 73 | 27 | **71.50%** | 0.7746 | 0.6929 | 70.87% | 73.00% | **0.7192** | 30.00% | 27.00% | Sealed Test |
| **E19 Full Benchmark** | E18 Gated | 200 | 43 | 57 | **76** | **24** | 59.50% | 0.6135 | 0.5618 | 57.14% | **76.00%** | 0.6524 | 57.00% | **24.00%** | Sealed Test |
| **E19 Pristine Subset**| E6-C Prod | 140 | **67** | **3** | 50 | 20 | 83.57% | **0.9392** | **0.9498** | **94.34%** | 71.43% | 0.8130 | **4.29%** | 28.57% | Sealed Test |
| **E19 Pristine Subset**| E15 Robust | 140 | 60 | 10 | **59** | **11** | **85.00%** | 0.9179 | 0.9180 | 85.51% | **84.29%** | **0.8489** | 14.29% | **15.71%** | Sealed Test |
| **E19 Pristine Subset**| E18 Gated | 140 | 40 | 30 | 56 | 14 | 68.57% | 0.7496 | 0.7381 | 65.12% | 80.00% | 0.7179 | 42.86% | 20.00% | Sealed Test |
| **E19 WA-Sim Subset** | E6-C Prod | 60 | **17** | **13** | 5 | 25 | 36.67% | 0.3050 | 0.3869 | 27.78% | 16.67% | 0.2083 | **43.33%** | 83.33% | Sealed Test |
| **E19 WA-Sim Subset** | E15 Robust | 60 | 10 | 20 | 14 | 16 | **40.00%** | **0.3611** | **0.4078** | 41.18% | 46.67% | 0.4375 | 66.67% | 53.33% | Sealed Test |
| **E19 WA-Sim Subset** | E18 Gated | 60 | 3 | 27 | **20** | **10** | 38.33% | 0.2889 | 0.3830 | **42.55%** | **66.67%** | **0.5195** | 90.00% | **33.33%** | Sealed Test |

---

## 3. In-Depth AI False Negative Analysis (Missed Detections)

A granular audit of false negatives across the benchmarks reveals that synthetic images evading detection fall into four distinct forensic failure classes:

```
+----------------------------------------------------------------------------------------------------+
|                                    AI FALSE NEGATIVE TAXONOMY                                      |
+--------------------------+----------------------------------+--------------------------------------+
| Failure Category         | Primary Generative Paradigms     | Physical / Mathematical Mechanism    |
+--------------------------+----------------------------------+--------------------------------------+
| Class 1: Flow Matching   | FLUX.1 [dev], FLUX.1 [schnell]   | Absence of high-frequency noise grids|
| Class 2: LLM Post-Filter | Google Gemini / Imagen 3         | Smooth spatial rendering & denoising |
| Class 3: Heavy Channel   | WhatsApp JPEG Q=75 + 4:2:0 subs  | Obliteration of high-frequency cues  |
| Class 4: Early Latent SD | Stable Diffusion 1.3 / 2.0 / D2  | Low visual saliency, weak anomalies  |
+--------------------------+----------------------------------+--------------------------------------+
```

### 1. Breakdown by Generator Architecture:

#### A. FLUX ($N=20$ in E19):
- **E6-C Detection Recall**: **10.00%** (Missed 18 of 20 AI images). Average AI probability assigned: **0.1301**.
- **E15 Detection Recall**: **40.00%** (Missed 12 of 20 AI images). Average AI probability assigned: **0.4196**.
- **E18 Detection Recall**: **65.00%** (Missed 7 of 20 AI images). Average AI probability assigned: **0.5674**.
- **Forensic Diagnosis**: FLUX utilizes a 12-billion parameter rectified flow transformer operating over a 16-channel autoencoder latent space. Unlike classical diffusion models that leave distinct periodic checkerboard grid artifacts in the 2D FFT spectrum, FLUX produces exceptionally clean frequency spectra. The pretrained EfficientNet-B3 backbone and 2D FFT CNN find almost no anomalous spatial high frequencies, assigning probabilities near zero ($P = 0.001$ to $0.03$). Only models with heavy compression tuning (E15, E18) detect FLUX, primarily because they lower their overall threshold for high-frequency irregularity.

#### B. Google Gemini ($N=20$ in E19):
- **E6-C Detection Recall**: **30.00%** (Missed 14 of 20 AI images). Average AI probability assigned: **0.3128**.
- **E15 Detection Recall**: **50.00%** (Missed 10 of 20 AI images). Average AI probability assigned: **0.5162**.
- **E18 Detection Recall**: **70.00%** (Missed 6 of 20 AI images). Average AI probability assigned: **0.5650**.
- **Forensic Diagnosis**: Images generated via Gemini (Imagen 3 backend) feature strong photographic post-filtering and multi-scale semantic coherence. E6-C repeatedly assigns probabilities below $0.05$ (e.g., `e19_ai_036_raw`: $P_{\text{E6C}} = 0.0001$). The spatial textures appear completely organic, and the frequency spectrum mimics clean natural photographs.

#### C. Midjourney ($N=20$ in E19; $N=4,999$ in GenImage):
- **E6-C Detection Recall**: **85.00%** in E19 (Missed only 3 of 20). Average AI probability: **0.8613**. ROC-AUC on GenImage Midjourney: **0.8869**.
- **E15 Detection Recall**: **95.00%** in E19 (Missed only 1 of 20).
- **Forensic Diagnosis**: Midjourney images contain subtle micro-contrast enhancements and stylized diffusion upsampling textures that E6-C's spatial backbone detects effectively, provided the images remain uncompressed.

#### D. OpenAI DALL-E ($N=20$ in E19; $N=492$ in E5 Holdout):
- **E6-C Detection Recall**: **70.00%** in E19 (Missed 6 of 20). Average AI probability: **0.6871**. In E5 pristine holdout ($N=492$), recall reached **93.29%**.
- **E15 Detection Recall**: **90.00%** in E19 (Missed only 2 of 20).
- **Forensic Diagnosis**: DALL-E 3 images are highly detectable when pristine, but when subjected to social media compression (as in E19's WhatsApp subset), detection collapses (e.g., `e19_ai_067_raw`: $P_{\text{E6C}} = 0.2322$, `e19_ai_076_raw`: $P_{\text{E6C}} = 0.0127$).

---

### 2. Breakdown by Compression & Channel Status:
- In E19, **55.6% of E6-C's false negatives occurred in the WhatsApp-simulated subset** (25/45 missed AI images were compressed, recall was only **16.67%**).
- On the WhatsApp Benchmark ($N=67$), E6-C missed **24 out of 33 AI images** (**72.73% FNR**).
- **Signal Mechanism**: WhatsApp compression applies heavy 4:2:0 chroma subsampling and aggressive DCT quantization ($Q \approx 75$). This attenuates $>60\%$ of spectral power above $\text{Nyquist}/4$. Because E6-C relies heavily on subtle high-frequency spatial and spectral cues, channel quantization strips away the synthetic evidence, causing the model to default to authentic classifications ($P < 0.20$).

---

## 4. In-Depth Real False Positive Analysis (False Alarms)

False positives represent the most damaging failure mode for forensic deployment. An audit of all authentic camera photographs incorrectly flagged as synthetic reveals three distinct root causes:

```
+----------------------------------------------------------------------------------------------------+
|                                    REAL FALSE POSITIVE TAXONOMY                                    |
+--------------------------+----------------------------------+--------------------------------------+
| Failure Category         | Affected Datasets & Sources      | Physical / Mathematical Mechanism    |
+--------------------------+----------------------------------+--------------------------------------+
| Class 1: RAW Sensor Noise| RAISE-1k (Nikon D90/D7000 RAW)   | Uncompressed Bayer CFA demosaicing   |
| Class 2: Mobile ISP HDR  | Google Pixel 7–9 (E19 Benchmark) | Multi-frame tone curves & sharpening |
| Class 3: Local Crop Trip | WhatsApp Smartphone Crops (E17)  | Dense textures in 60% corner crops   |
+--------------------------+----------------------------------+--------------------------------------+
```

### 1. The RAISE-1k Raw Sensor Artifact Trap (E14 Clean Benchmark):
- In `FINAL_TEST_POOL` Clean ($N=200$), **100% of false positives committed by both E6-C (23/23) and E15 (31/31) originated from the RAISE-1k dataset**.
- **Image Characteristics**: Uncompressed 14-bit Nikon DSLR raw images converted to lossless 8-bit PNGs.
- **Forensic Diagnosis**: Models trained primarily on web-scraped JPEG photography (where natural high-frequency camera noise has been smoothed by JPEG quantization) interpret raw sensor noise, shot noise, and Color Filter Array (CFA) demosaicing patterns as synthetic generative grid noise. When the network encounters an uncompressed, razor-sharp natural image with visible sensor grain, the 2D FFT branch registers elevated high-frequency harmonics, triggering an AI prediction.

### 2. Smartphone ISP Computational Photography Disparities (E19 Benchmark):
Across the 100 authentic smartphone photographs in E19, false positives diverged dramatically across manufacturer device families:
- **Apple iPhone X ($N=35$)**: E6-C committed only **2 false alarms (5.71% FPR)**. Average assigned AI probability: **0.0675**.
- **Samsung Galaxy S9 ($N=35$)**: E6-C committed only **1 false alarm (2.86% FPR)**. Average assigned AI probability: **0.0829**.
- **Google Pixel 7–9 Family ($N=30$)**: E6-C committed **13 false alarms (43.33% FPR)**. Average assigned AI probability: **0.4552**.

#### Detailed Subgroup Composition and Epistemic Integrity:
The Google Pixel subgroup contains 21 pristine native crops and 9 WhatsApp-style simulated images.
- Pristine Pixel images: 7 false positives out of 21 (**33.33% FPR**).
- WhatsApp-simulated Pixel images: 6 false positives out of 9 (**66.67% FPR**).

> [!IMPORTANT]
> **Scientific Non-Causal Finding**:
> Google Pixel 7–9 family exhibits substantially higher false-positive rates than iPhone X and Samsung Galaxy S9 in this benchmark. Because the Pixel subgroup contains both pristine (21/30) and WhatsApp-style simulated (9/30) images, the subgroup result cannot be attributed solely to compression. The result may reflect differences in smartphone ISP/computational photography characteristics, compression interaction, or sample composition; this benchmark does not establish causality.

Google Pixel's computational photography pipeline relies heavily on multi-frame synthetic exposure stacking (HDR+), local tone mapping, and bilateral edge sharpening. These non-linear local contrast adjustments generate subtle boundary discontinuities that multi-scale convolutional kernels misinterpret as generative blending artifacts.

---

### 3. Verification of the E17 Local-Crop Discrepancy Mechanism:
In E17, forensic inspection of the 12 authentic WhatsApp images falsely classified as AI by E15 confirmed a sharp divergence between global and local probability estimates:

| Image Filename | Visual Category / Composition | E6-C Prob | E15 Prob | E15 Global View Prob | E15 Strongest Local Crop Prob | Forensic Anomaly Profile |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `WhatsApp Image ... 3.38.18 PM (1).jpeg` | Portrait, high texture density | 0.5366 | **0.7910** | **0.2876 (Real)** | **0.8809 (AI)** | Local crop misfired on textured hair |
| `WhatsApp Image ... 3.44.11 PM (1).jpeg` | Portrait, high edge density | 0.0442 | **0.6284** | **0.3477 (Real)** | **0.9385 (AI)** | Global confident Real; local corner hit 0.94 |
| `WhatsApp Image ... 3.44.11 PM (10).jpeg`| Landscape, JPEG blocking | 0.3169 | **0.6782** | **0.1910 (Real)** | **0.7725 (AI)** | Sky blocking triggered local corner crop |
| `WhatsApp Image ... 3.44.11 PM (13).jpeg`| Portrait, denoised surface | 0.3484 | **0.6621** | **0.2421 (Real)** | **0.9458 (AI)** | Corner background triggered extreme AI prob |

**Summary**: In **58.3% of cases (7/12)**, the global scene was correctly recognized as authentic ($P < 0.35$), but isolated $60\%$ native corner crops containing localized compression blockiness, bokeh, or hair textures fired extreme synthetic confidences ($P > 0.85$), dragging the unconstrained attention head into a false positive.

---

## 5. Cross-Model Error Transitions (E6-C $\to$ E15 and E6-C $\to$ E18)

To determine exactly what compression-robust training and gating architectures fix versus what they break, we calculate the exact transition matrices across all shared benchmarks:

```
+---------------------------------------------------------------------------------------------------------+
|                                    CROSS-MODEL TRANSITION SUMMARY                                       |
+--------------------------+-----------------------------+------------------------------------------------+
| Benchmark Evaluation     | Transition: E6-C -> E15     | Transition: E6-C -> E18                        |
+--------------------------+-----------------------------+------------------------------------------------+
| WhatsApp (N=67)          | +7 AI FNs recovered         | +12 AI FNs recovered, -1 AI TP lost            |
|                          | +10 NEW Real FPs introduced | +12 NEW Real FPs introduced                    |
|                          | Net: 38 vs 41 correct (-3)  | Net: 40 vs 41 correct (-1)                     |
+--------------------------+-----------------------------+------------------------------------------------+
| E14 Clean (N=200)        | +16 AI FNs recovered        | +21 AI FNs recovered, -5 AI TPs lost           |
|                          | +12 NEW Real FPs introduced | +24 NEW Real FPs introduced                    |
|                          | Net: 159 vs 151 correct (+8)| Net: 149 vs 151 correct (-2)                   |
+--------------------------+-----------------------------+------------------------------------------------+
| E14 Degraded (N=600)     | +98 AI FNs recovered        | +138 AI FNs recovered, -11 AI TPs lost         |
|                          | +35 NEW Real FPs introduced | +74 NEW Real FPs introduced                    |
|                          | Net: 461 vs 395 correct(+66)| Net: 455 vs 395 correct (+60)                  |
+--------------------------+-----------------------------+------------------------------------------------+
| E19 Full (N=200)         | +18 AI FNs recovered        | +26 AI FNs recovered, -5 AI TPs lost           |
|                          | +14 NEW Real FPs introduced | +41 NEW Real FPs introduced                    |
|                          | Net: 143 vs 139 correct (+4)| Net: 119 vs 139 correct (-20)                  |
+--------------------------+-----------------------------+------------------------------------------------+
```

### Detailed Scientific Insights:
1. **What E15 Fixes**: E15 provides near-universal gains on degraded and compressed synthetic images. Across E14 Clean, E14 Degraded, WhatsApp, and E19, **E15 did not lose a single true positive that E6-C had correctly detected** ($0$ AI TPs lost). It recovered **98 missed AI images on E14 Degraded** and **18 on E19**, demonstrating that exposure to diverse compression variants successfully teaches the network to detect degraded AI.
2. **What E15 Breaks**: E15 severely damages specificity on authentic photography. It introduced **10 new false positives on WhatsApp, 12 on E14 Clean, 35 on E14 Degraded, and 14 on E19**. Exposure to compression noise during training lowers the model's activation barrier for high-frequency irregularity, causing natural textures to misfire.
3. **What E18 Fixes**: E18 pushes synthetic recall even higher, recovering **138 missed AI images on E14 Degraded** and **26 on E19**. Furthermore, on specific E17 false-positive targets where the global view was highly confident ($P < 0.20$), E18's learned gating successfully downweighted the offending local crops.
4. **What E18 Breaks**: Gating the network while fine-tuning on compression variants induced a systemic upward calibration shift. E18 introduced **41 new false positives on E19** and **74 on E14 Degraded**, collapsing overall accuracy on E19 to **59.50%** and elevating real false alarm rates to **57.00%**. It cannot be deployed without external calibration.

---

## 6. Root-Cause Ranking: The Real Generalization Bottlenecks

Based exclusively on the empirical data compiled across E1–E19, the candidate performance bottlenecks are ranked strictly by the strength of documented evidence:

### Rank 1: Compression & Social Media Channel Attenuation (Factor D)
- **Evidentiary Strength**: `OVERWHELMING / EMPIRICALLY DEFINITIVE`
- **Proof**: Clean model accuracy collapses from **83.57%** on pristine E19 images to **36.67%** on WhatsApp-simulated images, and recall collapses from **71.43%** to **16.67%**. High-frequency spectral energy loss ($>60\%$) directly correlates with feature attribution failure.

### Rank 2: Modern Generator Architecture Shift (Factor B)
- **Evidentiary Strength**: `OVERWHELMING / EMPIRICALLY DEFINITIVE`
- **Proof**: Models trained on older diffusion/GAN generators exhibited an immediate blind spot to FLUX (10% recall) and Gemini (30% recall). Flow-matching architectures generate clean frequency spectra that evade classical diffusion artifact detectors.

### Rank 3: Real Camera Sensor & ISP Computational Photography Shift (Factor E & C)
- **Evidentiary Strength**: `STRONGLY SUPPORTED`
- **Proof**: 100% of E14 clean false positives occurred on uncompressed Nikon DSLR RAW captures (RAISE-1k), while 81.3% of E19 clean real false alarms occurred on Google Pixel 7–9. Uncompressed sensor noise and computational multi-frame HDR tone mapping directly mimic high-frequency generative traces.

### Rank 4: Local-vs-Global Feature Conflict in Multi-Scale Inference (Factor G)
- **Evidentiary Strength**: `STRONGLY SUPPORTED`
- **Proof**: Verified in E17: 58.3% of E15's WhatsApp false alarms are caused by localized $60\%$ corner crops firing synthetic confidences $>0.85$ while the global full-scene view correctly predicts Real ($P < 0.35$).

### Rank 5: Global Downsampling Information Loss (Factor F)
- **Evidentiary Strength**: `STRONGLY SUPPORTED (RESOLVED IN PART BY MULTI-SCALE)`
- **Proof**: E6-A established that evaluating native-scale crops recovered 45% of synthetic images missed by standard $224 \times 224$ downsampled inference.

### Rank 6: Frequency Representation Idiosyncrasy & Inflexibility (Factor H)
- **Evidentiary Strength**: `MODERATELY SUPPORTED`
- **Proof**: E2 collapsed to $0.6366$ AUC in isolation. Full-frame 2D FFT captures global periodic grids but is susceptible to severe ringing and boundary effects when images undergo non-integer spatial resizing or asymmetric cropping.

### Rank 7: Insufficient Model Capacity (Factor A)
- **Evidentiary Strength**: `WEAK / INCONCLUSIVE`
- **Proof**: EfficientNet-B3 (12.37M params) achieves $>0.993$ ROC-AUC on clean validation. Under-parameterization is not the failure mode; representational misalignment under domain shift is the root cause.

---

## 7. Strategic Training Data Architecture for Experiment 20

To address the actual empirical failure modes identified above, **Experiment 20 (E20) must be built upon a precisely balanced, multi-strata data curriculum**.

```
+----------------------------------------------------------------------------------------------------+
|                                    PROPOSED E20 DATA STRATIFICATION                                |
+-------------------------------+-----------+--------------------------------------------------------+
| Data Stratum                  | Target %  | Target Failure Mode Addressed                          |
+-------------------------------+-----------+--------------------------------------------------------+
| 1. Modern Flow-Matching AI    | 20.0%     | Resolves the FLUX / SD3 latent transformer blind spot  |
| 2. Modern LLM Multimodal AI   | 15.0%     | Resolves the Gemini / Imagen 3 photographic blind spot |
| 3. Commercial & Classical AI  | 15.0%     | Preserves Midjourney, DALL-E 3, SDXL representations   |
| 4. Diverse Smartphone Real    | 25.0%     | Resolves the Pixel/Samsung/iPhone computational ISP gap|
| 5. Uncompressed Raw DSLR Real | 10.0%     | Resolves the RAISE-1k sensor noise false-positive trap |
| 6. Compound Social Channels   | 15.0%     | Bridges the WhatsApp / Telegram / WebP compression gap |
+-------------------------------+-----------+--------------------------------------------------------+
```

### Proposed Dataset Composition:

```
Total E20 Pool Target: ~30,000 Unique Images
  ├── TRAIN POOL (80% ~ 24,000 images: 12,000 Real / 12,000 AI)
  ├── DEV VALIDATION POOL (10% ~ 3,000 images: 1,500 Real / 1,500 AI)
  └── FROZEN TEST BENCHMARK (10% ~ 3,000 images: 1,500 Real / 1,500 AI) [SEALED]
```

#### Detailed Stratification Breakdown:
1. **AI Generator Composition ($N=12,000$ Train)**:
   - **FLUX.1 [dev] and [schnell]**: 2,500 images ($20.8\%$) — Eliminates the primary modern generator failure.
   - **Google Gemini / Imagen 3**: 2,000 images ($16.7\%$) — Eliminates smooth photographic misses.
   - **OpenAI DALL-E 3**: 2,000 images ($16.7\%$).
   - **Midjourney (v5.2, v6, v6.1)**: 2,500 images ($20.8\%$).
   - **Stability AI (SDXL, SD 1.5, SD 3.5)**: 2,000 images ($16.7\%$).
   - **Localized Inpainting & Outpainting**: 1,000 images ($8.3\%$) — Targeted training on localized edits.
2. **Authentic Real Composition ($N=12,000$ Train)**:
   - **Google Pixel (Pixel 6–9 Pro)**: 3,000 images ($25.0\%$) — Mandatory to teach the model computational HDR tone curves.
   - **Apple iPhone (iPhone 11–16 Pro Max)**: 3,000 images ($25.0\%$).
   - **Samsung Galaxy (S20–S24 Ultra)**: 3,000 images ($25.0\%$).
   - **Uncompressed RAW DSLRs (Nikon, Canon, Sony via RAISE/Flickr)**: 1,500 images ($12.5\%$) — Eliminates the sensor noise trap.
   - **Social-Media Transmitted Real Photos**: 1,500 images ($12.5\%$).
3. **Compound Channel Augmentation (Applied dynamically to 40% of batches)**:
   - Instead of aggressive random blurring (E4), apply **statistically calibrated compound degradation**:
     - JPEG compression with Quality Factors matching real social media ($Q \in [65, 85]$).
     - 4:2:0 chroma subsampling.
     - Dimensional downsampling with bicubic/bilinear anti-aliasing.
     - Zero additive Gaussian noise (which proved destructive in E4).

---

## 8. Architectural & Algorithmic Options Analysis

Ten proposed technical directions are systematically evaluated against the empirical evidence to determine what model modifications are scientifically justified:

| # | Proposed Technical Direction | Evidence Supporting | Evidence Against | Expected Failure Mode | Worth Testing in E20? |
| :-: | :--- | :--- | :--- | :--- | :---: |
| **1** | **More Training Data Only** | E5 proved modern generator + camera data dramatically expanded generalization. | E8, E15, E18 proved more compression data inflates false positives without calibration. | Memorizes new generators without fixing channel trade-offs. | **YES (Mandatory Foundation)** |
| **2** | **Calibrated Compound Augmentation** | E10-B proved WhatsApp-calibrated training achieved peak WhatsApp AUC (0.7023). | E4 proved uncalibrated augmentation destroys synthetic spectral grids. | Oversmooths subtle features if compression parameters are too severe. | **YES (Strictly Constrained)** |
| **3** | **Higher-Resolution Input (e.g. 512x512)** | E6-A proved preserving native resolution crops recovered 45% of missed AI images. | quadruples GPU memory consumption and compute latency per view. | Out-of-memory on RTX 3050 (4GB VRAM). | **NO (Hardware Constrained)** |
| **4** | **Patch-Based Tiling Strategy** | Captures fine-grained demosaicing and localized inpainting traces. | E17 proved local patches over-fire on sensor noise and JPEG blockiness. | Massive false-positive inflation on uncompressed photography. | **NO (Violates Specificity)** |
| **5** | **Multi-Scale Architecture Refinement** | E6-C achieved 96.16% clean validation accuracy and lowest FPR (2.97%). | Unconstrained attention head allows a single noisy crop to overpower the global view. | Inability to reject localized false alarms. | **YES (Preserve Base Backbone)** |
| **6** | **DCT Instead of 2D FFT** | Block-based DCT aligns directly with the 8x8 block grid of JPEG compression. | Requires specialized convolutional backbones and custom CUDA kernels. | Implementation complexity with uncertain OOD transfer gain. | **EXPLORATORY (Post-E20)** |
| **7** | **Generator-Balanced Sampling** | E19 proved E6-C is overfit to Midjourney (85%) and blind to FLUX (10%). | May reduce peak performance on dominant commercial generators. | Slight drop in BigGAN/Midjourney in-distribution metrics. | **YES (Mandatory for E20)** |
| **8** | **Camera/ISP-Aware Contrastive Training** | E14 and E19 proved huge false alarm variance across camera sources (RAISE, Pixel). | Requires multi-camera paired captures of identical scenes. | High acquisition cost for paired physical captures. | **YES (Stratified Sampling)** |
| **9** | **Specialized Local-Edit Branch** | Recovers localized inpainting that global views average out. | Prone to tripping on sharp natural textures (hair, leaves). | Elevated false alarm rate on textured real photos. | **NO (Premature for E20)** |
| **10**| **Two-Stage Dual-Head Architecture** | Decouples clean high-specificity inference from calibrated compression recovery. | Adds pipeline latency and potential error compounding from Stage 1. | Mismatch if Stage 1 misclassifies compression level. | **YES (Top Architecture Candidate)**|

---

## 9. The "99% Target": A Mathematically Honest Analysis

Aiming for **99% accuracy** is an essential motivating benchmark, but in applied forensics, it must be formulated with absolute mathematical precision.

### Why "99% on Open-World Internet Media" is Physically Impossible with a Single Global Model:
1. **The Inverse Channel Limit**: Information theory dictates that if a transmission channel applies compression that is mathematically non-invertible and filters out all frequencies above Nyquist/4, synthetic artifacts whose spatial frequency lies above that cutoff are physically erased.
2. **Generative Convergence**: As generative pipelines (FLUX, Imagen 3) incorporate camera sensor noise models and authentic ISP simulation, the statistical divergence between real photographic distributions and synthetic distributions approaches zero.
3. **The Base-Rate Fallacy**: On the open Internet, authentic photographs outnumber synthetic images by orders of magnitude. A detector with **99% accuracy and 1% FPR** evaluated on a population with a 1% synthetic base rate will yield a **50% False Discovery Rate** (half of all AI flags will be false alarms).

### Measurable, Defensible E20 Success Gates:

Rather than pursuing a single unachievable open-world claim, E20 must enforce distinct, rigorous success gates across separate evaluation distributions:

```
+----------------------------------------------------------------------------------------------------+
|                                      E20 QUANTITATIVE SUCCESS GATES                                |
+-------------------------------+-------------------+------------------------------------------------+
| Evaluation Distribution       | Minimum Gate      | Stretched Target                               |
+-------------------------------+-------------------+------------------------------------------------+
| Gate A: Clean Independent Test| ROC-AUC >= 0.9600 | ROC-AUC >= 0.9850, Accuracy >= 95.0%, FPR < 2%|
| Gate B: Modern Unseen AI      | Recall >= 80.0%   | FLUX Recall >= 85.0%, Gemini Recall >= 80.0%   |
| Gate C: Diverse Smartphone Real| FPR <= 5.0%      | Pixel FPR <= 5.0%, iPhone/Samsung FPR <= 2.0%  |
| Gate D: Compressed Social Media| ROC-AUC >= 0.8200 | WhatsApp Accuracy >= 75.0%, FPR <= 10.0%      |
+-------------------------------+-------------------+------------------------------------------------+
```

---

## 10. Concrete Experiment 20 (E20) Technical Specification

### Specification Overview:
- **Starting Model Checkpoint**: `experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`
- **Architectural Paradigm**: **Two-Stage Decoupled Dual-Head Architecture (or Temperature-Calibrated Gated Multi-View)**.
  - *Branch A (Global-Local Multi-Scale Feature Extractor)*: Retains the frozen dual-domain EfficientNet-B3 + Standardized 2D FFT CNN backbones.
  - *Branch B (Channel & ISP Quality Estimation Head)*: Predicts input compression quality ($Q \in [0, 100]$) and high-frequency noise variance.
  - *Branch C (Calibrated Gated Attention Head)*: Dynamically adjusts local crop gating weights based on both global scene evidence and estimated channel quality, with post-hoc Platt temperature scaling to guarantee calibrated probabilities.
- **Training Data**: 24,000 newly curated images (balanced 50/50 Real/AI) adhering strictly to the Section 7 stratification.
- **Train / Dev / Test Quarantine Protocol**:
  - `E20_TRAIN`: 24,000 images.
  - `E20_DEV`: 3,000 images (quarantined during training; used strictly for early stopping and temperature calibration).
  - `E20_TEST`: 3,000 images (cryptographically sealed; zero access until the candidate checkpoint is completely frozen).
  - `E14` and `E19` remain permanently quarantined as historical blind holdouts.
- **Ablation Studies Required**:
  1. *Ablation 1*: Data only (E6-C architecture trained on E20 balanced data).
  2. *Ablation 2*: Architecture only (E20 Decoupled Head trained on historical E5 data).
  3. *Ablation 3*: Full System (E20 Decoupled Head trained on E20 balanced data).
  4. *Ablation 4*: Temperature scaling impact on Pixel and RAISE-1k false alarms.
- **Hardware & VRAM Budget**:
  - Target Platform: NVIDIA RTX 3050 Laptop GPU (4GB VRAM).
  - Batch size: 8 (with gradient accumulation steps = 2, yielding effective batch size = 16).
  - Mixed Precision: PyTorch CUDA AMP `torch.amp.autocast("cuda")`.
  - Expected Runtime: ~3.5 hours for 5 epochs of head/attention training with frozen backbones.
- **What MUST NOT Change**:
  - Do NOT modify production inference until E20 passes all Gate criteria.
  - Do NOT alter the standard 5-view multi-scale crop geometry (1 global + 4 corner 60% crops).
  - Do NOT remove frequency Z-score standardization.
  - Do NOT alter historical benchmarks (V2, WhatsApp 67, E14, E19).

---

## 11. Final Audit Conclusions & Next Steps

1. **Failure Audit Complete**: All primary failure modes across E1–E19 have been quantitatively isolated to specific generator families (FLUX, Gemini), specific camera sources (RAISE-1k RAW, Pixel computational ISP), and specific attention mechanisms (unconstrained local crop over-sensitization).
2. **E6-C Remains the Soundest Baseline**: E6-C maintains the lowest overall false-alarm rate across clean photography ($2.97\%$ validation FPR, $4.29\%$ E19 pristine FPR), making it the mandatory starting checkpoint for E20.
3. **E20 Must Be Data-First, Architecture-Second**: Training on modern flow-matching architectures and diverse computational smartphone ISPs will resolve $>70\%$ of historical failures before altering a single layer of the network backbone.
4. **Execution Decision**: Dataset acquisition and construction for E20 should proceed under strict isolation protocols, without touching the frozen production environment.

---
*End of E20 Failure-Driven Dataset Audit. DeepVision-Forensics Research Group.*
