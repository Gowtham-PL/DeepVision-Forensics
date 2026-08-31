# DeepVision Forensics — Out-of-Distribution (OOD) Development Report
**Leave-One-Generator-Out (LOGO) Cross-Validation & Frequency Normalization Ablation**

---

## 1. Executive Summary

This report presents the design, execution, and empirical findings of a rigorous **5-Fold Leave-One-Generator-Out (LOGO) Development Protocol** to evaluate and improve cross-generator generalization in `DeepVision-Forensics`.

To prevent data snooping and maintain the absolute integrity of our final unseen-generator benchmarks, **the test generators (BigGAN and Midjourney) and all existing E1/E2/E3 checkpoints and final evaluation artifacts remain 100% frozen and unaccessed**. Model development, model selection, and frequency normalization ablation were conducted exclusively across the five development generators: **ADM, GLIDE, SDv5, VQDM, and Wukong**.

### Key Findings:
- **Baseline Protocol Metric (`minmax`):** Under the 5-fold LOGO protocol, the baseline Dual-Domain E3 model achieves a **Mean LOGO ROC-AUC of 0.8528** (Std: 0.1127), with a worst-case ROC-AUC of 0.7020 (VQDM) and best-case of 0.9605 (GLIDE).
- **Candidate Normalization Comparison:**
  1. **`standardize` (Per-Image Z-Score):** Achieves **Mean LOGO ROC-AUC of 0.8675** (Std: 0.1032, Mean Accuracy: 74.18%, Mean PR-AUC: 0.8670, Mean F1: 0.6408), strictly outperforming the `minmax` baseline on **all 5 folds**. It notably raises the worst-case generator performance from 0.7020 to 0.7328 on VQDM (+0.0308) and 0.7294 to 0.7509 on ADM (+0.0215).
  2. **`none` (Raw Log Spectrum):** Achieves **Mean LOGO ROC-AUC of 0.8569** (Std: 0.1000, Mean Accuracy: 73.96%, Mean PR-AUC: 0.8483, Mean F1: 0.6386).
  3. **`minmax` (Per-Image Min-Max Scaling):** Achieves **Mean LOGO ROC-AUC of 0.8528** (Std: 0.1127, Mean Accuracy: 73.68%, Mean PR-AUC: 0.8414, Mean F1: 0.6406).
- **Selection Decision:** **Candidate `standardize` is selected for final evaluation.**

> [!IMPORTANT]
> **Scientific Integrity Notice:** In accordance with rigorous machine learning methodology, no claim is made that `standardize` improves unseen-generator performance on the held-out final test set. We report only: *Based on the five-generator leave-one-generator-out development protocol, candidate `standardize` is selected for final evaluation.* The final test set (BigGAN and Midjourney) has not been evaluated.

---

## 2. Leave-One-Generator-Out (LOGO) Protocol Design

### 2.1 Motivation & Rationale
Standard in-distribution cross-validation (splitting randomly across all images) inflates performance estimates because generator-specific spectral and visual artifacts leak across the train and validation partitions. To faithfully simulate the challenge of detecting completely unseen AI generation engines, the **Leave-One-Generator-Out (LOGO)** protocol systematically partitions the dataset by generator family.

### 2.2 Fold Partition Matrix
For each fold $k \in \{1, 2, 3, 4, 5\}$, exactly one development generator is isolated for validation, while the remaining four generators form the training set:

| Fold Index | Held-Out Validation Generator | Training Generators | Train Samples ($N$) | Val Samples ($N$) |
| :--- | :--- | :--- | :---: | :---: |
| **Fold 1** | **ADM** | GLIDE, SDv5, VQDM, Wukong | 16,000 | 1,000 |
| **Fold 2** | **GLIDE** | ADM, SDv5, VQDM, Wukong | 16,000 | 1,000 |
| **Fold 3** | **SDv5** | ADM, GLIDE, VQDM, Wukong | 16,000 | 1,000 |
| **Fold 4** | **VQDM** | ADM, GLIDE, SDv5, Wukong | 16,000 | 1,000 |
| **Fold 5** | **Wukong** | ADM, GLIDE, SDv5, VQDM | 16,000 | 1,000 |

### 2.3 Strict Zero-Leakage Guarantees
- **Forbidden Test Generators:** `BigGAN` and `Midjourney` are strictly quarantined. Hard assertions (`assert 'BigGAN' not in dataset.generators`, `assert 'Midjourney' not in dataset.generators`) are embedded into the data loader pipeline.
- **Generator Isolation:** For any fold $k$, the held-out generator is guaranteed to have 0 samples in the training loader.
- **Class Balance:** Every training set consists of exactly 8,000 Real (`nature`) images and 8,000 AI-generated images (2,000 AI images per training generator). Every validation set consists of 500 Real (`nature`) images and 500 AI-generated images from the held-out generator.

---

## 3. Frequency Normalization Strategy Ablation

Using the Dual-Domain E3 architecture as the invariant baseline, we ablate the frequency normalization transform applied to the centered 2D log-magnitude spectrum $S = \log(1 + |F(u, v)|)$:

1. **`minmax` (Baseline):**
   $$S_{\text{norm}} = \frac{S - \min(S)}{\max(S) - \min(S) + \epsilon}$$
   Scales spectral amplitudes to the dynamic range $[0, 1]$ per image. While intuitive, it is sensitive to extreme DC or high-frequency spike outliers that can compress fine-grained spectral harmonic traces.

2. **`standardize` (Candidate A — Z-Score):**
   $$S_{\text{norm}} = \frac{S - \mu(S)}{\sigma(S) + \epsilon}$$
   Zero-centers and normalizes spectral energy variance per image. This standardizes the energy distribution across differing luminance and frequency levels, preserving subtle periodic artifacts and harmonic grid patterns invariant to overall image brightness or dynamic range.

3. **`none` (Candidate B — Raw Log Magnitude):**
   $$S_{\text{norm}} = S = \log(1 + |F(u, v)|)$$
   Preserves absolute spectral log-amplitudes without per-image normalization, allowing the convolutional frequency encoder to directly learn from absolute energy differentials.

---

## 4. Development Experiment Setup & Hyperparameters

To ensure strict scientific rigor, all hyperparameters, architectural backbones, seeds, optimizers, and loss formulations were held strictly identical across all 15 models (3 strategies $\times$ 5 folds):

- **Spatial Branch:** EfficientNet-B3 (`pretrained=True`), fine-tuned, outputting a 1536-D spatial embedding with internal ImageNet normalization.
- **Frequency Branch:** 2D FFT (`torch.fft.fft2`) $\to$ Centered Log-Magnitude $\to$ Configurable Normalization $\to$ 4-block CNN (`Conv2d-BN-ReLU-MaxPool`) $\to$ AdaptiveAvgPool2d((1,1)) $\to$ Linear projection to 256-D embedding.
- **Fusion Head:** Concatenation to 1792-D $\to$ `Linear(1792, 512)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.4)` $\to$ `Linear(512, 128)` $\to$ `ReLU` $\to$ `Dropout(0.2)` $\to$ `Linear(128, 1)`.
- **Loss Function:** `nn.BCEWithLogitsLoss()`.
- **Optimizer:** `AdamW` (`weight_decay=1e-4`).
  - Backbone learning rate: `1e-5`
  - Classifier head & frequency branch learning rate: `5e-4`
- **Learning Rate Schedule:** Linear Warmup (2 epochs) followed by Cosine Annealing (8 epochs) over 10 total epochs.
- **Batch Size:** 16.
- **Hardware & Precision:** NVIDIA GPU with Mixed Precision (`torch.cuda.amp.autocast`).
- **Random Seed:** 42 (deterministic fold generation and model parameter initialization).

---

## 5. Detailed 5-Fold LOGO Validation Results

### 5.1 Strategy 1: `minmax` (Baseline)

| Fold | Held-Out Generator | Train $N$ | Val $N$ | Best Epoch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | TN | FN |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ADM** | 16,000 | 1,000 | 4 | 0.5930 | 0.7268 | 0.2980 | 0.4227 | 0.7294 | 0.6897 | 149 | 56 | 444 | 351 |
| 2 | **GLIDE** | 16,000 | 1,000 | 8 | 0.8390 | 0.9426 | 0.7220 | 0.8177 | 0.9605 | 0.9510 | 361 | 22 | 478 | 139 |
| 3 | **SDv5** | 16,000 | 1,000 | 10 | 0.8380 | 0.9225 | 0.7380 | 0.8200 | 0.9399 | 0.9389 | 369 | 31 | 469 | 131 |
| 4 | **VQDM** | 16,000 | 1,000 | 4 | 0.5730 | 0.7967 | 0.1960 | 0.3146 | 0.7020 | 0.6939 | 98 | 25 | 475 | 402 |
| 5 | **Wukong** | 16,000 | 1,000 | 10 | 0.8410 | 0.9012 | 0.7660 | 0.8281 | 0.9324 | 0.9336 | 383 | 42 | 458 | 117 |

- **Aggregate `minmax`:** Mean ROC-AUC = **0.8528** $\pm$ 0.1127 | Mean Acc = 0.7368 | Mean PR-AUC = 0.8414 | Mean F1 = 0.6406

---

### 5.2 Strategy 2: `standardize` (Z-Score Normalization)

| Fold | Held-Out Generator | Train $N$ | Val $N$ | Best Epoch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | TN | FN |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ADM** | 16,000 | 1,000 | 8 | 0.5990 | 0.8462 | 0.2420 | 0.3764 | **0.7509** | **0.7607** | 121 | 22 | 478 | 379 |
| 2 | **GLIDE** | 16,000 | 1,000 | 6 | 0.8240 | 0.9500 | 0.6840 | 0.7953 | **0.9634** | **0.9582** | 342 | 18 | 482 | 158 |
| 3 | **SDv5** | 16,000 | 1,000 | 10 | **0.8490** | 0.9185 | 0.7660 | **0.8353** | **0.9545** | **0.9544** | 383 | 34 | 466 | 117 |
| 4 | **VQDM** | 16,000 | 1,000 | 7 | **0.5870** | 0.7959 | 0.2340 | **0.3617** | **0.7328** | **0.7251** | 117 | 30 | 470 | 383 |
| 5 | **Wukong** | 16,000 | 1,000 | 10 | **0.8500** | 0.9248 | 0.7620 | **0.8355** | **0.9359** | **0.9364** | 381 | 31 | 469 | 119 |

- **Aggregate `standardize`:** Mean ROC-AUC = **0.8675** $\pm$ 0.1032 | Mean Acc = 0.7418 | Mean PR-AUC = 0.8670 | Mean F1 = 0.6408

---

### 5.3 Strategy 3: `none` (Raw Log Magnitude Spectrum)

| Fold | Held-Out Generator | Train $N$ | Val $N$ | Best Epoch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TP | FP | TN | FN |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ADM** | 16,000 | 1,000 | 4 | 0.5780 | 0.7910 | 0.2120 | 0.3344 | 0.7420 | 0.7092 | 106 | 28 | 472 | 394 |
| 2 | **GLIDE** | 16,000 | 1,000 | 8 | 0.8390 | 0.9313 | 0.7320 | 0.8197 | 0.9470 | 0.9381 | 366 | 27 | 473 | 134 |
| 3 | **SDv5** | 16,000 | 1,000 | 10 | 0.8330 | 0.9152 | 0.7340 | 0.8147 | 0.9372 | 0.9372 | 367 | 34 | 466 | 133 |
| 4 | **VQDM** | 16,000 | 1,000 | 4 | 0.6040 | 0.8377 | 0.2580 | 0.3945 | 0.7275 | 0.7241 | 129 | 25 | 475 | 371 |
| 5 | **Wukong** | 16,000 | 1,000 | 10 | 0.8440 | 0.9135 | 0.7600 | 0.8297 | 0.9307 | 0.9330 | 380 | 36 | 464 | 120 |

- **Aggregate `none`:** Mean ROC-AUC = **0.8569** $\pm$ 0.1000 | Mean Acc = 0.7396 | Mean PR-AUC = 0.8483 | Mean F1 = 0.6386

---

## 6. Aggregate Performance & Candidate Comparison Leaderboard

| Rank | Strategy | Mean ROC-AUC | Std ROC-AUC | Worst-Case ROC-AUC | Best-Case ROC-AUC | Mean PR-AUC | Mean Accuracy | Mean F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`standardize`** | **0.8675** | **0.1032** | **0.7328** (VQDM) | **0.9634** (GLIDE) | **0.8670** | **0.7418** | **0.6408** |
| 🥈 **2** | **`none`** | **0.8569** | **0.1000** | **0.7275** (VQDM) | **0.9470** (GLIDE) | **0.8483** | **0.7396** | **0.6386** |
| 🥉 **3** | **`minmax`** (Baseline) | **0.8528** | **0.1127** | **0.7020** (VQDM) | **0.9605** (GLIDE) | **0.8414** | **0.7368** | **0.6406** |

### Per-Fold ROC-AUC Comparison:
$$\begin{array}{|l|c|c|c|c|}
\hline
\textbf{Held-Out Generator} & \textbf{minmax (Baseline)} & \textbf{standardize} & \textbf{none} & \textbf{Delta vs Baseline (standardize)} \\
\hline
\text{Fold 1: ADM} & 0.7294 & \textbf{0.7509} & 0.7420 & +0.0215 \\
\text{Fold 2: GLIDE} & 0.9605 & \textbf{0.9634} & 0.9470 & +0.0030 \\
\text{Fold 3: SDv5} & 0.9399 & \textbf{0.9545} & 0.9372 & +0.0146 \\
\text{Fold 4: VQDM} & 0.7020 & \textbf{0.7328} & 0.7275 & +0.0308 \\
\text{Fold 5: Wukong} & 0.9324 & \textbf{0.9359} & 0.9307 & +0.0034 \\
\hline
\textbf{Mean LOGO AUC} & 0.8528 & \textbf{0.8675} & 0.8569 & \mathbf{+0.0147} \\
\hline
\end{array}$$

---

## 7. Hard-Fold Generator Analysis (ADM & VQDM vs GLIDE, SDv5, Wukong)

The 5-fold LOGO cross-validation reveals a striking structural bifurcation in generator generalization:

1. **Diffusion/Latent Super-Generalizers (GLIDE, SDv5, Wukong):**
   - Held-out ROC-AUCs consistently exceed **0.935–0.963**.
   - These models share common latent diffusion backbones and similar spatial upsampling footprints. Training on any combination of them imparts strong spatial and spectral priors that generalize well to the other diffusion engines.
2. **Distinct Architectural Paradigms (ADM and VQDM):**
   - **ADM (Ablated Diffusion Model — Pixel-space Diffusion):** Held-out ROC-AUC drops to ~0.73–0.75 across all models. Pixel-space diffusion exhibits vastly different high-frequency noise profiles compared to latent diffusion models.
   - **VQDM (Vector Quantized Diffusion Model — Discrete Tokenizer):** Represents the hardest generator across all strategies (worst-case ROC-AUC 0.7020 in baseline, 0.7328 in standardize). The discrete codebook quantization introduces localized high-frequency grid artifacts that are not encountered when training exclusively on continuous diffusion engines.
3. **Why `standardize` Outperforms `minmax` on Hard Generators:**
   - In `minmax`, the presence of a few extreme spectral peaks (e.g. DC component or tokenizer grid spikes) forces the remainder of the spectrum into a narrow band near zero, destroying subtle harmonic traces.
   - In `standardize`, per-image mean subtraction and standard deviation scaling preserves relative spectral contrast regardless of whether the generator operates in continuous latent space or discrete codebook space. This yields significant gains on the hardest folds: **+0.0308 on VQDM** and **+0.0215 on ADM**.

---

## 8. Candidate Strategy Selection & Scientific Justification

### Formal Decision:
**Based on the five-generator leave-one-generator-out development protocol, candidate `standardize` is selected for final evaluation.**

### Empirical Justification:
1. **Unanimous Fold Superiority:** `standardize` achieved higher validation ROC-AUC than the `minmax` baseline on **every single one of the 5 folds** (+0.0215 on ADM, +0.0030 on GLIDE, +0.0146 on SDv5, +0.0308 on VQDM, +0.0034 on Wukong).
2. **Robustness to Worst-Case Distribution Shifts:** It elevated the worst-case generator floor from 0.7020 to 0.7328 (+0.0308), reducing the performance variance across generator families (Std AUC: 0.1032 vs 0.1127).
3. **Superior Precision-Recall Profile:** Mean PR-AUC improved from 0.8414 to **0.8670** (+0.0256), indicating significantly better confidence calibration on unseen AI images.

---

## 9. Next Steps (Pre-Registration for Final Evaluation)

Having concluded the development phase under strict quarantine of the final evaluation set:
1. **Candidate Pre-Registration:** The selected architecture is **Dual-Domain E3 with Frequency Branch Standardization (`norm_strategy='standardize'`)**.
2. **Full-Training Protocol:** Train the final candidate model on all 5 development generators (20,000 training images, 5,000 validation images) using the exact same hyperparameters (10 epochs, AdamW, warmup + cosine schedule, seed 42).
3. **Unblinded Test Evaluation:** Only after the candidate checkpoint is finalized, evaluate against the frozen test set:
   - BigGAN ($N=5,000$, GAN paradigm)
   - Midjourney ($N=5,000$, Proprietary Diffusion paradigm)
   - Cross-generator deduplication filter strictly enforced.
