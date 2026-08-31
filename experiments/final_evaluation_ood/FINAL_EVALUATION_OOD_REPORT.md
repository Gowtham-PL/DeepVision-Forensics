# DeepVision Forensics — Final Unseen-Generator Evaluation Report
**Evaluation of OOD Candidate (`standardize`) on Unseen BigGAN and Midjourney Benchmarks**

---

## 1. Executive Summary

This report documents the **first and only unblinded test evaluation** of the selected Out-of-Distribution (OOD) Candidate architecture: **Dual-Domain (E3) with Frequency Standardization (`standardize`)**.

The candidate model was trained exclusively on the five development generators (**ADM, GLIDE, SDv5, VQDM, Wukong**) and evaluated deterministically against the quarantined, completely unseen test benchmarks: **BigGAN** (GAN paradigm) and **Midjourney** (Proprietary diffusion paradigm).

### Key Test Findings:
1. **Generalization Improvement over Dual-Domain (E3 Baseline):**
   - Candidate `standardize` achieves an **Overall Unseen ROC-AUC of 0.8959** (vs E3 Dual baseline of **0.8851**, $\Delta = \mathbf{+0.0108}$ / **+1.08%**).
   - Achieves consistent gains across both test generators: **BigGAN ROC-AUC = 0.9511** (vs 0.9465, $\Delta = \mathbf{+0.0046}$) and **Midjourney ROC-AUC = 0.8392** (vs 0.8228, $\Delta = \mathbf{+0.0163}$).
2. **Comparison Against Spatial-Only (E1 Baseline):**
   - On the harder commercial diffusion engine (**Midjourney**), Candidate `standardize` **outperforms the E1 Spatial baseline** (**0.8392 vs 0.8224, $\Delta = \mathbf{+0.0167}$**).
   - On the GAN engine (**BigGAN**), E1 Spatial retains higher discrimination (**0.9732 vs 0.9511**), yielding an Overall ROC-AUC of **0.8959 vs 0.8991** ($\Delta = -0.0031$).
   - Candidate `standardize` achieves superior **Precision (92.56% vs 90.03%)**, producing significantly fewer False Positives on authentic photographs (**223 vs 390**).

---

## 2. Unseen Test Benchmark Leaderboard

Evaluated on $N=9,999$ test samples (1 cross-generator duplicate excluded):

| Model | Architecture | Frequency Norm | Overall Unseen ROC-AUC | Overall Unseen PR-AUC | Overall Accuracy | Precision | Recall | F1-Score | BigGAN ROC-AUC | Midjourney ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate OOD** | **Dual-Domain (E3)** | **`standardize`** | **0.8959** | **0.8993** | **75.52%** | **92.56%** | 55.50% | **0.6939** | **0.9511** | **0.8392** |
| **E1 Spatial (Frozen)** | Spatial-Only | N/A | **0.8991** | 0.9079 | **81.31%** | 90.03% | 70.42% | 0.7903 | **0.9732** | 0.8224 |
| **E3 Dual (Frozen)** | Dual-Domain (E3) | `minmax` | 0.8851 | 0.8908 | 77.25% | 90.37% | 61.00% | 0.7284 | 0.9465 | 0.8228 |
| **E2 Freq (Frozen)** | Frequency-Only | `minmax` | 0.6366 | 0.6438 | 58.74% | 77.56% | 23.44% | 0.3599 | 0.6385 | 0.6358 |

---

## 3. Exact Performance Deltas

### 3.1 Candidate (`standardize`) vs Dual-Domain Baseline (E3 `minmax`)

$$\begin{array}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{E3 Dual (minmax)} & \textbf{Candidate (standardize)} & \textbf{Delta } (\Delta) \\
\hline
\text{Overall Unseen ROC-AUC} & 0.8851 & \mathbf{0.8959} & \mathbf{+0.0108} \text{ (+1.08\%)} \\
\text{Overall Unseen PR-AUC} & 0.8908 & \mathbf{0.8993} & \mathbf{+0.0085} \text{ (+0.85\%)} \\
\text{BigGAN ROC-AUC} & 0.9465 & \mathbf{0.9511} & \mathbf{+0.0046} \text{ (+0.46\%)} \\
\text{Midjourney ROC-AUC} & 0.8228 & \mathbf{0.8392} & \mathbf{+0.0163} \text{ (+1.63\%)} \\
\text{Precision} & 90.37\% & \mathbf{92.56\%} & \mathbf{+2.19\%} \\
\text{False Positive Count (Real flagged as AI)} & 325 & \mathbf{223} & \mathbf{-102 \text{ (-31.4\% FP reduction)}} \\
\text{Overall Accuracy (Fixed 0.50 threshold)} & 77.25\% & 75.52\% & -1.73\% \\
\text{F1-Score (Fixed 0.50 threshold)} & 0.7284 & 0.6939 & -0.0345 \\
\hline
\end{array}$$

### 3.2 Candidate (`standardize`) vs Spatial Baseline (E1)

$$\begin{array}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{E1 Spatial} & \textbf{Candidate (standardize)} & \textbf{Delta } (\Delta) \\
\hline
\text{Overall Unseen ROC-AUC} & \mathbf{0.8991} & 0.8959 & -0.0031 \text{ (-0.31\%)} \\
\text{Midjourney ROC-AUC} & 0.8224 & \mathbf{0.8392} & \mathbf{+0.0167} \text{ (+1.67\%)} \\
\text{BigGAN ROC-AUC} & \mathbf{0.9732} & 0.9511 & -0.0221 \text{ (-2.21\%)} \\
\text{Precision} & 90.03\% & \mathbf{92.56\%} & \mathbf{+2.53\%} \\
\text{False Positive Count} & 390 & \mathbf{223} & \mathbf{-167 \text{ (-42.8\% FP reduction)}} \\
\text{Overall Accuracy} & \mathbf{81.31\%} & 75.52\% & -5.79\% \\
\hline
\end{array}$$

---

## 4. Breakdown by Test Generator Family

### 4.1 BigGAN ($N=5,000$, GAN Paradigm)
- **Sample Distribution:** 2,500 Real Photographs, 2,500 BigGAN AI Images.
- **Candidate Performance:** ROC-AUC = **0.9511**, PR-AUC = **0.9494**, Accuracy = **84.46%**, Precision = **93.75%**, Recall = **73.84%**, F1 = **0.8261**.
- **Confusion Matrix:** TP = 1,846, FP = 123, TN = 2,377, FN = 654.
- **Analysis:** BigGAN generates pronounced periodic spectral grid artifacts. The standardized frequency encoder reliably captures these spectral footprints while maintaining high precision.

### 4.2 Midjourney ($N=4,999$, Proprietary Diffusion Paradigm)
- **Sample Distribution:** 2,499 Real Photographs (1 duplicate excluded), 2,500 Midjourney AI Images.
- **Candidate Performance:** ROC-AUC = **0.8392**, PR-AUC = **0.8374**, Accuracy = **66.57%**, Precision = **90.28%**, Recall = **37.16%**, F1 = **0.5265**.
- **Confusion Matrix:** TP = 929, FP = 100, TN = 2,399, FN = 1,571.
- **Analysis:** Midjourney represents the most challenging benchmark due to its advanced proprietary post-processing pipeline. By standardizing spectral energy distributions, Candidate `standardize` extracted invariant high-frequency cues that allowed it to outperform both E1 Spatial (+0.0167 AUC) and E3 Dual (+0.0163 AUC).

---

## 5. Scientific Conclusions & Methodology Assessment

1. **Validation of the LOGO Selection Protocol:**
   - The 5-fold Leave-One-Generator-Out development protocol successfully identified `standardize` as superior to `minmax`.
   - On the completely unseen test split, this prediction was **empirically validated**: `standardize` improved Dual-Domain ROC-AUC by **+1.08% overall** (+0.46% on BigGAN, +1.63% on Midjourney).
2. **Impact of Frequency Standardization:**
   - Replacing per-image Min-Max scaling with per-image Z-Score standardization prevents extreme spectral DC/high-frequency outlier spikes from compressing subtle harmonic traces.
   - This provides stronger cross-generator invariance, notably boosting detection of state-of-the-art commercial diffusion models (Midjourney).
3. **Operational Trade-Off:**
   - Candidate `standardize` operates with high specificity (Precision 92.56%, FP rate 4.46%), making it highly reliable for forensics workflows where falsely accusing authentic media must be minimized.
