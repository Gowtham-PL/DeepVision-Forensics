# DeepVision-Forensics: Final Research Results Report

**Dual-Domain Spatial and Frequency Analysis for Generalized AI-Generated Image Detection**

---

## 1. Experimental Objective

The primary research objective of **DeepVision-Forensics** is to evaluate the generalization capacity of machine learning architectures designed to detect AI-generated imagery when confronted with **completely unseen image generators**. 

Specifically, this study investigates the central research hypothesis:
> *"Does adding frequency-domain representations (extracted via 2D Fast Fourier Transform) through feature-level concatenation improve out-of-distribution generalization to unseen AI generator architectures beyond spatial visual features alone?"*

To test this hypothesis rigorously and avoid data contamination, a strictly controlled three-model experimental paradigm was executed on an identical dataset split protocol with zero test leakage.

---

## 2. Dataset and Generator Split

The study utilizes a subset of the **GenImage** benchmark comprising **35,000 total images** (17,500 authentic photographic "nature" images from ImageNet-1K and 17,500 AI-synthesized images across 7 generator paradigms). The partitioning was established *a priori* in `data/manifest.csv` and frozen throughout all training and evaluation phases.

### Data Partitioning Table

| Split | Image Count | Authentic ("nature") | Synthetic ("AI") | Canonical Generators | Role in Benchmark |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Train** | **20,000** | 10,000 | 10,000 | `ADM`, `GLIDE`, `SDv5`, `VQDM`, `Wukong` | Model optimization (4,000 images per generator) |
| **Validation** | **5,000** | 2,500 | 2,500 | `ADM`, `GLIDE`, `SDv5`, `VQDM`, `Wukong` | In-distribution model & checkpoint selection |
| **Test (Holdout)**| **10,000** | 5,000 | 5,000 | `BigGAN`, `Midjourney` | Out-of-distribution generalization testing |

### Cross-Generator Duplicate Handling Rule
An automated pre-training dataset hash audit identified exactly **1 duplicate authentic image pair** (`n02109961_185.JPEG`) shared between ImageNet source subsets across `ADM/train` and `Midjourney/train`. To guarantee zero-leakage evaluation while preserving manifest integrity:
- The record was retained in `data/manifest.csv` with metadata flag `is_cross_gen_dup = True`.
- For **final test metric evaluation**, this single duplicate was filtered at runtime:
  - Total test records before filter: **10,000**
  - Excluded records: **1** (Midjourney sub-split)
  - Final evaluated test sample count: **9,999** (BigGAN: **5,000**, Midjourney: **4,999**)

---

## 3. Experimental Setup & Reproducibility

All experiments were trained and evaluated under strictly controlled, identical computational conditions on an **NVIDIA GeForce RTX 3050 Laptop GPU** (4GB physical VRAM, CUDA 12.6, PyTorch 2.13.0+cu126).

```
Hardware Platform:      NVIDIA GeForce RTX 3050 Laptop GPU
Operating System:       Windows 11 (Driver 610.47, CUDA 12.6)
Python Environment:     Python 3.11.9, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126
Mixed Precision:        CUDA AMP (torch.amp.autocast float16 with GradScaler)
Random Seed:            42 (fixed across all data loading and model initialization)
Batch Size:             16 (training) / 32 (evaluation inference)
Total Epochs:           10 epochs (Warmup: 2 epochs, Cosine Decay)
Loss Function:          BCEWithLogitsLoss()
Optimizer:              AdamW (Weight decay = 1e-4)
Learning Rates:         Backbone = 1e-5 (fine-tuning), Newly Initialized Layers = 5e-4
Checkpoint Metric:      Strictly Validation ROC-AUC (zero test-set visibility)
Decision Threshold:     0.50 (fixed, non-tuned)
```

---

## 4. Model Architecture Comparison

| Architectural Dimension | E1: Spatial-Only Baseline | E2: Frequency-Only Baseline | E3: Dual-Domain Fusion Model |
| :--- | :--- | :--- | :--- |
| **Model Class** | `SpatialClassifier` | `FrequencyClassifier` | `DeepVisionFusionModel` |
| **Spatial Backbone** | EfficientNet-B3 (ImageNet-1K Pretrained) | *None* | EfficientNet-B3 (ImageNet-1K Pretrained) |
| **Spatial Embedding** | 1536-D vector (GAP output) | *None* | 1536-D vector (GAP output) |
| **Frequency Transform** | *None* | 2D FFT Log-Magnitude (`minmax` norm) | 2D FFT Log-Magnitude (`minmax` norm) |
| **Frequency Encoder** | *None* | 4-Block CNN + GAP + Projection | 4-Block CNN + GAP + Projection |
| **Frequency Embedding**| *None* | 256-D vector | 256-D vector |
| **Fusion Layer** | *None* | *None* | Concatenation $\to$ **1792-D Representation** |
| **Classification Head** | Linear(1536, 1) Logit | Linear(256 $\to$ 128 $\to$ 64 $\to$ 1) | Linear(1792 $\to$ 512 $\to$ 128 $\to$ 1) |
| **Trainable Parameters**| **11,549,993** (~11.55M) | **495,841** (~0.50M) | **12,135,689** (~12.14M) |

---

## 5. Summary Results Table

All values are drawn directly from the frozen experiment artifacts:

| Experiment | Architecture | Trainable Params | Best Epoch | Val Acc | Val ROC-AUC | Unseen Acc | Unseen ROC-AUC | Unseen PR-AUC | Unseen Macro-F1 | BigGAN ROC-AUC | Midjourney ROC-AUC | Peak VRAM | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Spatial-Only | 11.55M | Ep 9 | 91.76% | 0.9759 | **81.31%** | **0.8991** | **0.9079** | **0.7903** | **0.9732** | 0.8224 | 1521 MB | 34.59 min |
| **E2** | Frequency-Only | 0.50M | Ep 9 | 76.76% | 0.8512 | 57.75% | 0.6366 | 0.6340 | 0.4393 | 0.5955 | 0.6765 | 342 MB | 16.76 min |
| **E3** | Dual-Domain (MinMax) | 12.14M | Ep 9 | 92.82% | 0.9795 | 77.25% | 0.8851 | 0.8908 | 0.7284 | 0.9465 | 0.8228 | 1833 MB | 41.91 min |
| **E3-Std** | Dual-Domain (Std) | 12.14M | Ep 10 | **93.34%** | **0.9838** | 75.52% | 0.8959 | 0.8993 | 0.6939 | 0.9511 | **0.8392** | 1833 MB | 41.27 min |

---

## 6. Training Dynamics & In-Distribution Validation

The training progression across all 10 epochs demonstrates smooth convergence under the AdamW warmup and cosine decay schedule:

### Epoch-by-Epoch Validation Progression

```
Experiment E1 (Spatial Baseline):
  Epoch 01: Val Acc: 75.22% | Val ROC-AUC: 0.8352 | Val Loss: 0.5306
  Epoch 03: Val Acc: 86.32% | Val ROC-AUC: 0.9396 | Val Loss: 0.3182
  Epoch 05: Val Acc: 90.40% | Val ROC-AUC: 0.9667 | Val Loss: 0.2746
  Epoch 07: Val Acc: 91.96% | Val ROC-AUC: 0.9747 | Val Loss: 0.2586
  Epoch 09: Val Acc: 91.76% | Val ROC-AUC: 0.9759 | Val Loss: 0.2219  <-- Best Checkpoint
  Epoch 10: Val Acc: 91.48% | Val ROC-AUC: 0.9740 | Val Loss: 0.3082

Experiment E2 (Frequency Baseline):
  Epoch 01: Val Acc: 67.12% | Val ROC-AUC: 0.7442 | Val Loss: 0.6028
  Epoch 03: Val Acc: 69.26% | Val ROC-AUC: 0.7853 | Val Loss: 0.5793
  Epoch 06: Val Acc: 75.02% | Val ROC-AUC: 0.8246 | Val Loss: 0.5176
  Epoch 08: Val Acc: 75.82% | Val ROC-AUC: 0.8465 | Val Loss: 0.4914
  Epoch 09: Val Acc: 76.76% | Val ROC-AUC: 0.8512 | Val Loss: 0.4786  <-- Best Checkpoint
  Epoch 10: Val Acc: 76.94% | Val ROC-AUC: 0.8509 | Val Loss: 0.4746

Experiment E3 (Dual-Domain Fusion):
  Epoch 01: Val Acc: 68.28% | Val ROC-AUC: 0.8713 | Val Loss: 0.5885
  Epoch 03: Val Acc: 87.42% | Val ROC-AUC: 0.9455 | Val Loss: 0.3193
  Epoch 05: Val Acc: 89.76% | Val ROC-AUC: 0.9659 | Val Loss: 0.2501
  Epoch 08: Val Acc: 92.44% | Val ROC-AUC: 0.9766 | Val Loss: 0.1953
  Epoch 09: Val Acc: 92.82% | Val ROC-AUC: 0.9795 | Val Loss: 0.1829  <-- Best Checkpoint
  Epoch 10: Val Acc: 92.88% | Val ROC-AUC: 0.9794 | Val Loss: 0.1866
```

### In-Distribution Takeaway
On the in-distribution validation split (`ADM`, `GLIDE`, `SDv5`, `VQDM`, `Wukong`), **E3 Dual-Domain** achieved the strongest performance across all metrics:
- **Validation ROC-AUC**: **0.9795** (E3) vs. **0.9759** (E1) vs. **0.8512** (E2)
- **Validation Accuracy**: **92.82%** (E3) vs. **91.76%** (E1) vs. **76.76%** (E2)
- **Validation Loss**: **0.1829** (E3) vs. **0.2219** (E1) vs. **0.4786** (E2)

---

## 7. Unseen-Generator Test Performance

When evaluated on the 9,999 unseen holdout test images (`BigGAN` and `Midjourney`), the model rankings shifted substantially:

### Confusion Matrices ($N = 9,999$, Threshold = 0.50)

- **E1 Spatial Baseline**:
  - $TP = 3,521, \quad FP = 390, \quad TN = 4,609, \quad FN = 1,479$
  - Accuracy = **81.31%**, Precision = **90.03%**, Recall = **70.42%**, F1 = **0.7903**
- **E2 Frequency Baseline**:
  - $TP = 1,655, \quad FP = 880, \quad TN = 4,119, \quad FN = 3,345$
  - Accuracy = **57.75%**, Precision = **65.29%**, Recall = **33.10%**, F1 = **0.4393**
- **E3 Dual-Domain Fusion**:
  - $TP = 3,050, \quad FP = 325, \quad TN = 4,674, \quad FN = 1,950$
  - Accuracy = **77.25%**, Precision = **90.37%**, Recall = **61.00%**, F1 = **0.7284**

---

## 8. Generator Sub-Split Dissection

### BigGAN Sub-Split (GAN Paradigm, $N = 5,000$)
BigGAN represents a generative adversarial network architecture that produces distinctive high-frequency grid and spectral artifacts:
- **E1 (Spatial)**: Accuracy: **92.06%** | ROC-AUC: **0.9732** | PR-AUC: **0.9699** | F1: **0.9205**
- **E2 (Frequency)**: Accuracy: **57.34%** | ROC-AUC: **0.5955** | PR-AUC: **0.6311** | F1: **0.4335**
- **E3 (Dual-Domain)**: Accuracy: **85.60%** | ROC-AUC: **0.9465** | PR-AUC: **0.9448** | F1: **0.8440**

*Finding*: E1 generalized exceptionally well to BigGAN (0.9732 ROC-AUC). E3 exhibited a $-2.67\%$ drop in ROC-AUC relative to E1.

### Midjourney Sub-Split (Commercial Diffusion Paradigm, $N = 4,999$)
Midjourney represents a proprietary commercial diffusion pipeline with advanced post-processing and filtering:
- **E1 (Spatial)**: Accuracy: **70.55%** | ROC-AUC: **0.8224** | PR-AUC: **0.8249** | F1: **0.6245**
- **E2 (Frequency)**: Accuracy: **58.15%** | ROC-AUC: **0.6765** | PR-AUC: **0.6318** | F1: **0.4451**
- **E3 (Dual-Domain)**: Accuracy: **68.89%** | ROC-AUC: **0.8228** | PR-AUC: **0.8225** | F1: **0.5865**

*Finding*: Both spatial and dual-domain models experienced substantial performance degradation on Midjourney due to domain shift. E3 and E1 reached approximate parity on ROC-AUC ($0.8228$ vs. $0.8224$, $+0.04\%$ delta).

---

## 9. Generalization Gap Analysis

### Generalization Drop: In-Distribution Val $\to$ Unseen Test

| Model | Val ROC-AUC | Unseen ROC-AUC | ROC-AUC Drop ($\Delta_{\text{Gen}}$) | Val Accuracy | Unseen Accuracy | Accuracy Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 (Spatial)** | 0.9759 | 0.8991 | **-0.0768** (-7.68%) | 91.76% | 81.31% | **-10.45%** |
| **E2 (Frequency)** | 0.8512 | 0.6366 | **-0.2146** (-21.46%) | 76.76% | 57.75% | **-19.01%** |
| **E3 (Dual-Domain)**| 0.9795 | 0.8851 | **-0.0944** (-9.44%) | 92.82% | 77.25% | **-15.57%** |

### Direct Performance Differences ($\Delta = \text{E3} - \text{E1}$)
- **In-Distribution Validation ROC-AUC**: $0.9795 - 0.9759 = \mathbf{+0.0036}$ ($+0.36\%$)
- **Unseen Overall ROC-AUC**: $0.8851 - 0.8991 = \mathbf{-0.0140}$ ($-1.40\%$)
- **Unseen Overall Accuracy**: $77.25\% - 81.31\% = \mathbf{-4.06\%}$
- **Unseen Overall F1-Score**: $0.7284 - 0.7903 = \mathbf{-0.0619}$
- **BigGAN ROC-AUC**: $0.9465 - 0.9732 = \mathbf{-0.0267}$ ($-2.67\%$)
- **Midjourney ROC-AUC**: $0.8228 - 0.8224 = \mathbf{+0.0004}$ ($+0.04\%$)

### Explaining the Discrepancy
The observation that *E3 performs better than E1 on in-distribution validation, but worse than E1 on unseen-generator test data* is not a contradiction; it represents a classical **generalization gap**.

One possible explanation (to be treated as a hypothesis rather than an experimentally proven fact) is that the frequency branch learned spectral characteristics that were more specific to the training-generator distribution and therefore transferred less effectively to unseen generators:
1. *Hypothesized In-Distribution Specialization*: During training, the 2D FFT frequency branch may capture high-frequency spectral signatures characteristic of the 5 training generator families (`ADM`, `GLIDE`, `SDv5`, `VQDM`, `Wukong`), assisting in-distribution validation performance ($0.9795$ ROC-AUC).
2. *Hypothesized Out-of-Distribution Misalignment*: When exposed to completely unseen generator architectures (`BigGAN` GAN artifacts and proprietary `Midjourney` diffusion filters), these training-specific frequency patterns may not generalize identically, contributing to a larger out-of-distribution drop ($-9.44\%$ for E3 vs. $-7.68\%$ for E1).

---

## 10. Resource and Computational Efficiency Analysis

| Metric | E1: Spatial-Only | E2: Frequency-Only | E3: Dual-Domain Fusion |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 11,549,993 (~11.55M) | 495,841 (~0.50M) | 12,135,689 (~12.14M) |
| **Peak GPU VRAM (RTX 3050)** | **1,521.2 MB** | **342.3 MB** | **1,833.2 MB** |
| **Total Training Time (10 Ep)** | **34.59 min** (207.5 s/ep) | **16.76 min** (100.5 s/ep) | **41.91 min** (251.5 s/ep) |
| **VRAM Overhead vs. E1** | Baseline | -77.5% | +20.5% |
| **Compute Time Overhead vs. E1**| Baseline | -51.5% | +21.2% |

### Practical Trade-Off Discussion
- **E1 (Spatial Baseline)** provides the optimal balance of accuracy, generalization, and computational efficiency, achieving top unseen accuracy (81.31%) in ~35 minutes with modest memory consumption (1.52 GB).
- **E2 (Frequency Baseline)** is extremely lightweight (342 MB VRAM, 16.8 min), making it useful for constrained environments or fast preliminary screening, but lacks sufficient standalone discriminative power for production generalization.
- **E3 (Dual-Domain Fusion)** requires 21% more compute time and 20% more memory without translating into superior out-of-distribution generalization under the current concatenation strategy.

---

## 11. Core Scientific Finding

> **Core Research Finding**:  
> On the GenImage benchmark protocol evaluated in this study, the empirical evidence **does not support** the hypothesis that simple late concatenation of 2D FFT frequency features improves generalization to unseen AI generators beyond a strong pretrained spatial backbone alone.

### Summary of Observed Empirical Facts:
1. **Spatial Features Generalize More Robustly**: The pretrained EfficientNet-B3 backbone alone (E1) achieved the highest overall unseen ROC-AUC (**0.8991**) and accuracy (**81.31%**).
2. **Frequency Alone Provides Diagnostic Signal**: The frequency-only baseline (E2) achieved an unseen ROC-AUC of **0.6366**, demonstrating that frequency artifacts carry non-trivial synthetic signals well above chance (0.50), but are insufficient in isolation.
3. **In-Distribution Superiority vs. Out-of-Distribution Generalization Drop**: The dual-domain fusion model (E3) achieved peak performance on known validation generators (**0.9795** ROC-AUC) but underperformed E1 out-of-distribution (**0.8851** ROC-AUC), particularly on GAN-synthesized BigGAN images ($-2.67\%$).

### Hypothesized Mechanisms (To Be Tested in Future Research):
- *Hypothesis A (Spectral Idiosyncrasy)*: High-frequency spectral artifacts may be idiosyncratic to specific upsampling layers, noise schedules, and kernel sizes of individual generator architectures, causing simple frequency encoders to memorize generator-specific artifacts rather than universal synthetic invariants.
- *Hypothesis B (Fusion Dominance / Misalignment)*: Direct unweighted concatenation may permit the classification head to place high decision weight on high-frequency frequency features that are informative for training generators but misleading for unseen architectures.

---

## 12. Limitations

1. **Fusion Topology**: The study evaluated direct feature-level concatenation followed by an MLP classifier; alternative fusion mechanisms (e.g., cross-attention, gated bilinear pooling, or decision-level ensembles) were not explored in Phase 2.
2. **Frequency Normalization**: The frequency branch utilized per-image min-max log-magnitude scaling; alternative normalizations (instance standardization, radial averaging, or bandpass filtering) may yield different invariant properties.
3. **Generator Coverage**: The benchmark evaluated 5 training generators and 2 unseen holdout generators; broader generator coverage across autoregressive and masked generative models is needed for comprehensive generalization mapping.
4. **Resolution Constraints**: Images were evaluated at standard $224 \times 224$ resolution; higher native resolutions ($512 \times 512$) may preserve finer frequency grid artifacts.

---

## 13. Future Work

The following experimental directions are prioritized for future investigation (without modifying current benchmark results):

1. **Frequency Normalization & Filtering Ablation**: Evaluate bandpass filtering, radial spectral power profiling ($1\text{D}$ azimuthal averaging), and instance standardization to isolate generator-invariant frequency components.
2. **FFT vs. DCT Transform Comparison**: Implement and benchmark 2D Discrete Cosine Transform (DCT) representations, which avoid boundary leakage and spectral ringing effects inherent to non-periodic windowed FFT.
3. **Adaptive and Gated Fusion Mechanisms**: Replace static concatenation with cross-domain attention or gating mechanisms that dynamically weigh spatial vs. frequency confidence based on image content.
4. **Multi-Seed Statistical Replication**: Execute multi-seed runs ($N=5$) with bootstrap confidence intervals to establish rigorous statistical significance bounds on $\Delta(E3 - E1)$.
5. **Expanded Holdout Testing**: Test frozen checkpoints against newer generative paradigms (e.g., FLUX, Stable Diffusion XL, DALL-E 3, Sora frames).

---

## 14. Extended OOD Generalization Study

### 14.1 Motivation & Protocol Rationale
Following the initial findings where Dual-Domain E3 underperformed Spatial E1 out-of-distribution, an extended research study was conducted to determine whether **spectral normalization strategies** could improve cross-generator invariance without altering the underlying dual-domain backbone or accessing the frozen test sets.

To develop and select model candidates under strict zero-leakage conditions, we instituted a **5-Fold Leave-One-Generator-Out (LOGO) Cross-Validation Protocol** using only the 5 training generators (`ADM`, `GLIDE`, `SDv5`, `VQDM`, `Wukong`), while keeping `BigGAN` and `Midjourney` completely quarantined.

### 14.2 5-Fold LOGO Development Protocol & Ablation
Across all 5 folds, exactly one development generator was held out for validation while the remaining four formed the training set (16,000 train images, 1,000 validation images per fold; seed 42; 10 epochs):

Three frequency log-magnitude normalization transforms were ablated within the E3 architecture:
1. **`minmax` (Baseline):** Per-image linear scaling $S_{\text{norm}} = (S - \min S) / (\max S - \min S + \epsilon)$ to $[0, 1]$.
2. **`standardize` (Candidate):** Per-image Z-score standardization $S_{\text{norm}} = (S - \mu) / (\sigma + \epsilon)$ to zero mean and unit variance.
3. **`none` (Candidate):** Unnormalized log-magnitude spectrum $S_{\text{norm}} = \log(1 + |F(u, v)|)$.

### 14.3 LOGO Cross-Validation Results & Candidate Selection
Across 15 complete model training runs (3 strategies $\times$ 5 folds):

| Rank | Normalization Strategy | Mean LOGO ROC-AUC | Std ROC-AUC | Worst-Case AUC | Best-Case AUC | Mean PR-AUC | Mean Accuracy | Mean F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`standardize`** (Z-Score) | **0.8675** | **0.1032** | **0.7328** (VQDM) | **0.9634** (GLIDE) | **0.8670** | **74.18%** | **0.6408** |
| 🥈 **2** | **`none`** (Raw Log) | **0.8569** | **0.1000** | **0.7275** (VQDM) | **0.9470** (GLIDE) | **0.8483** | **73.96%** | **0.6386** |
| 🥉 **3** | **`minmax`** (Baseline) | **0.8528** | **0.1127** | **0.7020** (VQDM) | **0.9605** (GLIDE) | **0.8414** | **73.68%** | **0.6406** |

**Selection Decision:** Candidate `standardize` outperformed `minmax` on **all 5 individual folds** (+0.0215 on ADM, +0.0030 on GLIDE, +0.0146 on SDv5, +0.0308 on VQDM, +0.0034 on Wukong). Based solely on development data, `standardize` was pre-registered and selected for final unseen evaluation.

---

### 14.4 Final Unseen-Generator Evaluation (Test Split, $N=9,999$)
The finalized candidate (`E3-Std`) was trained on all 5 development generators (20,000 train, 5,000 val; reaching validation ROC-AUC **0.9838**) and evaluated once on the unseen test benchmarks (`BigGAN` and `Midjourney`):

| Model | Architecture | Freq Norm | Overall Unseen ROC-AUC | Overall Unseen PR-AUC | Overall Accuracy | Precision | Recall | F1-Score | BigGAN ROC-AUC | Midjourney ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | Spatial-Only | N/A | **0.8991** | **0.9079** | **81.31%** | 90.03% | **70.42%** | **0.7903** | **0.9732** | 0.8224 |
| **E3-Std** | Dual-Domain | `standardize` | **0.8959** | 0.8993 | 75.52% | **92.56%** | 55.50% | 0.6939 | **0.9511** | **0.8392** |
| **E3** | Dual-Domain | `minmax` | 0.8851 | 0.8908 | 77.25% | 90.37% | 61.00% | 0.7284 | 0.9465 | 0.8228 |
| **E2** | Frequency-Only | `minmax` | 0.6366 | 0.6340 | 57.75% | 65.29% | 33.10% | 0.4393 | 0.5955 | 0.6765 |

### 14.5 Comparative Analysis & Exact Deltas

#### 1. E3-Std vs Original E3 (Dual-Domain Comparison):
- **Overall ROC-AUC:** $0.8851 \to \mathbf{0.8959} \quad (\mathbf{+0.0108} \text{ or } \mathbf{+1.08\%})$
- **Overall PR-AUC:** $0.8908 \to \mathbf{0.8993} \quad (+0.0085)$
- **BigGAN ROC-AUC:** $0.9465 \to \mathbf{0.9511} \quad (+0.0046)$
- **Midjourney ROC-AUC:** $0.8228 \to \mathbf{0.8392} \quad (\mathbf{+0.0163} \text{ or } \mathbf{+1.63\%})$
- **Precision:** $90.37\% \to \mathbf{92.56\%} \quad (+2.19\%)$
- **False Positive Reduction:** $325 \to \mathbf{223} \quad (\mathbf{-31.4\% \text{ fewer false positives on real images}})$

#### 2. E3-Std vs E1 (Spatial Baseline Comparison):
- **Overall ROC-AUC:** $0.8959$ vs $\mathbf{0.8991} \quad (-0.0031 \text{ or } -0.31\%)$
- **BigGAN ROC-AUC:** $0.9511$ vs $\mathbf{0.9732} \quad (-0.0221)$
- **Midjourney ROC-AUC:** $\mathbf{0.8392}$ vs $0.8224 \quad (\mathbf{+0.0167} \text{ or } \mathbf{+1.67\% \text{ gain on commercial diffusion}})$
- **Precision:** $\mathbf{92.56\%}$ vs $90.03\% \quad (+2.53\%)$
- **False Positives:** $\mathbf{223}$ vs $390 \quad (\mathbf{-42.8\% \text{ fewer false positives}})$
- **Accuracy / Recall / F1:** At the default 0.50 threshold, E1 maintains higher recall (70.42% vs 55.50%) and accuracy (81.31% vs 75.52%).

### 14.6 Scientific Interpretation & Limitations
- **Primary Conclusion:** Frequency standardization improves dual-domain OOD generalization and narrows the gap to the spatial baseline, while improving Midjourney transfer.
- **Why Standardization Helps:** Min-Max scaling is vulnerable to extreme spectral DC/high-frequency outliers that compress subtle harmonic traces. Z-score standardization preserves relative spectral variance across frequency bands, preventing spectral saturation and enabling more invariant frequency representations.
- **Boundary of Claims:** While `E3-Std` represents the best dual-domain architecture and achieves peak performance on `Midjourney`, **Spatial-Only (E1) retains the highest overall unseen ROC-AUC (0.8991) and accuracy (81.31%)**. Therefore, E1 remains the recommended default model for general production deployment, while E3-Std provides an effective dual-domain architecture for specialized workflows where false alarms must be minimized and commercial diffusion robustness is paramount.

---

## 15. Experiment E4: Robustness Augmentation (Documented Failed Experiment)

### 15.1 Motivation & Experimental Design
Experiment E4 was formulated to investigate whether aggressive data augmentation (including simulated JPEG compression artifacts, additive Gaussian noise, and spatial blurring applied to both spatial and frequency representations) would improve model robustness against real-world degradation without compromising clean synthetic detection.

The E4 model utilized the E3-Std architecture (EfficientNet-B3 spatial backbone + standardized 4-block spectral CNN with late fusion) trained on the 5 GenImage development generators with active online robustness augmentations.

### 15.2 Empirical Evaluation Results
When evaluated on the standardized holdout benchmarks and the real-world test sets, Experiment E4 produced catastrophic performance degradation across all domains:

| Metric | E3-Std (Baseline) | E4 (Robustness Augmented) | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **In-Distribution Val ROC-AUC** | **0.9838** | 0.9570 | -0.0268 | Degraded |
| **Overall Unseen Test ROC-AUC** | **0.8959** | 0.7878 | -0.1081 (-10.8%) | Severe Failure |
| **Unseen BigGAN ROC-AUC** | **0.9511** | 0.7627 | -0.1884 (-18.8%) | Severe Failure |
| **Unseen Midjourney ROC-AUC** | **0.8392** | 0.8137 | -0.0255 | Degraded |
| **DALL-E 3 Standalone Recall ($N=492$)** | 33.94% (167/492) | 34.96% (172/492) | +1.02% | Near Zero Transfer |
| **Real-World V2 ROC-AUC ($N=76$)** | 0.4619 | **0.2590** | -0.2029 | Catastrophic Failure |
| **Real-World V2 AI Detected** | 5 / 38 | **0 / 38** (0.0% Recall) | -5 detections | Total Blindness |
| **Real-World V2 AI FNR** | 86.84% | **100.0%** | +13.16% | Complete Miss Rate |

### 15.3 Scientific Post-Mortem & Rejection
- **Spectral Artifact Destruction:** Applying artificial noise and compression during training effectively masked the subtle high-frequency harmonic grids that neural generators leave behind. The frequency branch was rendered unable to distinguish synthetic generator traces from artificial noise.
- **Decision Boundary Distortion:** Rather than learning invariant features, the model shifted its decision threshold excessively toward authentic classifications, completely failing on real-world AI images (0 / 38 detected on V2).
- **Official Status:** Experiment E4 is explicitly documented as a **FAILED EXPERIMENT**. Its checkpoints are preserved for scientific transparency but are completely excluded from deployment.

---

## 16. Experiment E5: Modern-Generator Domain Generalization (Final Production Model)

### 16.1 Motivation & Controlled External Acquisition
Following the failure of E4 and the domain limitation of classical GenImage training generators (which lacked modern diffusion transformer and flow-matching paradigms), Experiment E5 was instituted to achieve robust cross-generator domain generalization.

Under a strict zero-leakage protocol, 4,432 external images were acquired and verified:
1. **Modern AI Generators (2,000 images):** FLUX.1 [dev] (500), FLUX.1 [schnell] (500), Synthbuster SDXL (1,000).
2. **Authentic Camera Photographs (1,940 accepted):** Native and social-media transmitted smartphone images from the VISION dataset (Apple iPhone and Android/Samsung devices).
3. **Quarantined Evaluation Holdout (492 images):** Synthbuster DALL-E 3 images reserved strictly for zero-shot holdout evaluation (permanently barred from training or validation).
4. **Perceptual Leakage Gate:** A perceptual hash ($d_H \le 3$) and exact MD5 audit against the frozen V2 benchmark intercepted 7 candidate collisions, quarantining them with zero contamination.

### 16.2 Training Configuration & Convergence
- **Architecture:** `DeepVisionFusionModel` with EfficientNet-B3 spatial branch and standardized 2D FFT spectral CNN (`freq_norm_strategy="standardize"`, late fusion).
- **Dataset:** Unified manifest (`data/e5_external/manifests/e5_manifest.csv`) with 23,165 training images and 5,775 validation images.
- **Outcome:** Smooth convergence reaching **0.9820 Validation ROC-AUC** at Epoch 10 (`experiments/e5_generalization/best_model.pt`).

### 16.3 Final Unseen Benchmark Performance ($N = 9,999$, Holdout Test)

| Model | Architecture | Overall Unseen ROC-AUC | Overall Unseen PR-AUC | Overall Accuracy | BigGAN ROC-AUC | Midjourney ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E5 (Generalization)** | Dual-Domain (Std) | **0.9195** | **0.9227** | 78.93% | **0.9517** | **0.8869** |
| **E1 (Spatial Baseline)** | Spatial-Only | 0.8991 | 0.9079 | **81.31%** | **0.9732** | 0.8224 |
| **E3-Std (Standardized)** | Dual-Domain (Std) | 0.8959 | 0.8993 | 75.52% | 0.9511 | 0.8392 |
| **E3 (Baseline MinMax)** | Dual-Domain (MinMax) | 0.8851 | 0.8908 | 77.25% | 0.9465 | 0.8228 |
| **E4 (Robustness)** | Dual-Domain (Aug) | 0.7878 | 0.7824 | 64.33% | 0.7627 | 0.8137 |

*Key Findings:*
- **Overall Unseen ROC-AUC** reached **0.9195**, exceeding all prior models including E1 Spatial Baseline (0.8991).
- **Midjourney Generalization** surged to **0.8869 ROC-AUC**, representing a $+6.45\%$ improvement over E1 (0.8224) and $+4.77\%$ over E3-Std (0.8392).
- **BigGAN Generalization** remained exceptionally strong at **0.9517 ROC-AUC**.

### 16.4 DALL-E 3 Standalone Zero-Shot Holdout Evaluation ($N = 492$)
DALL-E 3 images were evaluated strictly out-of-distribution with zero prior model exposure:

| Evaluation Dimension | Metric | Comparative Context |
| :--- | :---: | :--- |
| **Total Test Samples ($N$)** | **492** | Synthbuster DALL-E 3 partition |
| **ROC-AUC (vs. Unseen Real)** | **0.9883** | vs. E3-Std 0.8158, E1 0.7665, E4 0.7780 |
| **PR-AUC (vs. Unseen Real)** | **0.9382** | vs. E3-Std 0.3634, E1 0.2608 |
| **Detection Rate (Recall @ 0.50)** | **93.29%** | **459 / 492 AI Images Detected** |
| **Historical Comparison** | +59.35% | E3-Std detected only 167/492 (33.94%) |
| **Mean AI Probability** | **0.9141** | High confidence synthetic separation |
| **Median AI Probability** | **0.9946** | Strong clustering near 1.0 |

---

## 17. Final Frozen Real-World Benchmark: Dataset V2 ($N = 76$)

The Real-World Test Dataset V2 ($N=76$: 38 authentic smartphone photographs, 38 modern AI-generated images from Gemini, ChatGPT-4o, and Midjourney v6) remained permanently frozen and unaccessed during training.

### Comparative Real-World Benchmark Table (Threshold = 0.50)

| Metric | E1 (Spatial) | E3-Std (Dual) | E4 (Robustness) | E5 (Generalization) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Samples ($N$)** | 76 | 76 | 76 | **76** |
| **Authentic Real Photos** | 38 | 38 | 38 | **38** |
| **Synthetic AI Images** | 38 | 38 | 38 | **38** |
| **ROC-AUC** | 0.2621 | 0.4619 | 0.2590 | **0.7715** |
| **PR-AUC** | 0.3541 | 0.4578 | 0.3600 | **0.8284** |
| **Accuracy** | 47.37% | 48.68% | 40.79% | **76.32%** |
| **Precision** | 33.33% | 40.00% | 0.00% | **88.46%** |
| **Recall (AI Sensitivity)** | 5.26% | 5.26% | 0.00% | **60.53%** |
| **F1-Score** | 0.0909 | 0.0930 | 0.0000 | **0.7188** |
| **Real False Positive Rate (FPR)** | 10.53% (4/38) | 7.89% (3/38) | 18.42% (7/38) | **7.89%** (3/38) |
| **AI False Negative Rate (FNR)** | 94.74% (36/38) | 94.74% (36/38) | 100.0% (38/38) | **39.47%** (15/38) |
| **AI Detected Count** | 2 / 38 | 2 / 38 | 0 / 38 | **23 / 38** |

### Confusion Matrix ($N = 76$, Threshold = 0.50)
```
                  Predicted Real (0)    Predicted AI (1)
Actual Real (0)        35 (TN)               3 (FP)        [FPR = 7.89%]
Actual AI   (1)        15 (FN)              23 (TP)        [Recall = 60.53%]
```

### Scientific Takeaway on Real-World Generalization
1. **Breakthrough Real-World AI Detection:** Prior models (E1, E3, E3-Std) detected at most 2 out of 38 real-world AI images (failing catastrophically due to generator domain shift). E5 detects **23 out of 38** real-world AI images (**60.53% recall**), elevating ROC-AUC from $0.4619 \to \mathbf{0.7715}$.
2. **Preservation of Low False Alarm Rate:** Despite a massive increase in sensitivity, E5 maintains a strict **7.89% false positive rate** on authentic smartphone photos (35/38 correctly classified as real), resulting in **88.46% precision**.
3. **Production Model Recommendation:** Experiment E5 is officially designated as the **primary production model** for DeepVision-Forensics.

---

## 18. Forensic Explainability Standards & Disclaimers

### 18.1 Model Attention (Grad-CAM)
- **Definition:** Grad-CAM calculates gradient-weighted class activation heatmaps showing which spatial features and regions within the image contributed most strongly to the model's authenticity decision.
- **Scientific Constraint:** Grad-CAM reflects internal **model attention patterns**. It is **NOT** a pixel-level forgery mask, tamper localization map, or edit boundary delineator. Claims of exact manipulation localization from Grad-CAM are technically unfounded and strictly disclaimed.

### 18.2 Frequency Evidence (2D FFT Log-Magnitude)
- **Definition:** Centered 2D Fast Fourier Transform log-magnitude spectrum $\log(1 + |F(u, v)|)$ visualizes spatial frequency distributions across radial and angular bands.
- **Forensic Role:** Serves as complementary frequency evidence to reveal periodic sampling artifacts, checkerboard traces, and upsampling grid signatures characteristic of generative neural synthesis.

---

*Report compiled from frozen artifacts in `experiments/e1_spatial/`, `experiments/e2_frequency/`, `experiments/e3_dual_domain/`, `experiments/candidate_standardize/`, `experiments/e4_robustness/`, and `experiments/e5_generalization/`.*

