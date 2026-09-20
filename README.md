# DeepVision-Forensics 🔬🔍

**Dual-Domain Spatial and Frequency Analysis for Generalized AI-Generated Image Detection**

DeepVision-Forensics is an end-to-end research and demonstration platform for detecting AI-synthesized imagery across modern and classical generative architectures. The system fuses deep spatial visual representations (via EfficientNet-B3) with 2D Fast Fourier Transform (FFT) log-magnitude spectral features to identify both perceptual anomalies and imperceptible high-frequency periodic synthesis artifacts.

---

## 🚀 Key Experimental Findings & Highlights

### 1. Final Production Model: Experiment E5 (Modern-Generator Generalization)
Experiment E5 incorporates modern generative models (FLUX.1 [dev], FLUX.1 [schnell], Synthbuster SDXL) and verified smartphone camera photographs (VISION dataset) with zero test-set leakage, achieving breakthrough cross-domain generalization:

- **Overall Unseen GenImage ROC-AUC ($N=9,999$):** **0.9195**
  - **BigGAN ROC-AUC:** **0.9517**
  - **Midjourney ROC-AUC:** **0.8869**
- **DALL-E 3 Standalone Zero-Shot Holdout ($N=492$):**
  - **ROC-AUC:** **0.9883**
  - **Recall (@ 0.50 Threshold):** **93.29%** (459 / 492 detected AI vs. 33.94% for E3-Std)
  - **PR-AUC:** **0.9382**
- **Frozen Real-World Benchmark Dataset V2 ($N=76$, 38 Real / 38 AI):**
  - **ROC-AUC:** **0.7715** (vs. E3-Std 0.4619, E1 0.2621, E4 0.2590)
  - **Accuracy:** **76.32%**
  - **Precision:** **88.46%**
  - **Recall (Sensitivity):** **60.53%** (23 / 38 AI detected vs. 2 / 38 for prior models)
  - **F1-Score:** **0.7188**
  - **Real-Photo False Positive Rate (FPR):** **7.89%** (only 3 false alarms out of 38)
  - **AI Miss Rate (FNR):** **39.47%** (down from 94.74%)
  - **AI Detected Count:** **23 / 38**

---

### 2. Experiment E4: Robustness Augmentation (Documented Failed Experiment)
To maintain complete scientific integrity, **Experiment E4 is explicitly documented as a failed robustness experiment**:
- **Design:** Online aggressive augmentation (JPEG compression, Gaussian noise, blur) applied during dual-domain training.
- **Outcome:** Catastrophic failure across all test benchmarks:
  - In-Distribution Val ROC-AUC: `0.9570` (vs. E3-Std `0.9838`)
  - Overall Unseen Holdout ROC-AUC: `0.7878` (vs. E3-Std `0.8959`, E1 `0.8991`)
  - BigGAN ROC-AUC: `0.7627` (vs. E3-Std `0.9511`)
  - Midjourney ROC-AUC: `0.8137` (vs. E3-Std `0.8392`)
  - Real-World Test V2 ROC-AUC: **`0.2590`** with **0 / 38 AI detected** (0.0% recall, 100% false negative rate).
- **Post-Mortem:** Aggressive distortions destroyed subtle high-frequency grid artifacts, corrupting the spectral encoder into acting as noise and excessively biasing the classifier toward predicting "real". E4 was officially classified as **FAILED** and barred from deployment.

---

## 📊 Benchmark Summary Table

| Model | Architecture | Freq Strategy | Unseen ROC-AUC | BigGAN AUC | Midjourney AUC | DALL-E 3 Recall | V2 Real-World AUC | V2 AI Detected |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E5 (Generalization)** ⭐ | Dual-Domain | `standardize` | **0.9195** | **0.9517** | **0.8869** | **93.29%** | **0.7715** | **23 / 38** |
| **E3-Std (Standardized)** | Dual-Domain | `standardize` | 0.8959 | 0.9511 | 0.8392 | 33.94% | 0.4619 | 2 / 38 |
| **E1 (Spatial Baseline)** | Spatial-Only | N/A | 0.8991 | 0.9732 | 0.8224 | 35.57% | 0.2621 | 2 / 38 |
| **E3 (Baseline MinMax)** | Dual-Domain | `minmax` | 0.8851 | 0.9465 | 0.8228 | 42.48% | 0.4287 | 3 / 38 |
| **E4 (Robustness — FAILED)** | Dual-Domain | `standardize` | 0.7878 | 0.7627 | 0.8137 | 34.96% | 0.2590 | 0 / 38 |
| **E2 (Frequency Baseline)** | Freq-Only | `minmax` | 0.6366 | 0.5955 | 0.6765 | N/A | N/A | N/A |

*All evaluations were conducted at the frozen decision threshold of 0.50 with zero post-hoc tuning.*

---

## 🧠 Explainability & Interpretability Standards

The DeepVision-Forensics interface provides visual forensic evidence alongside every probabilistic prediction:

1. **Model Attention (Grad-CAM):**
   - Visualizes gradient-weighted class activation patterns across the spatial backbone to highlight regions contributing most heavily to the classification score.
   - ⚠️ **Forensic Disclaimer:** Heatmaps reflect **internal neural network attention** patterns. They do **NOT** represent pixel-level manipulation masks, forgery boundaries, or exact edit localization.

2. **Frequency Evidence (2D FFT Log-Magnitude):**
   - Centered 2D Fast Fourier Transform log-magnitude spectrum $\log(1 + |F(u, v)|)$ visualizes 2D frequency distributions to reveal periodic synthesis artifacts, checkerboard traces, and high-frequency grid irregularities.

---

## 🏛️ System Architecture & Multi-Model Selection

The application is structured into a modular Python/FastAPI backend and a responsive Vanilla CSS/JS frontend:

```
DeepVision-Forensics/
├── backend/
│   ├── config.py             # Global constants & SUPPORTED_MODELS catalog
│   ├── inference.py          # ModelService singleton, Grad-CAM & FFT extraction
│   ├── main.py               # FastAPI REST endpoints & frontend static mount
│   └── schemas.py            # Pydantic request/response schemas
├── frontend/
│   ├── index.html            # Web UI with model selector & visual evidence tabs
│   ├── app.js                # Frontend API client & forensic visualization renderer
│   └── styles.css            # Modern dark-mode forensic styling
├── models/
│   └── fusion.py             # Spatial, Frequency, and Dual-Domain PyTorch architectures
├── experiments/
│   ├── e1_spatial/           # E1 Spatial Baseline checkpoint
│   ├── candidate_standardize/# E3-Std checkpoint
│   ├── e4_robustness/        # E4 Robustness checkpoint (documented failed)
│   └── e5_generalization/    # E5 Generalization checkpoint (production)
└── tests/                    # Comprehensive unit and integration test suite
```

### Supported Runtime Models:
- **`e5_generalization` (Default):** DeepVision-E5 Generalization (Dual-Domain Spatial + Frequency Fusion, 12.14M parameters).
- **`e3_std`:** DeepVision-E3-Std Dual-Domain (GenImage Standardized Fusion, 12.14M parameters).
- **`e1_spatial`:** DeepVision-E1 Spatial Baseline (EfficientNet-B3, 11.55M parameters).

---

## 🛠️ Quickstart & Local Demo

### 1. Environment Setup
```powershell
# Activate Python 3.11 virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
python -m pytest -v
```

### 3. Launch Demo Application
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser:
1. Select active model (defaults to **DeepVision-E5 Generalization**).
2. Click **Load Test Sample Image** or upload an image (JPEG, PNG, WEBP).
3. Click **Analyze Image** to view the probabilistic prediction, **Model Attention (Grad-CAM)** overlay, and **Frequency Evidence (2D FFT)** spectrum.

---

## ⚖️ Legal & Forensic Research Disclaimer
DeepVision-Forensics is a research platform that outputs probabilistic indicators of synthetic generation. The predictions and visual attention maps provided by this system should not be treated as definitive or legal proof in critical decision-making or judicial contexts.
