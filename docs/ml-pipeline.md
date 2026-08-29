# DeepVision Forensics - ML Pipeline

## 1. Input and Preprocessing
- **Input:** Single image (RGB).
- **Preprocessing:** 
  - Uniform resize to 224x224 (300x300 is a future controlled scaling experiment).
  - Both branches receive the exact same resized `[0, 1]` tensor.
  - **Spatial Branch:** Receives ImageNet-normalized input (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - **Frequency Branch:** Receives the **unnormalized `[0, 1]` tensor** (must NOT receive ImageNet-normalized values).

## 2. Spatial Branch
- **Core Model:** EfficientNet-B3 (pretrained on ImageNet, fine-tuned).
- **Function:** Extracts deep visual and spatial features from the image.
- **Output:** Global Average Pooling produces a **1536-dimensional spatial embedding**.

## 3. Frequency Branch
- **Core Technique:** 2D Fast Fourier Transform (FFT) as the primary Phase 2 transform (2D Discrete Cosine Transform / DCT is preserved for future experimental comparison).
- **Pipeline:** Unnormalized `[0, 1]` RGB → Luminance conversion ($Y = 0.299R + 0.587G + 0.114B$) → 2D FFT (`torch.fft.fft2`) → Magnitude Spectrum ($|F(u,v)|$) → `fftshift` (DC centered) → $\log(1 + |F(u,v)|)$ → Configurable Frequency Normalization (e.g. min-max, standardization, batch/instance norm) → Lightweight 4-block CNN → AdaptiveAvgPool2d((1,1)) → Linear projection → **256-dimensional frequency embedding**.

## 4. Feature Fusion
- **Mechanism:** Direct concatenation of the 1536-D spatial embedding and 256-D frequency embedding into a **1792-dimensional fused representation**.
- **Classification Head:** 
  - `Linear(1792, 512)` → `BatchNorm1d` (or `LayerNorm`) → `ReLU` → `Dropout(0.4)`
  - `Linear(512, 128)` → `ReLU` → `Dropout(0.2)`
  - `Linear(128, 1)` → Single binary logit.

## 5. Classification & Loss
- **Loss Function:** `nn.BCEWithLogitsLoss()` on the unscaled binary logit.
- **Inference:** Logit $\to$ Sigmoid $\to$ continuous "AI Probability" $\in [0.0, 1.0]$.

## 6. Explainability (Grad-CAM & Frequency Visualization)
- **Grad-CAM Mechanism:** Gradient-weighted Class Activation Mapping applied to the final convolutional layer of the EfficientNet-B3 spatial backbone (`features[-1]`).
- **Scope:** Explains spatial activations and visual regions driving the prediction. Spatial Grad-CAM does not explain frequency evidence.
- **Frequency Explainability:** The centered 2D log-magnitude spectrum and spectral saliency maps are provided separately to visualize frequency-domain energy peaks.

## 7. Training Strategy
- **Optimizer:** `AdamW` with weight decay (~`1e-4`).
- **Differential Learning Rates:** Spatial backbone initial LR ~`1e-5`, newly initialized layers ~`5e-4`.
- **Learning Rate Schedule:** Linear warmup followed by cosine decay.
- **Precision:** Mixed precision with `torch.amp.autocast('cuda')` and `GradScaler`.
- **Batch Size:** Configurable; conservative defaults (8 or 16) suitable for 4GB NVIDIA RTX 3050 VRAM.
- **Validation & Checkpoint Selection:** Checkpoint selection is strictly driven by validation performance (validation ROC-AUC) on the validation split of the 5 training generators. The test split is **never** used for checkpoint selection.

## 8. Robustness and Evaluation Strategy
- **Datasets:** 
  - GenImage (primary): 7 generators × 5,000 images = 35,000 total.
  - CIFAKE (additional benchmarking, Phase 3+).
- **Generator-level holdout split** (implemented in `manifest.csv` — no physical file copies):

  | Logical split | Generators | Source dirs | Images |
  |---|---|---|---:|
  | `train` | ADM, VQDM, SDv5, Wukong, GLIDE | `*/train/ai` + `*/train/nature` | 20,000 |
  | `val`   | ADM, VQDM, SDv5, Wukong, GLIDE | `*/val/ai`   + `*/val/nature`   |  5,000 |
  | `test`  | **BigGAN, Midjourney** (held-out) | all of `*/train/*` + `*/val/*` | 10,000 |

  BigGAN (unseen GAN generator) and Midjourney (unseen commercial image generator) are **completely invisible** during training and validation.  
  The test set measures **cross-generator generalization** — whether the model detects  
  AI-generated images from generators it has never encountered before.

- **Label encoding:** `nature` → 0 (real photographs), `ai` → 1 (generated).
- **Cross-generator duplicate:** One nature-class image pair (ADM/train ≡ Midjourney/train)  
  is flagged `is_cross_gen_dup=True` in the manifest. The Midjourney copy must be  
  excluded from test evaluation metrics to maintain zero train/test overlap.
- **Transformations (Phase 3+):**
  - JPEG compression (various quality levels).
  - Resizing/Scaling.
  - Gaussian Blur.
  - Noise addition.

## 9. Output to Backend
The ML pipeline inference script will return:
1. `classification_label`: "Real" or "AI-generated"
2. `ai_probability`: Float between 0.0 and 1.0 (Model-estimated probability/confidence that the image is AI-generated).
3. `spatial_evidence_summary`: Textual or structured summary of spatial activations.
4. `frequency_evidence_summary`: Textual or structured summary of frequency artifacts.
5. `gradcam_heatmap`: Base64 encoded image or raw numpy array of the generated heatmap.
