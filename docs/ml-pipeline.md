# DeepVision Forensics - ML Pipeline

## 1. Input and Preprocessing
- **Input:** Single image (RGB).
- **Preprocessing:** 
  - Standard resize (e.g., 224x224 or 300x300 based on EfficientNet requirements).
  - Normalization using ImageNet statistics for the spatial branch.
  - Grayscale conversion and specific dimension sizing for the frequency branch if necessary.

## 2. Spatial Branch
- **Core Model:** EfficientNet-B3 (pretrained on ImageNet, fine-tuned).
- **Function:** Extracts deep visual and spatial features from the image.
- **Output:** A dense feature vector representing spatial evidence.

## 3. Frequency Branch
- **Core Technique:** 2D Fast Fourier Transform (FFT) or Discrete Cosine Transform (DCT).
- **Function:** Extracts spectral magnitude and phase information to identify high-frequency artifacts and regularities typical of generative models (e.g., checkerboard artifacts).
- **Output:** The exact FFT/DCT representation, preprocessing, frequency encoder architecture and embedding dimensionality are open technical decisions and will be finalized during Phase 3 based on experiments. Do not prematurely hard-code a large flattened frequency vector directly into the fusion layer.

## 4. Feature Fusion
- **Mechanism:** Concatenation of the spatial and frequency feature vectors.
- **Processing:** Passed through subsequent fully connected (linear) layers with Dropout and ReLU activations to learn the interaction between visual and spectral artifacts.

## 5. Classification
- **Output Layer:** A final linear layer with a Sigmoid (for binary classification) or Softmax activation.
- **Prediction:** Outputs a single continuous value representing the "AI Probability" (Model-estimated probability/confidence that the image is AI-generated).

## 6. Explainability (Grad-CAM)
- **Mechanism:** Gradient-weighted Class Activation Mapping (Grad-CAM).
- **Target:** The final convolutional layer of the EfficientNet-B3 model.
- **Output:** A heatmap highlighting the spatial regions that most strongly influenced the model's prediction.

## 7. Robustness and Evaluation Strategy
- **Datasets:** 
  - GenImage (primary): 7 generators × 5,000 images = 35,000 total.
  - CIFAKE (additional benchmarking, Phase 3+).
- **Generator-level holdout split** (implemented in `manifest.csv` — no physical file copies):

  | Logical split | Generators | Source dirs | Images |
  |---|---|---|---:|
  | `train` | ADM, VQDM, SDv5, Wukong, GLIDE | `*/train/ai` + `*/train/nature` | 20,000 |
  | `val`   | ADM, VQDM, SDv5, Wukong, GLIDE | `*/val/ai`   + `*/val/nature`   |  5,000 |
  | `test`  | **BigGAN, Midjourney** (held-out) | all of `*/train/*` + `*/val/*` | 10,000 |

  BigGAN and Midjourney are **completely invisible** during training and validation.  
  The test set measures **cross-generator generalization** — whether the model detects  
  AI-generated images from generators it has never encountered before.

- **Label encoding:** `nature` → 0 (real photographs), `ai` → 1 (generated).
- **Cross-generator duplicate:** One nature-class image pair (ADM/train ≡ Midjourney/train)  
  is flagged `is_cross_gen_dup=True` in the manifest. The Midjourney copy should be  
  excluded from test evaluation metrics to avoid a 1-image train/test overlap.
- **Transformations (Phase 3+):**
  - JPEG compression (various quality levels).
  - Resizing/Scaling.
  - Gaussian Blur.
  - Noise addition.

## 8. Output to Backend
The ML pipeline inference script will return:
1. `classification_label`: "Real" or "AI-generated"
2. `ai_probability`: Float between 0.0 and 1.0 (Model-estimated probability/confidence that the image is AI-generated).
3. `spatial_evidence_summary`: Textual or structured summary of spatial activations.
4. `frequency_evidence_summary`: Textual or structured summary of frequency artifacts.
5. `gradcam_heatmap`: Base64 encoded image or raw numpy array of the generated heatmap.
