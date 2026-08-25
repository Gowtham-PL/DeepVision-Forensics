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
  - GenImage (primary).
  - CIFAKE (additional benchmarking).
  - Reserve at least one generator (e.g., specific Diffusion Model) exclusively for testing cross-generator generalization.
- **Transformations (Experimental):**
  - JPEG compression (various quality levels).
  - Resizing/Scaling.
  - Gaussian Blur.
  - Noise addition.
- **Data Splits:** Strict 70% Train, 15% Validation, 15% Test separation. No data leakage between splits.

## 8. Output to Backend
The ML pipeline inference script will return:
1. `classification_label`: "Real" or "AI-generated"
2. `ai_probability`: Float between 0.0 and 1.0 (Model-estimated probability/confidence that the image is AI-generated).
3. `spatial_evidence_summary`: Textual or structured summary of spatial activations.
4. `frequency_evidence_summary`: Textual or structured summary of frequency artifacts.
5. `gradcam_heatmap`: Base64 encoded image or raw numpy array of the generated heatmap.
