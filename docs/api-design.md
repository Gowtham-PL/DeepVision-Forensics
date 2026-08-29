# DeepVision Forensics - API Design

## Overview
The backend is powered by FastAPI, providing RESTful endpoints for the frontend to upload images, retrieve probabilistic forensic analyses, inspect Grad-CAM spatial heatmaps, and extract diagnostic 2D FFT spectra.

## Base URL
`/api/v1`

## Endpoints

### 1. `POST /api/v1/analyze`
Uploads a single image for forensic analysis using the deployed E1 SpatialClassifier.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file (`PNG`, `JPEG`, `WEBP` up to 10MB)
  - `include_fft`: (Optional boolean, default `true`) Whether to generate 2D FFT log-magnitude spectrum visualization.

**Response (200 OK):**
```json
{
  "status": "success",
  "model_info": {
    "name": "DeepVision-E1-Spatial",
    "backbone": "EfficientNet-B3",
    "parameters": 11549993,
    "device": "cuda"
  },
  "prediction": {
    "classification_label": "AI-generated",
    "ai_probability": 0.9142,
    "authenticity_assessment": "Model estimate: 91.4% probability of AI generation.",
    "risk_indicator": "HIGH",
    "threshold_used": 0.50
  },
  "evidence": {
    "spatial_summary": "Grad-CAM visualization showing spatial regions receiving stronger model attention.",
    "frequency_summary": "Diagnostic 2D Fast Fourier Transform (FFT) log-magnitude spectrum showing spectral energy distribution."
  },
  "visualizations": {
    "gradcam_heatmap": "data:image/png;base64,iVBORw0KGgo...",
    "fft_spectrum": "data:image/png;base64,iVBORw0KGgo..."
  },
  "disclaimer": "This analysis provides probabilistic forensic indicators for research and screening purposes, not definitive proof."
}
```

**Response (400 Bad Request):**
```json
{
  "status": "error",
  "detail": "Unsupported image format 'GIF'. Allowed formats: PNG, JPEG, WEBP."
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "error",
  "detail": "Model is not loaded or currently initializing. Please try again shortly."
}
```

---

### 2. `GET /api/v1/health`
Checks API and model loading status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "DeepVision-E1-Spatial",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"
}
```

## Security Risk Indicator Rules
- `ai_probability >= 0.70`: **HIGH**
- `0.30 <= ai_probability < 0.70`: **MEDIUM**
- `ai_probability < 0.30`: **LOW**
