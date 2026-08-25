# DeepVision Forensics - API Design

## Overview
The backend is powered by FastAPI, providing RESTful endpoints for the React frontend to upload images, retrieve forensic analyses, and access system status.

## Base URL
`/api/v1`

## Endpoints

### 1. `POST /analyze`
Uploads a single image for forensic analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (The image file: JPG, PNG)

**Response (200 OK):**
```json
{
  "id": "uuid-string",
  "status": "success",
  "analysis": {
    "classification": "AI-generated",
    "ai_probability": 0.98,
    "authenticity_assessment": "The image exhibits strong spectral anomalies and synthetic spatial textures typical of diffusion models.",
    "risk_indicator": "HIGH",
    "evidence": {
      "spatial_summary": "High activation in background textures and unnatural edge blending.",
      "frequency_summary": "Checkerboard artifacts detected in high-frequency spectrum."
    },
    "visualizations": {
      "gradcam_heatmap": "data:image/png;base64,..."
    }
  }
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Invalid file format. Please upload a valid image (JPEG/PNG)."
}
```

### 2. `GET /health`
Checks API and model loading status.

**Response (200 OK):**
```json
{
  "status": "online",
  "model_loaded": true,
  "device": "cuda"
}
```

## Security Risk Indicator Rules
Risk thresholds are TBD and will be determined using validation/evaluation results and documented as application-level rules. Do not hard-code arbitrary thresholds at this stage.

## Future Endpoints (Database Integration)
- `GET /history`: Fetch past analyses.
- `GET /report/{id}`: Fetch a specific forensic report.

## Implementation Note
This API design is an interface contract only. Do not create fake ML outputs, fake Grad-CAM images, fake probabilities or fabricated forensic evidence merely to make the API functional. The real `/analyze` implementation will be completed after the ML pipeline is validated.
