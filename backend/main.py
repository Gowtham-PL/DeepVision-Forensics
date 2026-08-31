"""
FastAPI application for DeepVision-Forensics.

Provides RESTful endpoints:
- GET /api/v1/health: Health check, device metadata, and model status.
- POST /api/v1/analyze: Multipart image upload, forensic inference, Grad-CAM, and FFT spectrum.
"""

from contextlib import asynccontextmanager
from typing import Optional
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.inference import model_service
from backend.schemas import AnalyzeResponse, ErrorResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model once at application startup."""
    try:
        model_service.load_model()
        print(
            f"[*] DeepVision-Forensics model loaded successfully on "
            f"{model_service.device} ({model_service.param_count:,} parameters)."
        )
    except Exception as exc:
        print(f"[!] Warning: Model failed to load during startup: {exc}")
    yield


app = FastAPI(
    title="DeepVision-Forensics API",
    description="Probabilistic AI-Generated Image Forensic Detection & Explainability Suite",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    f"{config.API_V1_PREFIX}/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and active model status",
)
async def health_check() -> HealthResponse:
    """Returns the operational status, compute device, and model availability."""
    gpu_name: Optional[str] = None
    if torch.cuda.is_available() and model_service.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(model_service.device)

    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "degraded",
        model_loaded=model_service.is_loaded(),
        model_name=config.MODEL_NAME,
        device=str(model_service.device),
        gpu_name=gpu_name,
    )


@app.post(
    f"{config.API_V1_PREFIX}/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image payload or format"},
        413: {"model": ErrorResponse, "description": "Payload exceeds maximum allowed size"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
    tags=["Forensics"],
    summary="Analyze an uploaded image for AI generation artifacts",
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, WEBP up to 10MB)"),
    include_fft: bool = Form(True, description="Whether to generate 2D FFT log-magnitude spectrum"),
) -> AnalyzeResponse:
    """
    Accepts an uploaded image file, verifies payload integrity, executes E1 SpatialClassifier
    inference, derives probabilistic AI classification, generates Grad-CAM spatial heatmap overlay,
    and optionally extracts a diagnostic 2D FFT frequency spectrum.
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or currently initializing. Please try again shortly.",
        )

    # Read uploaded bytes
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {str(exc)}",
        )

    # Validate image payload
    try:
        image = model_service.validate_and_decode_image(contents)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # Perform forensic inference and explainability extraction
    try:
        result = model_service.predict_and_analyze(image, include_fft=include_fft)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(exc)}",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail},
    )


# Mount static frontend application
frontend_dir = config.PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

