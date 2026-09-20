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
from backend.schemas import (
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    ModelsListResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads default model at startup and preloads other supported models if available."""
    try:
        model_service.load_model(config.DEFAULT_MODEL_KEY)
        print(
            f"[*] DeepVision-Forensics default model ({config.DEFAULT_MODEL_KEY}) loaded successfully on "
            f"{model_service.device} ({model_service.param_count:,} parameters)."
        )
        # Attempt preloading other supported models if checkpoints are present
        for preload_key, m_cfg in config.SUPPORTED_MODELS.items():
            if preload_key == config.DEFAULT_MODEL_KEY:
                continue
            if m_cfg["checkpoint_path"].exists():
                try:
                    model_service.load_model(preload_key)
                    print(
                        f"[*] DeepVision-Forensics {preload_key} model preloaded successfully on "
                        f"{model_service.device} ({model_service.model_params.get(preload_key, 0):,} parameters)."
                    )
                except Exception as e_pre:
                    print(f"[!] Note: {preload_key} preload deferred: {e_pre}")
    except Exception as exc:
        print(f"[!] Warning: Default model failed to load during startup: {exc}")
    yield


app = FastAPI(
    title="DeepVision-Forensics API",
    description="Probabilistic AI-Generated Image Forensic Detection & Explainability Suite",
    version="1.1.0",
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
    """Returns operational status, compute device, and available models."""
    gpu_name: Optional[str] = None
    if torch.cuda.is_available() and model_service.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(model_service.device)

    active_info = model_service.get_model_info()

    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "degraded",
        model_loaded=model_service.is_loaded(),
        model_name=active_info.name,
        active_model=model_service.active_model_key,
        available_models=list(config.SUPPORTED_MODELS.keys()),
        device=str(model_service.device),
        gpu_name=gpu_name,
    )


@app.get(
    f"{config.API_V1_PREFIX}/models",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List available forensic detection models",
)
async def list_models() -> ModelsListResponse:
    """Returns list of selectable model architectures and currently active default."""
    return model_service.list_available_models()


@app.post(
    f"{config.API_V1_PREFIX}/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image payload, format, or model key"},
        413: {"model": ErrorResponse, "description": "Payload exceeds maximum allowed size"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
    tags=["Forensics"],
    summary="Analyze an uploaded image for AI generation artifacts",
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, WEBP up to 10MB)"),
    include_fft: bool = Form(True, description="Whether to generate 2D FFT log-magnitude spectrum"),
    model: Optional[str] = Form(
        None,
        description="Model identifier ('e1_spatial' or 'e3_std'). Defaults to currently active model.",
    ),
) -> AnalyzeResponse:
    """
    Accepts an uploaded image file, verifies payload integrity, executes selected
    forensic model inference (E1 Spatial or E3-Std Dual-Domain), derives probabilistic
    AI classification, generates Grad-CAM spatial heatmap overlay, and optionally
    extracts a diagnostic 2D FFT frequency spectrum.
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not loaded or currently initializing. Please try again shortly.",
        )

    # Validate model selection if explicitly passed
    norm_model_key = model_service.normalize_model_key(model) if model else model_service.active_model_key
    if norm_model_key not in config.SUPPORTED_MODELS:
        supported = list(config.SUPPORTED_MODELS.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{model}'. Supported models: {supported}",
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
        result = model_service.predict_and_analyze(
            image=image,
            include_fft=include_fft,
            model_key=norm_model_key,
        )
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

