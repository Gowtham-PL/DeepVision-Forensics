"""
Pydantic response and request schemas for DeepVision-Forensics API.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check and loaded model status."""
    status: str = Field(..., description="System operational status", examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether the model is loaded in memory", examples=[True])
    model_name: str = Field(..., description="Active model architecture name", examples=["DeepVision-E1-Spatial"])
    device: str = Field(..., description="Compute device being utilized", examples=["cuda"])
    gpu_name: Optional[str] = Field(None, description="Physical GPU name if CUDA is enabled", examples=["NVIDIA GeForce RTX 3050 Laptop GPU"])


class ModelInfo(BaseModel):
    """Metadata regarding the deployed inference model."""
    name: str = Field(..., examples=["DeepVision-E1-Spatial"])
    backbone: str = Field(..., examples=["EfficientNet-B3"])
    parameters: int = Field(..., examples=[11549993])
    device: str = Field(..., examples=["cuda"])


class PredictionResult(BaseModel):
    """Probabilistic prediction outputs."""
    classification_label: str = Field(..., examples=["AI-generated"])
    ai_probability: float = Field(..., ge=0.0, le=1.0, examples=[0.9142])
    authenticity_assessment: str = Field(
        ..., examples=["Model estimate: 91.4% probability of AI generation."]
    )
    risk_indicator: str = Field(..., examples=["HIGH"])
    threshold_used: float = Field(..., examples=[0.50])


class EvidenceSummary(BaseModel):
    """Textual descriptions of forensic signals and visual regions."""
    spatial_summary: str = Field(
        ...,
        examples=["Grad-CAM visualization showing spatial regions receiving stronger model attention."],
    )
    frequency_summary: Optional[str] = Field(
        None,
        examples=["Diagnostic 2D Fast Fourier Transform (FFT) log-magnitude spectrum."],
    )


class Visualizations(BaseModel):
    """In-memory Base64 data URLs for visual evidence."""
    gradcam_heatmap: Optional[str] = Field(None, examples=["data:image/png;base64,..."])
    fft_spectrum: Optional[str] = Field(None, examples=["data:image/png;base64,..."])


class AnalyzeResponse(BaseModel):
    """Comprehensive forensic report returned to clients."""
    status: str = Field("success", examples=["success"])
    model_info: ModelInfo
    prediction: PredictionResult
    evidence: EvidenceSummary
    visualizations: Visualizations
    disclaimer: str = Field(
        default=(
            "This analysis provides probabilistic forensic indicators for "
            "research and screening purposes, not definitive proof."
        ),
        examples=[
            "This analysis provides probabilistic forensic indicators for "
            "research and screening purposes, not definitive proof."
        ],
    )


class ErrorResponse(BaseModel):
    """Standardized error payload."""
    status: str = Field("error", examples=["error"])
    detail: str = Field(..., examples=["Invalid file format or corrupted image payload."])
