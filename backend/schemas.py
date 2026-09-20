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
    active_model: Optional[str] = Field(None, description="Active model key identifier", examples=["e1_spatial"])
    available_models: Optional[List[str]] = Field(None, description="List of supported model identifiers", examples=[["e1_spatial", "e3_std"]])
    device: str = Field(..., description="Compute device being utilized", examples=["cuda"])
    gpu_name: Optional[str] = Field(None, description="Physical GPU name if CUDA is enabled", examples=["NVIDIA GeForce RTX 3050 Laptop GPU"])


class ModelInfo(BaseModel):
    """Metadata regarding the deployed inference model."""
    name: str = Field(..., examples=["DeepVision-E1-Spatial"])
    model_id: Optional[str] = Field(None, description="Model identifier key", examples=["e1_spatial"])
    backbone: str = Field(..., examples=["EfficientNet-B3"])
    parameters: int = Field(..., examples=[11549993])
    device: str = Field(..., examples=["cuda"])
    description: Optional[str] = Field(None, examples=["Spatial-only baseline (EfficientNet-B3)"])


class ModelOption(BaseModel):
    """Available model specification."""
    id: str = Field(..., examples=["e1_spatial"])
    name: str = Field(..., examples=["DeepVision-E1-Spatial"])
    backbone: str = Field(..., examples=["EfficientNet-B3"])
    description: str = Field(..., examples=["Spatial-only baseline (EfficientNet-B3 backbone)"])
    is_default: bool = Field(False, examples=[True])
    benchmark_unseen_auc: Optional[float] = Field(None, examples=[0.8991])


class ModelsListResponse(BaseModel):
    """List of selectable models and current active default."""
    status: str = Field("success", examples=["success"])
    active_model: str = Field(..., examples=["e1_spatial"])
    models: List[ModelOption]


class PredictionResult(BaseModel):
    """Probabilistic prediction outputs."""
    classification_label: str = Field(..., examples=["AI-generated"])
    ai_probability: float = Field(..., ge=0.0, le=1.0, examples=[0.9142])
    authenticity_assessment: str = Field(
        ..., examples=["Model estimate: 91.4% probability of AI generation."]
    )
    risk_indicator: str = Field(..., examples=["HIGH"])
    threshold_used: float = Field(..., examples=[0.50])
    strongest_local_probability: Optional[float] = Field(
        None,
        description="Strongest local crop AI probability among local corner views (Views 1-4)",
        examples=[0.9595],
    )
    inference_time_ms: Optional[float] = Field(
        None,
        description="Inference execution latency in milliseconds",
        examples=[45.2],
    )


class MultiViewDiagnostics(BaseModel):
    """Diagnostic outputs for multi-view forensic architecture."""
    overall_ai_probability: float = Field(..., description="Overall aggregated AI probability from learned attention", examples=[0.8638])
    strongest_local_probability: float = Field(..., description="Strongest localized crop AI probability", examples=[0.9595])
    view_probabilities: List[float] = Field(..., description="Probabilities for each of the 5 views: [Global, Top-Left, Top-Right, Bottom-Left, Bottom-Right]", examples=[[0.8628, 0.4138, 0.9595, 0.7578, 0.7891]])
    attention_weights: List[float] = Field(..., description="Learned attention weights across the 5 views", examples=[[0.2148, 0.2041, 0.2061, 0.1983, 0.1767]])
    predicted_class: str = Field(..., description="Classification label at 0.50 threshold", examples=["AI-generated"])
    inference_time_ms: float = Field(..., description="Inference runtime in milliseconds", examples=[45.2])
    strongest_local_prob: Optional[float] = Field(None, description="Alias for strongest_local_probability")
    global_view_prob: Optional[float] = Field(None, description="Global view probability")
    top_left_prob: Optional[float] = Field(None, description="Top-left corner probability")
    top_right_prob: Optional[float] = Field(None, description="Top-right corner probability")
    bottom_left_prob: Optional[float] = Field(None, description="Bottom-left corner probability")
    bottom_right_prob: Optional[float] = Field(None, description="Bottom-right corner probability")
    aggregation_strategy: str = Field("learned_attention", description="Aggregation strategy applied")
    views_description: Optional[List[str]] = Field(
        default=[
            "View 0 (Global)",
            "View 1 (Top-Left 60%)",
            "View 2 (Top-Right 60%)",
            "View 3 (Bottom-Left 60%)",
            "View 4 (Bottom-Right 60%)",
        ]
    )


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
    diagnostics: Optional[MultiViewDiagnostics] = Field(
        None,
        description="Diagnostic multi-view forensic outputs (available for E6-C Multi-View model)",
    )
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
