"""
Inference engine and model service for DeepVision-Forensics.

Manages the singleton E1 SpatialClassifier model, input image decoding and validation,
forward probabilistic inference, Grad-CAM spatial explainability, and optional FFT spectrum extraction.
"""

import base64
import io
import threading
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from backend import config
from backend.schemas import (
    AnalyzeResponse,
    EvidenceSummary,
    ModelInfo,
    PredictionResult,
    Visualizations,
)
from ml.gradcam import GradCAM, compute_log_magnitude_spectrum, overlay_heatmap
from models.fusion import build_model


class ModelService:
    """
    Singleton service managing the deployed E1 SpatialClassifier model.
    """
    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device(config.DEVICE)
        self.checkpoint_path: Path = config.MODEL_CHECKPOINT_PATH
        self.param_count: int = 0
        self.infer_lock = threading.Lock()
        self._initialized = True

    def load_model(
        self,
        checkpoint_path: Optional[Path] = None,
        device_str: Optional[str] = None,
    ) -> None:
        """
        Loads the E1 SpatialClassifier model weights into memory.
        
        Args:
            checkpoint_path: Path to checkpoint .pt file.
            device_str: Target device string ('cuda', 'cpu', etc.).
        """
        if device_str is not None:
            self.device = torch.device(device_str)
        elif torch.cuda.is_available() and config.DEVICE.startswith("cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        target_ckpt = checkpoint_path or self.checkpoint_path

        if not target_ckpt.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {target_ckpt}. "
                "Ensure experiments/e1_spatial/best_model.pt exists or configure MODEL_CHECKPOINT_PATH."
            )

        # Build E1 spatial model architecture
        model = build_model(experiment="E1", pretrained=False)
        
        # Load weights safely
        ckpt = torch.load(target_ckpt, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        # Freeze parameter gradients for inference safety
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.checkpoint_path = target_ckpt
        self.param_count = sum(p.numel() for p in model.parameters())

        # Perform one-time warmup pass to initialize CUDA kernels
        try:
            dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
        except Exception:
            pass

    def is_loaded(self) -> bool:
        """Returns True if the model is loaded and ready for inference."""
        return self.model is not None

    def get_model_info(self) -> ModelInfo:
        """Returns metadata about the active inference model."""
        return ModelInfo(
            name=config.MODEL_NAME,
            backbone=config.BACKBONE_NAME,
            parameters=self.param_count,
            device=str(self.device),
        )

    def validate_and_decode_image(self, image_bytes: bytes) -> Image.Image:
        """
        Validates raw payload bytes and decodes into a PIL RGB Image.
        
        Raises ValueError on invalid format, excessive size, or corrupted payload.
        """
        if len(image_bytes) == 0:
            raise ValueError("Uploaded file payload is empty.")

        if len(image_bytes) > config.MAX_UPLOAD_BYTES:
            raise ValueError(
                f"File size ({len(image_bytes) / (1024**2):.1f} MB) exceeds maximum "
                f"allowed limit of {config.MAX_UPLOAD_SIZE_MB} MB."
            )

        try:
            # First pass: verify image integrity
            bio = io.BytesIO(image_bytes)
            with Image.open(bio) as img:
                img.verify()
                img_format = img.format

            # Second pass: decode pixel data
            bio.seek(0)
            img = Image.open(bio)
            
            # Format validation
            if img_format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError(
                    f"Unsupported image format '{img_format}'. Allowed formats: PNG, JPEG, WEBP."
                )

            # Ensure 3-channel RGB representation
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Validate dimensions
            if img.width < 16 or img.height < 16:
                raise ValueError(
                    f"Image resolution ({img.width}x{img.height}) is too small. Minimum resolution: 16x16."
                )

            return img

        except (Image.UnidentifiedImageError, OSError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Invalid or corrupted image payload: {str(exc)}")

    def _encode_rgb_to_base64_png(self, rgb_array: np.ndarray) -> str:
        """Encodes an RGB uint8 numpy array to a Base64 data URL string."""
        if rgb_array.dtype != np.uint8:
            rgb_array = np.clip(rgb_array * 255.0, 0, 255).astype(np.uint8)
        
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", bgr_array)
        if not success:
            raise RuntimeError("Failed to encode image to PNG format.")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def _encode_spectrum_to_base64_png(
        self,
        spec_norm_np: np.ndarray,
        colormap: int = cv2.COLORMAP_VIRIDIS,
    ) -> str:
        """Encodes a normalized 2D spectrum float array to a colorized Base64 PNG."""
        spec_uint8 = np.clip(spec_norm_np * 255.0, 0, 255).astype(np.uint8)
        colorized = cv2.applyColorMap(spec_uint8, colormap)
        colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        return self._encode_rgb_to_base64_png(colorized_rgb)

    def predict_and_analyze(
        self,
        image: Image.Image,
        include_fft: bool = True,
    ) -> AnalyzeResponse:
        """
        Executes end-to-end forensic inference on a validated PIL RGB image.
        
        Args:
            image: Decoded RGB PIL image.
            include_fft: Whether to generate diagnostic 2D FFT visualization.
            
        Returns:
            Structured AnalyzeResponse.
        """
        if not self.is_loaded():
            raise RuntimeError("Model service is not loaded. Call load_model() first.")

        orig_w, orig_h = image.size
        orig_rgb_np = np.array(image)

        # Standard preprocessing: resize to 224x224 and scale to [0, 1] tensor
        preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess_transform(image).unsqueeze(0).to(self.device)

        with self.infer_lock:
            # 1. Forward inference for classification
            with torch.no_grad():
                logit = self.model(input_tensor)
                if isinstance(logit, dict):
                    logit = logit["logit"]
                ai_prob = float(torch.sigmoid(logit).item())

            # 2. Derive classification and risk indicators
            is_ai = ai_prob >= config.CLASSIFICATION_THRESHOLD
            classification_label = "AI-generated" if is_ai else "Real"

            if ai_prob >= 0.70:
                risk_indicator = "HIGH"
            elif ai_prob >= 0.30:
                risk_indicator = "MEDIUM"
            else:
                risk_indicator = "LOW"

            authenticity_assessment = (
                f"Model estimate: {ai_prob * 100:.1f}% probability of AI generation."
            )

            # 3. Generate Grad-CAM spatial explainability
            gradcam = GradCAM(self.model)
            try:
                # Grad-CAM requires localized gradient tracking through features
                with torch.enable_grad():
                    # Enable gradient tracking on input tensor so feature activations receive gradients
                    cam_tensor = input_tensor.clone().detach().requires_grad_(True)
                    cam_heatmap = gradcam.generate_heatmap(cam_tensor)
                
                # Overlay heatmap onto original un-resized image (resizing heatmap to match original dims if needed)
                if cam_heatmap.shape != (orig_h, orig_w):
                    cam_heatmap = cv2.resize(
                        cam_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                    )

                blended_overlay = overlay_heatmap(
                    image_rgb_np=orig_rgb_np,
                    heatmap=cam_heatmap,
                    alpha=0.5,
                    colormap=cv2.COLORMAP_JET,
                )
                gradcam_b64 = self._encode_rgb_to_base64_png(blended_overlay)
            finally:
                gradcam.remove_hooks()
                self.model.zero_grad()

            # 4. Optional Diagnostic Frequency Spectrum
            fft_b64: Optional[str] = None
            if include_fft:
                # Extract log magnitude spectrum on unnormalized [0, 1] tensor
                spec_norm = compute_log_magnitude_spectrum(input_tensor, norm_strategy="minmax")
                fft_b64 = self._encode_spectrum_to_base64_png(spec_norm, colormap=cv2.COLORMAP_VIRIDIS)

        # 5. Compile structured response
        return AnalyzeResponse(
            status="success",
            model_info=self.get_model_info(),
            prediction=PredictionResult(
                classification_label=classification_label,
                ai_probability=round(ai_prob, 4),
                authenticity_assessment=authenticity_assessment,
                risk_indicator=risk_indicator,
                threshold_used=config.CLASSIFICATION_THRESHOLD,
            ),
            evidence=EvidenceSummary(
                spatial_summary=(
                    "Grad-CAM visualization showing spatial regions receiving stronger model attention."
                ),
                frequency_summary=(
                    "Diagnostic 2D Fast Fourier Transform (FFT) log-magnitude spectrum showing spectral energy distribution."
                    if include_fft
                    else None
                ),
            ),
            visualizations=Visualizations(
                gradcam_heatmap=gradcam_b64,
                fft_spectrum=fft_b64,
            ),
            disclaimer=(
                "This analysis provides probabilistic forensic indicators for research "
                "and screening purposes, not definitive proof."
            ),
        )


# Global service instance
model_service = ModelService()
