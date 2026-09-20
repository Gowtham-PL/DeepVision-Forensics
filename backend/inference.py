"""
Inference engine and model service for DeepVision-Forensics.

Manages the singleton E1 SpatialClassifier model, input image decoding and validation,
forward probabilistic inference, Grad-CAM spatial explainability, and optional FFT spectrum extraction.
"""

import base64
import io
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    ModelOption,
    ModelsListResponse,
    MultiViewDiagnostics,
    PredictionResult,
    Visualizations,
)
from ml.gradcam import GradCAM, compute_log_magnitude_spectrum, overlay_heatmap
from models.fusion import build_model


def generate_5_crops(img: Image.Image) -> List[Image.Image]:
    """Generates 1 global (224x224) and 4 deterministic corner crops (60% scale resized to 224x224)."""
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    crops = []
    crops.append(img.resize((224, 224), Image.Resampling.BILINEAR))  # View 0: Global
    crops.append(img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR))  # View 1: Top-Left
    crops.append(img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR))  # View 2: Top-Right
    crops.append(img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR))  # View 3: Bottom-Left
    crops.append(img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR))  # View 4: Bottom-Right
    return crops


class ModelService:
    """
    Singleton service managing deployed forensic detection models.
    Supports on-demand loading, memory caching, and runtime selection between
    E1 Spatial and E3-Std Dual-Domain architectures.
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
        self.models: Dict[str, nn.Module] = {}
        self.model_params: Dict[str, int] = {}
        self.model_checkpoints: Dict[str, Path] = {}
        self.active_model_key: str = config.DEFAULT_MODEL_KEY
        self.device: torch.device = torch.device(config.DEVICE)
        self.checkpoint_path: Path = config.MODEL_CHECKPOINT_PATH
        self.param_count: int = 0
        self.model: Optional[nn.Module] = None
        self.infer_lock = threading.Lock()
        self.load_lock = threading.Lock()
        self._initialized = True

    @staticmethod
    def normalize_model_key(key: Optional[str]) -> str:
        """
        Normalizes aliases and case variations to standard model IDs.
        Defaults to config.DEFAULT_MODEL_KEY if key is None or empty.
        """
        if not key:
            return config.DEFAULT_MODEL_KEY
        k = str(key).strip().lower().replace("-", "_")
        if k in {"e6c", "e6_c", "e6c_multiview", "deepvision_e6_c_multiview", "multiview"}:
            return "e6c_multiview"
        if k in {"e1", "e1_spatial", "deepvision_e1_spatial", "spatial"}:
            return "e1_spatial"
        if k in {"e3", "e3_std", "deepvision_e3_std", "candidate_standardize", "dual", "dual_domain"}:
            return "e3_std"
        if k in {"e5", "e5_generalization", "deepvision_e5_generalization", "generalization", "production"}:
            return "e5_generalization"
        return k

    def load_model(
        self,
        model_key: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        device_str: Optional[str] = None,
    ) -> nn.Module:
        """
        Loads the specified model architecture and checkpoint into memory.
        
        Args:
            model_key: Target model key ('e6c_multiview', 'e5_generalization', 'e1_spatial', or 'e3_std'). Defaults to DEFAULT_MODEL_KEY.
            checkpoint_path: Path to checkpoint .pt file (optional override).
            device_str: Target device string ('cuda', 'cpu', etc.).

        Returns:
            Loaded PyTorch nn.Module in eval mode.
        """
        if device_str is not None:
            self.device = torch.device(device_str)
        elif torch.cuda.is_available() and config.DEVICE.startswith("cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        target_key = self.normalize_model_key(model_key)
        if target_key not in config.SUPPORTED_MODELS:
            supported = list(config.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unknown model key '{model_key}'. Supported models: {supported}")

        model_cfg = config.SUPPORTED_MODELS[target_key]
        target_ckpt = checkpoint_path or model_cfg["checkpoint_path"]

        if not Path(target_ckpt).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {target_ckpt}. "
                f"Ensure {model_cfg['checkpoint_rel']} exists."
            )

        with self.load_lock:
            # Build corresponding model architecture
            if target_key == "e1_spatial":
                model = build_model(experiment="E1", pretrained=False)
            elif target_key in {"e3_std", "e5_generalization"}:
                model = build_model(
                    experiment="E3",
                    pretrained=False,
                    freq_norm_strategy="standardize",
                )
            elif target_key == "e6c_multiview":
                from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model
                model = MultiViewE5Model(
                    checkpoint_path=None,
                    num_views=5,
                    freq_norm_strategy="standardize",
                    freq_embedding_dim=256,
                )
            else:
                raise ValueError(f"No architecture builder configured for '{target_key}'")

            # Load checkpoint weights safely
            ckpt = torch.load(target_ckpt, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            # Freeze parameter gradients for inference safety
            for param in model.parameters():
                param.requires_grad = False

            param_count = sum(p.numel() for p in model.parameters())

            # Warmup pass to initialize CUDA kernels if available
            try:
                if target_key == "e6c_multiview":
                    dummy_input = torch.zeros((1, 5, 3, 224, 224), dtype=torch.float32, device=self.device)
                else:
                    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
            except Exception:
                pass

            # Cache loaded model
            self.models[target_key] = model
            self.model_params[target_key] = param_count
            self.model_checkpoints[target_key] = Path(target_ckpt)

            # Update default active model references for backward compatibility
            if target_key == self.active_model_key or self.model is None:
                self.model = model
                self.param_count = param_count
                self.checkpoint_path = Path(target_ckpt)

            return model

    def get_or_load_model(self, model_key: Optional[str] = None) -> Tuple[str, nn.Module]:
        """
        Retrieves a cached model instance, loading it on demand if not yet cached.

        Returns:
            Tuple of (normalized_model_key, model_instance).
        """
        target_key = self.normalize_model_key(model_key or self.active_model_key)
        if target_key not in self.models:
            self.load_model(model_key=target_key)
        return target_key, self.models[target_key]

    def set_active_model(self, model_key: str) -> None:
        """Sets the system default active model."""
        target_key = self.normalize_model_key(model_key)
        _, model = self.get_or_load_model(target_key)
        self.active_model_key = target_key
        self.model = model
        self.param_count = self.model_params[target_key]
        self.checkpoint_path = self.model_checkpoints[target_key]

    def is_loaded(self, model_key: Optional[str] = None) -> bool:
        """Returns True if the requested model (or default model) is loaded."""
        if model_key is None:
            return self.model is not None or len(self.models) > 0
        target_key = self.normalize_model_key(model_key)
        return target_key in self.models

    def get_model_info(self, model_key: Optional[str] = None) -> ModelInfo:
        """Returns metadata about the requested or active inference model."""
        target_key = self.normalize_model_key(model_key or self.active_model_key)
        model_cfg = config.SUPPORTED_MODELS.get(target_key, {})
        param_count = self.model_params.get(target_key, self.param_count)

        return ModelInfo(
            name=model_cfg.get("name", config.MODEL_NAME),
            model_id=target_key,
            backbone=model_cfg.get("backbone", config.BACKBONE_NAME),
            parameters=param_count,
            device=str(self.device),
            description=model_cfg.get("description"),
        )

    def list_available_models(self) -> ModelsListResponse:
        """Lists all supported models with metadata and current active state."""
        options = []
        for key, info in config.SUPPORTED_MODELS.items():
            options.append(
                ModelOption(
                    id=key,
                    name=info["name"],
                    backbone=info["backbone"],
                    description=info["description"],
                    is_default=(key == self.active_model_key),
                    benchmark_unseen_auc=info.get("benchmark_unseen_auc"),
                )
            )
        return ModelsListResponse(
            status="success",
            active_model=self.active_model_key,
            models=options,
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
        model_key: Optional[str] = None,
    ) -> AnalyzeResponse:
        """
        Executes end-to-end forensic inference on a validated PIL RGB image.
        
        Args:
            image: Decoded RGB PIL image.
            include_fft: Whether to generate diagnostic 2D FFT visualization.
            model_key: Identifier of model to use.
            
        Returns:
            Structured AnalyzeResponse.
        """
        start_time = time.perf_counter()
        target_key, target_model = self.get_or_load_model(model_key)

        orig_w, orig_h = image.size
        orig_rgb_np = np.array(image)

        # Standard preprocessing transform: 224x224, [0, 1] tensor
        preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        diagnostics: Optional[MultiViewDiagnostics] = None
        strongest_local_prob: Optional[float] = None

        if target_key == "e6c_multiview":
            # Multi-view 5-crop preparation
            crop_images = generate_5_crops(image)
            crop_tensors = [preprocess_transform(c) for c in crop_images]
            x_views = torch.stack(crop_tensors, dim=0).unsqueeze(0).to(self.device)  # (1, 5, 3, 224, 224)
            input_tensor = crop_tensors[0].unsqueeze(0).to(self.device)  # Global view (1, 3, 224, 224)
        else:
            input_tensor = preprocess_transform(image).unsqueeze(0).to(self.device)

        with self.infer_lock:
            # 1. Forward inference for classification
            with torch.no_grad():
                if target_key == "e6c_multiview":
                    # Multi-view forward pass with feature extraction for per-view diagnostics
                    B, V, C, H, W = x_views.shape
                    x_flat = x_views.view(B * V, C, H, W)
                    feats = target_model.base_model(x_flat, return_features=True)
                    e_fused = feats["fused_embedding"]  # (5, 1792)
                    e_fused_views = e_fused.view(B, V, target_model.fused_dim)

                    attn_scores = target_model.view_attention(e_fused_views)  # (1, 5, 1)
                    attn_weights = torch.softmax(attn_scores, dim=1)  # (1, 5, 1)

                    # Learned attention pooled embedding
                    pooled_embedding = torch.sum(attn_weights * e_fused_views, dim=1)  # (1, 1792)
                    logit = target_model.classifier(pooled_embedding)  # (1, 1)
                    ai_prob = float(torch.sigmoid(logit).item())

                    # Per-view individual probabilities
                    view_logits = target_model.classifier(e_fused)  # (5, 1)
                    view_probs_tensor = torch.sigmoid(view_logits).view(5)
                    view_probs = [round(float(p), 4) for p in view_probs_tensor.cpu().numpy()]
                    view_weights = [round(float(w), 4) for w in attn_weights.view(5).cpu().numpy()]

                    strongest_local_prob = round(float(max(view_probs[1:])), 4)
                else:
                    logit = target_model(input_tensor)
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

            infer_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            if target_key == "e6c_multiview":
                diagnostics = MultiViewDiagnostics(
                    overall_ai_probability=round(ai_prob, 4),
                    strongest_local_probability=strongest_local_prob,
                    strongest_local_prob=strongest_local_prob,
                    view_probabilities=view_probs,
                    attention_weights=view_weights,
                    predicted_class=classification_label,
                    inference_time_ms=infer_time_ms,
                    global_view_prob=view_probs[0],
                    top_left_prob=view_probs[1],
                    top_right_prob=view_probs[2],
                    bottom_left_prob=view_probs[3],
                    bottom_right_prob=view_probs[4],
                    aggregation_strategy="learned_attention",
                )

            # 3. Generate Grad-CAM spatial explainability on global image view
            gradcam_target = target_model.base_model if hasattr(target_model, "base_model") else target_model
            gradcam = GradCAM(gradcam_target)
            try:
                with torch.enable_grad():
                    cam_tensor = input_tensor.clone().detach().requires_grad_(True)
                    cam_heatmap = gradcam.generate_heatmap(cam_tensor)
                
                # Resize heatmap to match original image dimensions if needed
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
                gradcam_target.zero_grad()

            # 4. Diagnostic Frequency Spectrum (always minmax scaled for visual contrast)
            fft_b64: Optional[str] = None
            if include_fft:
                spec_norm = compute_log_magnitude_spectrum(input_tensor, norm_strategy="minmax")
                fft_b64 = self._encode_spectrum_to_base64_png(spec_norm, colormap=cv2.COLORMAP_VIRIDIS)

        infer_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        if diagnostics is not None:
            diagnostics.inference_time_ms = infer_time_ms

        # 5. Model-specific evidence summaries
        if target_key == "e6c_multiview":
            spatial_summary = (
                "Multi-View Attention & Grad-CAM: Analyzes 5 multi-scale views (1 global + 4 local crops) "
                "using learned self-attention. The heatmap visualizes spatial regions receiving primary focus on the global view. "
                "Note: Saliency heatmaps indicate model attention patterns rather than pixel-exact manipulation boundaries or edit masks."
            )
            frequency_summary = (
                "Dual-Domain Frequency Analysis (2D FFT Log-Magnitude): Evaluates spectral energy across views "
                "using standardized frequency representations to detect synthesis artifacts and localized tampering."
                if include_fft
                else None
            )
        elif target_key == "e5_generalization":
            spatial_summary = (
                "Model Attention (Grad-CAM): Visualizes spatial feature regions receiving primary attention in the dual-branch backbone. "
                "Note: Saliency heatmaps indicate model attention patterns rather than pixel-exact manipulation boundaries or edit masks."
            )
            frequency_summary = (
                "Frequency Evidence (2D FFT Log-Magnitude): Illustrates frequency energy distribution across spatial frequencies. "
                "The dual-branch architecture leverages standardized spectral features to detect cross-generator synthesis artifacts."
                if include_fft
                else None
            )
        elif target_key == "e3_std":
            spatial_summary = (
                "Model Attention (Grad-CAM): Visualizes spatial feature regions receiving primary attention in the dual-branch backbone. "
                "Note: Saliency heatmaps indicate model attention patterns rather than pixel-exact manipulation boundaries or edit masks."
            )
            frequency_summary = (
                "Frequency Evidence (2D FFT Log-Magnitude): Illustrates frequency energy distribution. "
                "The dual-branch model uses standardized frequency features to identify periodic synthesis artifacts."
                if include_fft
                else None
            )
        else:
            spatial_summary = (
                "Model Attention (Grad-CAM): Visualizes spatial regions receiving stronger model attention in the EfficientNet-B3 backbone. "
                "Note: Saliency heatmaps indicate model attention patterns rather than pixel-exact manipulation boundaries or edit masks."
            )
            frequency_summary = (
                "Frequency Evidence (2D FFT Log-Magnitude): Illustrates spectral energy distribution across spatial frequencies."
                if include_fft
                else None
            )

        # 6. Compile structured response
        return AnalyzeResponse(
            status="success",
            model_info=self.get_model_info(target_key),
            prediction=PredictionResult(
                classification_label=classification_label,
                ai_probability=round(ai_prob, 4),
                authenticity_assessment=authenticity_assessment,
                risk_indicator=risk_indicator,
                threshold_used=config.CLASSIFICATION_THRESHOLD,
                strongest_local_probability=strongest_local_prob,
                inference_time_ms=infer_time_ms,
            ),
            evidence=EvidenceSummary(
                spatial_summary=spatial_summary,
                frequency_summary=frequency_summary,
            ),
            visualizations=Visualizations(
                gradcam_heatmap=gradcam_b64,
                fft_spectrum=fft_b64,
            ),
            diagnostics=diagnostics,
            disclaimer=(
                "This analysis provides probabilistic forensic indicators for research "
                "and screening purposes, not definitive proof."
            ),
        )


# Global service instance
model_service = ModelService()
