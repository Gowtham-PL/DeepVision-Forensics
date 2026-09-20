"""
Unit and integration tests for DeepVision-Forensics Phase 4 backend inference layer.
"""

import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from backend import config
from backend.inference import ModelService, model_service
from backend.main import app
from backend.schemas import AnalyzeResponse, HealthResponse


def create_synthetic_image_bytes(
    width: int = 64,
    height: int = 64,
    fmt: str = "PNG",
    mode: str = "RGB",
    color: tuple = (128, 64, 200),
) -> bytes:
    """Helper to generate in-memory synthetic image bytes."""
    img = Image.new(mode, (width, height), color=color)
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


@pytest.fixture(scope="module")
def loaded_service():
    """Fixture ensuring the model service is loaded for tests."""
    if not model_service.is_loaded():
        model_service.load_model()
    return model_service


@pytest.fixture(scope="module")
def api_client(loaded_service):
    """Test client for FastAPI app."""
    with TestClient(app) as client:
        yield client


class TestModelServiceLoading:
    """Tests for ModelService loading and device handling."""

    def test_model_loading_and_singleton(self, loaded_service):
        assert loaded_service.is_loaded() is True
        assert loaded_service.model is not None
        assert loaded_service.param_count == 12365322

        # Verify singleton pattern
        service2 = ModelService()
        assert service2 is loaded_service

    def test_device_selection(self, loaded_service):
        assert isinstance(loaded_service.device, torch.device)
        assert loaded_service.device.type in {"cuda", "cpu"}

    def test_missing_checkpoint_raises_error(self):
        service = ModelService()
        with pytest.raises(FileNotFoundError):
            service.load_model(checkpoint_path=Path("non_existent_path/checkpoint.pt"))


class TestImageValidationAndDecoding:
    """Tests for image validation, decoding, and format rejection."""

    def test_valid_png_decoding(self, loaded_service):
        png_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG", mode="RGB")
        img = loaded_service.validate_and_decode_image(png_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_valid_jpeg_decoding(self, loaded_service):
        jpg_bytes = create_synthetic_image_bytes(64, 64, fmt="JPEG", mode="RGB")
        img = loaded_service.validate_and_decode_image(jpg_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_valid_webp_decoding(self, loaded_service):
        webp_bytes = create_synthetic_image_bytes(64, 64, fmt="WEBP", mode="RGB")
        img = loaded_service.validate_and_decode_image(webp_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_rgba_and_grayscale_rgb_conversion(self, loaded_service):
        rgba_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG", mode="RGBA", color=(100, 150, 200, 255))
        img_rgba = loaded_service.validate_and_decode_image(rgba_bytes)
        assert img_rgba.mode == "RGB"

        gray_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG", mode="L", color=128)
        img_gray = loaded_service.validate_and_decode_image(gray_bytes)
        assert img_gray.mode == "RGB"

    def test_corrupted_image_rejection(self, loaded_service):
        corrupted_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"random_corrupted_garbage_bytes"
        with pytest.raises(ValueError, match="Invalid or corrupted"):
            loaded_service.validate_and_decode_image(corrupted_bytes)

    def test_empty_payload_rejection(self, loaded_service):
        with pytest.raises(ValueError, match="empty"):
            loaded_service.validate_and_decode_image(b"")

    def test_non_image_file_rejection(self, loaded_service):
        text_bytes = b"Hello, this is a plain text file, not an image."
        with pytest.raises(ValueError, match="Invalid or corrupted"):
            loaded_service.validate_and_decode_image(text_bytes)

    def test_oversized_payload_rejection(self, loaded_service, monkeypatch):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
        large_bytes = b"0" * 200
        with pytest.raises(ValueError, match="exceeds maximum"):
            loaded_service.validate_and_decode_image(large_bytes)


class TestInferenceAndExplainability:
    """Tests for model forward pass, probabilistic output, Grad-CAM, and FFT."""

    def test_predict_and_analyze_schema(self, loaded_service):
        img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        response = loaded_service.predict_and_analyze(img, include_fft=True)

        assert isinstance(response, AnalyzeResponse)
        assert response.status == "success"
        assert response.model_info.name == config.MODEL_NAME
        assert response.model_info.parameters == 12365322
        
        # Probabilities and labels
        assert 0.0 <= response.prediction.ai_probability <= 1.0
        assert response.prediction.classification_label in {"Real", "AI-generated"}
        if response.prediction.ai_probability >= 0.50:
            assert response.prediction.classification_label == "AI-generated"
        else:
            assert response.prediction.classification_label == "Real"

        # Evidence summaries
        assert "Grad-CAM" in response.evidence.spatial_summary
        assert response.evidence.frequency_summary is not None

        # Multi-view diagnostics
        assert response.diagnostics is not None
        assert 0.0 <= response.diagnostics.global_view_prob <= 1.0
        assert 0.0 <= response.diagnostics.strongest_local_prob <= 1.0
        assert len(response.diagnostics.attention_weights) == 5
        assert response.diagnostics.aggregation_strategy == "learned_attention"
        assert response.prediction.strongest_local_probability is not None
        assert response.prediction.inference_time_ms is not None

        # Visualizations
        assert response.visualizations.gradcam_heatmap is not None
        assert response.visualizations.gradcam_heatmap.startswith("data:image/png;base64,")
        assert response.visualizations.fft_spectrum is not None
        assert response.visualizations.fft_spectrum.startswith("data:image/png;base64,")

    def test_include_fft_false_behavior(self, loaded_service):
        img = Image.new("RGB", (64, 64), color=(50, 100, 150))
        resp_with_fft = loaded_service.predict_and_analyze(img, include_fft=True)
        resp_no_fft = loaded_service.predict_and_analyze(img, include_fft=False)

        # Probabilities should be identical
        assert resp_with_fft.prediction.ai_probability == resp_no_fft.prediction.ai_probability
        assert resp_with_fft.prediction.classification_label == resp_no_fft.prediction.classification_label

        # FFT fields
        assert resp_no_fft.visualizations.fft_spectrum is None
        assert resp_no_fft.evidence.frequency_summary is None

    def test_model_parameter_integrity_before_and_after_inference(self, loaded_service):
        """Verifies that forward pass and Grad-CAM backprop do not alter model weights."""
        def get_model_hash():
            hasher = hashlib.sha256()
            for p in loaded_service.model.parameters():
                hasher.update(p.detach().cpu().numpy().tobytes())
            return hasher.hexdigest()

        hash_before = get_model_hash()

        # Run multiple inference passes
        img = Image.new("RGB", (64, 64), color=(80, 120, 160))
        for _ in range(3):
            _ = loaded_service.predict_and_analyze(img, include_fft=True)

        hash_after = get_model_hash()
        assert hash_before == hash_after, "Model parameters were mutated during inference or Grad-CAM!"


class TestFastAPIEndpoints:
    """Integration tests for FastAPI REST endpoints."""

    def test_health_endpoint(self, api_client):
        response = api_client.get(f"{config.API_V1_PREFIX}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == config.MODEL_NAME
        assert "device" in data

    def test_analyze_endpoint_success(self, api_client):
        img_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG", color=(150, 75, 220))
        files = {"file": ("test_image.png", img_bytes, "image/png")}
        data = {"include_fft": "true"}

        response = api_client.post(f"{config.API_V1_PREFIX}/analyze", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "success"
        assert "prediction" in payload
        assert "ai_probability" in payload["prediction"]
        assert 0.0 <= payload["prediction"]["ai_probability"] <= 1.0
        assert payload["visualizations"]["gradcam_heatmap"].startswith("data:image/png;base64,")
        assert payload["visualizations"]["fft_spectrum"].startswith("data:image/png;base64,")

    def test_analyze_endpoint_no_fft(self, api_client):
        img_bytes = create_synthetic_image_bytes(64, 64, fmt="JPEG", color=(100, 200, 100))
        files = {"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        data = {"include_fft": "false"}

        response = api_client.post(f"{config.API_V1_PREFIX}/analyze", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()
        assert payload["visualizations"]["fft_spectrum"] is None

    def test_analyze_endpoint_invalid_file_rejection(self, api_client):
        files = {"file": ("bad_file.txt", b"Plain text content not an image", "text/plain")}
        response = api_client.post(f"{config.API_V1_PREFIX}/analyze", files=files)
        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "error"
        assert "detail" in payload


class TestModelSelection:
    """Tests for model selection between E6-C Multi-View, E5 Generalization, E1 Spatial, and E3-Std Dual-Domain."""

    def test_e6c_is_default_model(self, loaded_service):
        assert loaded_service.active_model_key == "e6c_multiview"
        info = loaded_service.get_model_info()
        assert info.name == "DeepVision-E6-C Multi-View Forensics"
        assert info.parameters == 12365322

    def test_load_e5_generalization_model(self, loaded_service):
        e5_model = loaded_service.load_model("e5_generalization")
        assert e5_model is not None
        assert loaded_service.is_loaded("e5_generalization") is True
        e5_info = loaded_service.get_model_info("e5_generalization")
        assert e5_info.name == "DeepVision-E5-Generalization"
        assert e5_info.parameters == 12135689

    def test_load_e1_spatial_model(self, loaded_service):
        e1_model = loaded_service.load_model("e1_spatial")
        assert e1_model is not None
        assert loaded_service.is_loaded("e1_spatial") is True
        e1_info = loaded_service.get_model_info("e1_spatial")
        assert e1_info.name == "DeepVision-E1-Spatial"
        assert e1_info.parameters == 11549993

    def test_load_e3_std_model(self, loaded_service):
        e3_model = loaded_service.load_model("e3_std")
        assert e3_model is not None
        assert loaded_service.is_loaded("e3_std") is True
        e3_info = loaded_service.get_model_info("e3_std")
        assert e3_info.name == "DeepVision-E3-Std"
        assert e3_info.parameters == 12135689

    def test_predict_and_analyze_e5_generalization(self, loaded_service):
        img = Image.new("RGB", (128, 128), color=(220, 180, 100))
        resp = loaded_service.predict_and_analyze(img, include_fft=True, model_key="e5_generalization")
        assert resp.status == "success"
        assert resp.model_info.name == "DeepVision-E5-Generalization"
        assert resp.model_info.parameters == 12135689
        assert 0.0 <= resp.prediction.ai_probability <= 1.0
        assert resp.visualizations.gradcam_heatmap.startswith("data:image/png;base64,")
        assert resp.visualizations.fft_spectrum.startswith("data:image/png;base64,")

    def test_models_list_endpoint(self, api_client):
        response = api_client.get(f"{config.API_V1_PREFIX}/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "models" in data
        model_ids = [m["id"] for m in data["models"]]
        assert "e6c_multiview" in model_ids
        assert "e5_generalization" in model_ids
        assert "e1_spatial" in model_ids
        assert "e3_std" in model_ids
        assert data["active_model"] == "e6c_multiview"

    def test_analyze_endpoint_with_explicit_model_selection(self, api_client):
        img_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG", color=(75, 150, 225))
        files = {"file": ("test_e1.png", img_bytes, "image/png")}
        data = {"include_fft": "true", "model": "e1_spatial"}

        response = api_client.post(f"{config.API_V1_PREFIX}/analyze", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_info"]["name"] == "DeepVision-E1-Spatial"
        assert payload["model_info"]["parameters"] == 11549993

    def test_analyze_endpoint_default_fallback(self, api_client):
        """Omitting the model parameter must use default E6-C Multi-View behavior."""
        img_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG", color=(100, 100, 100))
        files = {"file": ("test_default.png", img_bytes, "image/png")}
        response = api_client.post(f"{config.API_V1_PREFIX}/analyze", files=files)
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_info"]["name"] == "DeepVision-E6-C Multi-View Forensics"
        assert payload["model_info"]["parameters"] == 12365322
        assert payload["diagnostics"] is not None
        assert payload["diagnostics"]["aggregation_strategy"] == "learned_attention"
        assert payload["prediction"]["strongest_local_probability"] is not None

    def test_analyze_endpoint_invalid_model_rejected(self, api_client):
        img_bytes = create_synthetic_image_bytes(64, 64, fmt="PNG")
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {"model": "non_existent_model_xyz"}
        response = api_client.post(f"{config.API_V1_PREFIX}/analyze", files=files, data=data)
        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "error"
        assert "Unsupported model" in payload["detail"]
