"""
Unit tests for DeepVision-Forensics model architectures, transforms, explainability,
and training components.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.spatial import SpatialBranch, SpatialClassifier
from models.frequency import (
    FFTTransform,
    FrequencyEncoder,
    FrequencyBranch,
    FrequencyClassifier,
)
from models.fusion import DeepVisionFusionModel, build_model
from ml.gradcam import GradCAM, overlay_heatmap, compute_log_magnitude_spectrum
from ml.train import build_parameter_groups, compute_roc_auc
from ml.evaluate import compute_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rgb_batch() -> torch.Tensor:
    """Returns a synthetic unnormalized [0, 1] RGB batch of shape (2, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)


# ── Spatial Branch Tests ──────────────────────────────────────────────────────

class TestSpatialBranch:
    def test_spatial_branch_embedding_shape(self, sample_rgb_batch):
        model = SpatialBranch(pretrained=False, apply_imagenet_norm=True)
        model.eval()
        with torch.no_grad():
            emb = model(sample_rgb_batch)
        assert emb.shape == (2, 1536), f"Expected shape (2, 1536), got {emb.shape}"
        assert not torch.isnan(emb).any(), "Spatial embedding contains NaN"

    def test_spatial_classifier_logit_shape(self, sample_rgb_batch):
        model = SpatialClassifier(pretrained=False)
        model.eval()
        with torch.no_grad():
            logit = model(sample_rgb_batch)
        assert logit.shape == (2, 1), f"Expected shape (2, 1), got {logit.shape}"

    def test_spatial_branch_normalization(self):
        model = SpatialBranch(pretrained=False, apply_imagenet_norm=True)
        ones_input = torch.ones(1, 3, 224, 224)
        norm_output = model.normalize(ones_input)
        assert norm_output.shape == (1, 3, 224, 224)
        # Expected normalized value for channel 0 (R): (1.0 - 0.485) / 0.229 ≈ 2.2489
        expected_r = (1.0 - 0.485) / 0.229
        assert torch.isclose(norm_output[0, 0, 0, 0], torch.tensor(expected_r), atol=1e-4)


# ── Frequency Branch Tests ────────────────────────────────────────────────────

class TestFrequencyBranch:
    def test_fft_transform_shape(self, sample_rgb_batch):
        fft_trans = FFTTransform(norm_strategy="minmax")
        spec = fft_trans(sample_rgb_batch)
        assert spec.shape == (2, 1, 224, 224), f"Expected spectrum shape (2, 1, 224, 224), got {spec.shape}"
        assert not torch.isnan(spec).any(), "FFT spectrum contains NaN"
        # Minmax output should be in [0, 1]
        assert spec.min() >= 0.0 - 1e-6, "Spectrum below 0.0"
        assert spec.max() <= 1.0 + 1e-6, "Spectrum above 1.0"

    @pytest.mark.parametrize("strategy", ["minmax", "standardize", "instance_norm", "none"])
    def test_frequency_normalization_strategies(self, sample_rgb_batch, strategy):
        fft_trans = FFTTransform(norm_strategy=strategy)
        spec = fft_trans(sample_rgb_batch)
        assert spec.shape == (2, 1, 224, 224)
        assert not torch.isnan(spec).any()
        if strategy == "standardize":
            # Zero mean per spectrum
            mean_val = spec[0, 0].mean().item()
            assert abs(mean_val) < 1e-4, f"Standardize mean not close to 0: {mean_val}"

    def test_frequency_encoder_shape(self):
        encoder = FrequencyEncoder(in_channels=1, embedding_dim=256)
        encoder.eval()
        dummy_spec = torch.rand(2, 1, 224, 224)
        with torch.no_grad():
            emb = encoder(dummy_spec)
        assert emb.shape == (2, 256), f"Expected shape (2, 256), got {emb.shape}"
        assert not torch.isnan(emb).any()

    def test_frequency_branch_forward(self, sample_rgb_batch):
        branch = FrequencyBranch(norm_strategy="minmax", embedding_dim=256)
        branch.eval()
        with torch.no_grad():
            emb, spec = branch(sample_rgb_batch, return_spectrum=True)
        assert emb.shape == (2, 256)
        assert spec.shape == (2, 1, 224, 224)

    def test_frequency_classifier_logit_shape(self, sample_rgb_batch):
        classifier = FrequencyClassifier(norm_strategy="minmax", embedding_dim=256)
        classifier.eval()
        with torch.no_grad():
            logit = classifier(sample_rgb_batch)
        assert logit.shape == (2, 1), f"Expected shape (2, 1), got {logit.shape}"


# ── Dual-Branch Fusion Tests ──────────────────────────────────────────────────

class TestFusionModel:
    def test_fusion_model_forward_shapes(self, sample_rgb_batch):
        model = DeepVisionFusionModel(spatial_pretrained=False, freq_embedding_dim=256)
        model.eval()
        with torch.no_grad():
            logit = model(sample_rgb_batch)
            features_dict = model(sample_rgb_batch, return_features=True)

        assert logit.shape == (2, 1), f"Expected logit shape (2, 1), got {logit.shape}"
        assert features_dict["spatial_embedding"].shape == (2, 1536)
        assert features_dict["frequency_embedding"].shape == (2, 256)
        assert features_dict["fused_embedding"].shape == (2, 1792)
        assert features_dict["fft_spectrum"].shape == (2, 1, 224, 224)

    def test_factory_builder(self):
        m_e1 = build_model("E1", pretrained=False)
        m_e2 = build_model("E2")
        m_e3 = build_model("E3", pretrained=False)

        assert isinstance(m_e1, SpatialClassifier)
        assert isinstance(m_e2, FrequencyClassifier)
        assert isinstance(m_e3, DeepVisionFusionModel)

    def test_gradient_flow_both_branches(self, sample_rgb_batch):
        """Verifies that gradients propagate through both spatial and frequency branches."""
        model = DeepVisionFusionModel(spatial_pretrained=False, freq_embedding_dim=256)
        model.train()
        criterion = nn.BCEWithLogitsLoss()

        sample_rgb_batch.requires_grad_(True)
        labels = torch.tensor([[1.0], [0.0]])

        logits = model(sample_rgb_batch)
        loss = criterion(logits, labels)
        loss.backward()

        # Check spatial branch gradient
        spatial_grad = model.spatial_branch.features[0][0].weight.grad
        assert spatial_grad is not None, "Spatial branch conv layer did not receive gradients"
        assert not torch.isnan(spatial_grad).any()

        # Check frequency branch gradient
        freq_grad = model.frequency_branch.encoder.block1[0].weight.grad
        assert freq_grad is not None, "Frequency branch conv layer did not receive gradients"
        assert not torch.isnan(freq_grad).any()

        # Check classification head gradient
        head_grad = model.classifier[0].weight.grad
        assert head_grad is not None, "Classifier head did not receive gradients"


# ── CUDA Device Compatibility Tests ───────────────────────────────────────────

class TestDeviceCompatibility:
    def test_cuda_forward_if_available(self, sample_rgb_batch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this machine")

        device = torch.device("cuda")
        model = DeepVisionFusionModel(spatial_pretrained=False).to(device)
        model.eval()

        cuda_batch = sample_rgb_batch.to(device)
        with torch.no_grad():
            output = model(cuda_batch)

        assert output.device.type == "cuda"
        assert output.shape == (2, 1)


# ── Explainability Tests ──────────────────────────────────────────────────────

class TestExplainability:
    def test_gradcam_generation(self, sample_rgb_batch):
        model = DeepVisionFusionModel(spatial_pretrained=False)
        gradcam = GradCAM(model)

        single_image = sample_rgb_batch[0:1]
        heatmap = gradcam.generate_heatmap(single_image)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224), f"Expected heatmap shape (224, 224), got {heatmap.shape}"
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

        # Test overlay
        raw_rgb = (single_image.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        overlay = overlay_heatmap(raw_rgb, heatmap)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

        gradcam.remove_hooks()

    def test_log_magnitude_spectrum_extraction(self, sample_rgb_batch):
        spec = compute_log_magnitude_spectrum(sample_rgb_batch[0])
        assert isinstance(spec, np.ndarray)
        assert spec.shape == (224, 224)


# ── Training & Evaluation Utility Tests ────────────────────────────────────────

class TestTrainingUtilities:
    def test_parameter_group_partitioning(self):
        model = DeepVisionFusionModel(spatial_pretrained=False)
        groups = build_parameter_groups(model, lr_backbone=1e-5, lr_head=5e-4)

        assert len(groups) == 2, "Expected 2 parameter groups (backbone vs scratch)"
        assert groups[0]["name"] == "spatial_backbone"
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["name"] == "scratch_layers"
        assert groups[1]["lr"] == 5e-4

    def test_roc_auc_metric_computation(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        auc = compute_roc_auc(y_true, y_prob)
        assert auc == 1.0, f"Expected perfect AUC of 1.0, got {auc}"

    def test_compute_metrics_dictionary(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["tp"] == 2
        assert metrics["tn"] == 2
