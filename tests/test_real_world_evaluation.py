"""
Unit tests for Real-World Generalization Evaluation pipeline (ml/evaluate_real_world.py).

Tests:
1. Image discovery across supported extensions (.png, .jpg, .jpeg, .webp) and subdirectories.
2. RealWorldDataset loading and preprocessing.
3. Metric calculations (Accuracy, Precision, Recall, F1, FPR on real, FNR on AI, ROC-AUC).
4. End-to-end evaluation runner producing results.json, summary.csv, and REAL_WORLD_REPORT.md.
"""

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from ml.evaluate_real_world import (
    RealWorldDataset,
    compute_real_world_metrics,
    discover_images,
    evaluate_model_on_dataset,
    run_real_world_evaluation,
)
from models.fusion import build_model


@pytest.fixture
def temp_real_world_data(tmp_path: Path) -> Path:
    """Creates a temporary synthetic real_world_test dataset structure."""
    dataset_dir = tmp_path / "real_world_test"
    real_dir = dataset_dir / "real"
    ai_dir = dataset_dir / "ai"
    real_dir.mkdir(parents=True)
    ai_dir.mkdir(parents=True)

    # Create real images with various extensions
    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    for i, ext in enumerate(extensions):
        img = Image.new("RGB", (100 + i * 10, 100 + i * 10), color=(50 * i, 100, 150))
        img.save(real_dir / f"real_sample_{i}{ext}")

    # Subdirectory within real
    sub_real = real_dir / "camera_subfolder"
    sub_real.mkdir()
    img_sub = Image.new("RGB", (64, 64), color=(200, 200, 200))
    img_sub.save(sub_real / "sub_real_image.JPG")  # Test uppercase extension

    # Create AI images with various extensions
    for i, ext in enumerate(extensions):
        img = Image.new("RGB", (120, 120), color=(150, 50 * i, 100))
        img.save(ai_dir / f"ai_sample_{i}{ext}")

    # Create an invalid file that should be ignored
    (real_dir / "ignored_notes.txt").write_text("This is not an image", encoding="utf-8")

    return dataset_dir


class TestImageDiscovery:
    """Tests image file discovery logic."""

    def test_discover_images_finds_all_supported_types(self, temp_real_world_data: Path):
        records = discover_images(temp_real_world_data)
        # 4 top real + 1 subfolder real = 5 real; 4 ai = 4 ai; Total = 9
        assert len(records) == 9

        labels = [r["label"] for r in records]
        assert labels.count(0) == 5
        assert labels.count(1) == 4

        filenames = [r["filename"] for r in records]
        assert "ignored_notes.txt" not in filenames
        assert "sub_real_image.JPG" in filenames

    def test_discover_images_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_test"
        empty_dir.mkdir()
        records = discover_images(empty_dir)
        assert len(records) == 0


class TestRealWorldDataset:
    """Tests RealWorldDataset loading and tensor conversion."""

    def test_dataset_item_shape_and_range(self, temp_real_world_data: Path):
        records = discover_images(temp_real_world_data)
        dataset = RealWorldDataset(records)

        assert len(dataset) == 9
        tensor, label, rel_path = dataset[0]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert label in [0, 1]
        assert isinstance(rel_path, str)


class TestRealWorldMetricsComputation:
    """Tests calculation of evaluation metrics."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_real_world_metrics(y_true, y_prob, threshold=0.5)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["fpr_on_real"] == 0.0
        assert metrics["fnr_on_ai"] == 0.0
        assert metrics["tp"] == 2
        assert metrics["tn"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_mixed_predictions(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 4 real, 4 ai
        y_prob = np.array([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.2, 0.6])  # 1 FP (index 2), 1 FN (index 6)
        metrics = compute_real_world_metrics(y_true, y_prob, threshold=0.5)

        assert metrics["sample_count"] == 8
        assert metrics["n_real"] == 4
        assert metrics["n_ai"] == 4
        assert metrics["tp"] == 3
        assert metrics["tn"] == 3
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["accuracy"] == 6 / 8
        assert metrics["fpr_on_real"] == 1 / 4  # 25%
        assert metrics["fnr_on_ai"] == 1 / 4   # 25%
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_empty_input(self):
        metrics = compute_real_world_metrics(np.array([]), np.array([]))
        assert metrics == {}


class TestEndToEndRealWorldPipeline:
    """Tests full pipeline execution and file creation using synthetic models."""

    def test_evaluate_model_on_dataset(self, temp_real_world_data: Path):
        records = discover_images(temp_real_world_data)
        dataset = RealWorldDataset(records)

        # Simple linear dummy model outputting (B, 1)
        class DummyModel(nn.Module):
            def forward(self, x):
                return torch.zeros((x.size(0), 1))

        model = DummyModel()
        probs, per_image = evaluate_model_on_dataset(
            model, dataset, batch_size=4, device=torch.device("cpu")
        )

        assert len(probs) == 9
        assert len(per_image) == 9
        # Sigmoid(0) = 0.5
        assert np.allclose(probs, 0.5, atol=1e-3)
        assert per_image[0]["predicted_class"] == "ai"  # threshold >= 0.5

    def test_run_real_world_evaluation_artifacts_generation(
        self, temp_real_world_data: Path, tmp_path: Path
    ):
        # Save synthetic checkpoints for E1 and E3
        e1_ckpt_path = tmp_path / "e1_mock.pt"
        e3_ckpt_path = tmp_path / "e3_mock.pt"

        dummy_e1 = build_model(experiment="E1", pretrained=False)
        dummy_e3 = build_model(experiment="E3", pretrained=False, freq_norm_strategy="standardize")

        torch.save({
            "epoch": 1,
            "experiment": "E1",
            "model_state_dict": dummy_e1.state_dict(),
            "config": {"experiment": "E1"}
        }, e1_ckpt_path)

        torch.save({
            "epoch": 1,
            "experiment": "E3",
            "model_state_dict": dummy_e3.state_dict(),
            "config": {"experiment": "E3", "norm_strategy": "standardize", "freq_embedding_dim": 256}
        }, e3_ckpt_path)

        out_dir = tmp_path / "eval_output"
        results = run_real_world_evaluation(
            data_dir=str(temp_real_world_data),
            save_dir=str(out_dir),
            e1_checkpoint=str(e1_ckpt_path),
            e3_checkpoint=str(e3_ckpt_path),
            batch_size=4,
            device=torch.device("cpu"),
        )

        # Verify files generated
        assert (out_dir / "results.json").exists()
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "REAL_WORLD_REPORT.md").exists()

        # Verify JSON contents
        with open(out_dir / "results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "E1_Spatial" in data["models"]
        assert "E3_Std" in data["models"]
        assert data["metadata"]["total_images"] == 9

        # Verify CSV contents
        with open(out_dir / "summary.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["Model_ID"] == "E1_Spatial"
        assert rows[1]["Model_ID"] == "E3_Std"
