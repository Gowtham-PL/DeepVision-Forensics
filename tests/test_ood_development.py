"""
Tests for Leave-One-Generator-Out (LOGO) cross-validation, frequency normalization ablation,
and strict leakage prevention.
"""

import os
import hashlib
from pathlib import Path
import pytest
import torch

from ml.data import config
from ml.data.dataset import DeepVisionDataset
from ml.ood_development import (
    DEV_GENERATORS,
    FORBIDDEN_TEST_GENERATORS,
    NORM_STRATEGIES,
    assert_zero_test_leakage,
    get_logo_folds,
)
from models.fusion import build_model


class TestLOGOFoldIntegrity:
    """Validates the mathematical correctness and isolation of LOGO folds."""

    def test_all_five_generators_used_exactly_once(self):
        folds = get_logo_folds(DEV_GENERATORS)
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

        held_out_set = {f["held_out_generator"] for f in folds}
        assert held_out_set == set(DEV_GENERATORS), (
            f"Folds must cover all 5 dev generators. Got {held_out_set}"
        )

    def test_four_training_generators_per_fold(self):
        folds = get_logo_folds(DEV_GENERATORS)
        for fold in folds:
            held_out = fold["held_out_generator"]
            train_gens = fold["train_generators"]
            assert len(train_gens) == 4, f"Fold {held_out} must have 4 train generators"
            assert held_out not in train_gens, f"Held-out {held_out} leaked into train generators {train_gens}"
            assert set(train_gens) == set(DEV_GENERATORS) - {held_out}

    def test_fold_assignments_are_deterministic(self):
        folds1 = get_logo_folds(DEV_GENERATORS)
        folds2 = get_logo_folds(DEV_GENERATORS)
        assert folds1 == folds2, "Fold assignment is non-deterministic"

    def test_forbidden_generators_raise_error(self):
        with pytest.raises(ValueError):
            get_logo_folds(["ADM", "GLIDE", "BigGAN"])

        with pytest.raises(ValueError):
            get_logo_folds(["Midjourney", "SDv5"])


class TestZeroLeakageEnforcement:
    """Verifies that no test generators ever leak into OOD development."""

    def test_assert_zero_test_leakage_clean(self):
        clean_records = [
            {"generator": "ADM", "label": 0},
            {"generator": "GLIDE", "label": 1},
            {"generator": "SDv5", "label": 0},
        ]
        # Should not raise
        assert_zero_test_leakage(clean_records, "Clean Test Records")

    def test_assert_zero_test_leakage_detects_biggan(self):
        tainted_records = [
            {"generator": "ADM", "label": 0},
            {"generator": "BigGAN", "label": 1},
        ]
        with pytest.raises(RuntimeError) as exc_info:
            assert_zero_test_leakage(tainted_records, "Tainted Records")
        assert "BigGAN" in str(exc_info.value)

    def test_assert_zero_test_leakage_detects_midjourney(self):
        tainted_records = [
            {"generator": "Midjourney", "label": 1},
            {"generator": "Wukong", "label": 0},
        ]
        with pytest.raises(RuntimeError) as exc_info:
            assert_zero_test_leakage(tainted_records, "Tainted Records")
        assert "Midjourney" in str(exc_info.value)

    def test_dataset_generator_filtering_isolation(self):
        """Verifies DeepVisionDataset respects generator filters and excludes held-out generator."""
        manifest_path = Path("data/manifest.csv")
        if not manifest_path.exists():
            manifest_path = config.MANIFEST_PATH
        if not manifest_path.exists():
            pytest.skip("Manifest not found on disk.")

        # Read available generators in this manifest
        raw_ds = DeepVisionDataset(split=None, manifest_path=manifest_path)
        manifest_gens = {r.get("generator") for r in raw_ds.records}
        testable_dev_gens = [g for g in DEV_GENERATORS if g in manifest_gens]

        for held_out in testable_dev_gens:
            train_gens = [g for g in testable_dev_gens if g != held_out]
            ds_train = DeepVisionDataset(split="train", generators=train_gens, manifest_path=manifest_path)

            # Assert held-out generator is absent
            gens_in_train = {r["generator"] for r in ds_train.records}
            assert held_out not in gens_in_train, (
                f"Held-out generator {held_out} was found in training split records!"
            )
            # Assert test generators are absent
            assert "BigGAN" not in gens_in_train
            assert "Midjourney" not in gens_in_train

            # Assert validation split contains ONLY held-out generator
            ds_val = DeepVisionDataset(split="val", generators=[held_out], manifest_path=manifest_path)
            gens_in_val = {r["generator"] for r in ds_val.records}
            assert gens_in_val == {held_out}, (
                f"Validation split must contain only {held_out}, got {gens_in_val}"
            )


class TestFrequencyNormalizationAblationModels:
    """Verifies all 3 normalization candidates instantiate and execute without errors."""

    @pytest.mark.parametrize("strategy", ["minmax", "standardize", "none"])
    def test_fusion_model_forward_pass_all_strategies(self, strategy: str):
        model = build_model(
            experiment="E3",
            pretrained=False,
            freq_norm_strategy=strategy,
            freq_embedding_dim=256,
        )
        model.eval()

        dummy_x = torch.rand(2, 3, 224, 224)
        out = model(dummy_x)
        assert out.shape == (2, 1), f"Expected shape (2, 1), got {out.shape}"
        assert not torch.isnan(out).any(), f"NaNs detected with strategy {strategy}"

    @pytest.mark.parametrize("strategy", ["minmax", "standardize", "none"])
    def test_fusion_model_return_features_all_strategies(self, strategy: str):
        model = build_model(
            experiment="E3",
            pretrained=False,
            freq_norm_strategy=strategy,
            freq_embedding_dim=256,
        )
        model.eval()

        dummy_x = torch.rand(2, 3, 224, 224)
        features = model(dummy_x, return_features=True)
        assert "logit" in features
        assert "spatial_embedding" in features
        assert "frequency_embedding" in features
        assert "fused_embedding" in features
        assert "fft_spectrum" in features
        assert features["spatial_embedding"].shape == (2, 1536)
        assert features["frequency_embedding"].shape == (2, 256)
        assert features["fused_embedding"].shape == (2, 1792)


class TestFrozenBaselineProtection:
    """Verifies that original frozen checkpoints and evaluation artifacts remain intact."""

    def test_e1_checkpoint_intact(self):
        ckpt_path = Path("experiments/e1_spatial/best_model.pt")
        assert ckpt_path.exists(), "E1 checkpoint missing"
        assert ckpt_path.stat().st_size > 100 * 1024 * 1024, "E1 checkpoint corrupted or truncated"

    def test_e2_checkpoint_intact(self):
        ckpt_path = Path("experiments/e2_frequency/best_model.pt")
        assert ckpt_path.exists(), "E2 checkpoint missing"
        assert ckpt_path.stat().st_size > 5 * 1024 * 1024, "E2 checkpoint corrupted or truncated"

    def test_e3_checkpoint_intact(self):
        ckpt_path = Path("experiments/e3_dual_domain/best_model.pt")
        assert ckpt_path.exists(), "E3 checkpoint missing"
        assert ckpt_path.stat().st_size > 100 * 1024 * 1024, "E3 checkpoint corrupted or truncated"

    def test_final_evaluation_results_intact(self):
        eval_path = Path("experiments/final_evaluation/evaluation_results.json")
        assert eval_path.exists(), "Final evaluation results missing"
        summary_path = Path("experiments/final_evaluation/summary_table.csv")
        assert summary_path.exists(), "Final evaluation summary table missing"
