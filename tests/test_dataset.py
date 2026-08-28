"""
test_dataset.py — Integration tests for the dataset pipeline.

Uses a synthetic genimage-style fixture dataset (see tests/fixtures.py).
Fixture layout (all counts are exact):
    imagenet_ai_0508_adm   (TRAIN gen): train/ai=5  train/nature=5  val/ai=3  val/nature=3  → 16 images
    imagenet_glide         (TRAIN gen): train/ai=5  train/nature=5  val/ai=3  val/nature=3  → 16 images
    imagenet_ai_0419_biggan (TEST gen): train/ai=4  train/nature=4  val/ai=2  val/nature=2  → 12 images
                                        + 1 duplicate of ADM train/nature/image_0           →  1 extra

    manifest split=train  : 20 images   (ADM.train/* + GLIDE.train/*)
    manifest split=val    : 12 images   (ADM.val/*   + GLIDE.val/*)
    manifest split=test   : 13 images   (BigGAN.train/* + BigGAN.val/*)  ← includes the dup
    TOTAL manifest rows   : 45
    Cross-gen dup records : 2 (ADM train/nature/image_0 ≡ BigGAN train/nature/dup_0)
"""

import csv
import os
from pathlib import Path

import pytest

# ── Point config at the test data directory BEFORE importing any ml modules ──
TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
os.environ["DEEPVISION_DATA_DIR"] = str(TEST_DATA_DIR)

from ml.data import config                          # noqa: E402
from ml.data.prepare import build_manifest          # noqa: E402
from ml.data.dataset import DeepVisionDataset       # noqa: E402
from ml.data.dataloaders import get_dataloader      # noqa: E402
from tests.fixtures import (                        # noqa: E402
    create_fixtures,
    FIXTURE_GENERATORS,
    GENIMAGE_DIR as FIXTURE_GENIMAGE_DIR,
)

# ── Fixture generator metadata (derived from fixtures.py) ────────────────────
_TRAIN_GEN_FOLDERS = {k for k, (_, is_test, _) in FIXTURE_GENERATORS.items() if not is_test}
_TEST_GEN_FOLDERS  = {k for k, (_, is_test, _) in FIXTURE_GENERATORS.items() if is_test}

_TRAIN_GEN_CANONICAL = {v for _, (v, is_test, _) in FIXTURE_GENERATORS.items() if not is_test}
_TEST_GEN_CANONICAL  = {v for _, (v, is_test, _) in FIXTURE_GENERATORS.items() if is_test}

# Folder map restricted to fixture generators
_FIXTURE_FOLDER_MAP = {k: v for k, (v, _, _) in FIXTURE_GENERATORS.items()}

# Expected counts from fixtures.py docstring
_EXPECTED_TRAIN = 20
_EXPECTED_VAL   = 12
_EXPECTED_TEST  = 13   # 12 + 1 duplicate file
_EXPECTED_TOTAL = _EXPECTED_TRAIN + _EXPECTED_VAL + _EXPECTED_TEST   # 45
_EXPECTED_DUPS  = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_manifest() -> list[dict]:
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def setup_fixtures():
    """Create synthetic images and build the manifest once per test module."""
    create_fixtures()
    build_manifest(
        genimage_dir         = FIXTURE_GENIMAGE_DIR,
        train_generators     = _TRAIN_GEN_CANONICAL,
        test_generators      = _TEST_GEN_CANONICAL,
        generator_folder_map = _FIXTURE_FOLDER_MAP,
        manifest_path        = config.MANIFEST_PATH,
    )
    yield
    # Teardown is intentionally skipped — leaves fixtures for inspection.


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestManifestCreation:
    def test_manifest_exists(self):
        assert config.MANIFEST_PATH.exists(), "Manifest CSV was not created"

    def test_total_record_count(self):
        rows = _read_manifest()
        assert len(rows) == _EXPECTED_TOTAL, (
            f"Expected {_EXPECTED_TOTAL} records, got {len(rows)}"
        )

    def test_split_counts(self):
        from collections import Counter
        rows = _read_manifest()
        counts = Counter(r["split"] for r in rows)
        assert counts["train"] == _EXPECTED_TRAIN, f"train={counts['train']}, expected {_EXPECTED_TRAIN}"
        assert counts["val"]   == _EXPECTED_VAL,   f"val={counts['val']}, expected {_EXPECTED_VAL}"
        assert counts["test"]  == _EXPECTED_TEST,  f"test={counts['test']}, expected {_EXPECTED_TEST}"

    def test_train_val_generators_are_correct(self):
        rows = _read_manifest()
        for row in rows:
            if row["split"] in ("train", "val"):
                assert row["generator"] in _TRAIN_GEN_CANONICAL, (
                    f"Generator {row['generator']} must not appear in train/val"
                )

    def test_test_generators_are_correct(self):
        rows = _read_manifest()
        for row in rows:
            if row["split"] == "test":
                assert row["generator"] in _TEST_GEN_CANONICAL, (
                    f"Generator {row['generator']} must not appear in test"
                )

    def test_no_test_generator_in_train_or_val(self):
        rows = _read_manifest()
        leakage = [
            r for r in rows
            if r["split"] in ("train", "val") and r["generator"] in _TEST_GEN_CANONICAL
        ]
        assert len(leakage) == 0, f"Generator leakage: {leakage}"

    def test_label_values(self):
        rows = _read_manifest()
        for row in rows:
            assert row["label"] in ("0", "1"), f"Unexpected label: {row['label']}"
            assert row["class_name"] in ("ai", "nature"), f"Unexpected class_name: {row['class_name']}"

    def test_label_consistency(self):
        """class_name and label must agree."""
        rows = _read_manifest()
        for row in rows:
            if row["class_name"] == "nature":
                assert row["label"] == "0", f"nature image has label {row['label']}"
            elif row["class_name"] == "ai":
                assert row["label"] == "1", f"ai image has label {row['label']}"

    def test_cross_gen_dup_count(self):
        rows = _read_manifest()
        dup_rows = [r for r in rows if r["is_cross_gen_dup"] == "True"]
        assert len(dup_rows) == _EXPECTED_DUPS, (
            f"Expected {_EXPECTED_DUPS} dup records, got {len(dup_rows)}"
        )

    def test_split_is_balanced(self):
        """
        train and val must be exactly balanced (nature == ai).
        test split carries +1 nature from the planted cross-gen duplicate,
        mirroring the real dataset (Midjourney duplicate adds 1 extra nature).
        """
        from collections import Counter
        rows = _read_manifest()
        for split in ("train", "val"):
            split_rows = [r for r in rows if r["split"] == split]
            label_counts = Counter(r["class_name"] for r in split_rows)
            assert label_counts["nature"] == label_counts["ai"], (
                f"Split '{split}' is imbalanced: nature={label_counts['nature']} "
                f"ai={label_counts['ai']}"
            )
        # test: BigGAN train has 4 ai + (3 unique + 1 dup) nature; val 2+2
        test_rows   = [r for r in rows if r["split"] == "test"]
        test_counts = Counter(r["class_name"] for r in test_rows)
        assert test_counts["ai"]     == 6, f"test ai expected 6, got {test_counts['ai']}"
        assert test_counts["nature"] == 7, f"test nature expected 7, got {test_counts['nature']}"

    def test_orig_split_values(self):
        rows = _read_manifest()
        for row in rows:
            assert row["orig_split"] in ("train", "val"), (
                f"Unexpected orig_split: {row['orig_split']}"
            )

    def test_required_columns_present(self):
        rows = _read_manifest()
        required = {
            "image_path", "label", "class_name", "generator",
            "orig_split", "split", "width", "height", "format", "is_cross_gen_dup",
        }
        assert required.issubset(rows[0].keys()), (
            f"Missing columns: {required - rows[0].keys()}"
        )


class TestDatasetLoading:
    def test_train_dataset_length(self):
        ds = DeepVisionDataset(split="train")
        assert len(ds) == _EXPECTED_TRAIN, (
            f"Expected train size {_EXPECTED_TRAIN}, got {len(ds)}"
        )

    def test_val_dataset_length(self):
        ds = DeepVisionDataset(split="val")
        assert len(ds) == _EXPECTED_VAL, (
            f"Expected val size {_EXPECTED_VAL}, got {len(ds)}"
        )

    def test_test_dataset_length(self):
        ds = DeepVisionDataset(split="test")
        assert len(ds) == _EXPECTED_TEST, (
            f"Expected test size {_EXPECTED_TEST}, got {len(ds)}"
        )

    def test_item_shape_and_label(self):
        ds = DeepVisionDataset(split="train")
        img_tensor, label = ds[0]
        assert img_tensor.shape == (3, 224, 224), (
            f"Tensor shape mismatch: {img_tensor.shape}"
        )
        assert label in (0, 1), f"Invalid label: {label}"

    def test_get_record_metadata(self):
        ds = DeepVisionDataset(split="train")
        rec = ds.get_record(0)
        assert "generator"       in rec
        assert "orig_split"      in rec
        assert "is_cross_gen_dup" in rec


class TestDataloaders:
    def test_train_dataloader_batch(self):
        loader = get_dataloader("train", batch_size=4, num_workers=0)
        images, labels = next(iter(loader))
        assert images.shape == (4, 3, 224, 224), f"Batch shape: {images.shape}"
        assert labels.shape == (4,), f"Labels shape: {labels.shape}"

    def test_val_dataloader_not_shuffled(self):
        """Calling twice with same seed should yield identical first batches."""
        loader1 = get_dataloader("val", batch_size=4, num_workers=0)
        loader2 = get_dataloader("val", batch_size=4, num_workers=0)
        _, labels1 = next(iter(loader1))
        _, labels2 = next(iter(loader2))
        import torch
        assert torch.equal(labels1, labels2), "val loader is shuffled (should not be)"
