import os
import pytest
from pathlib import Path

# Override config DATA_DIR to use test_data BEFORE importing other modules
os.environ["DEEPVISION_DATA_DIR"] = str(Path(__file__).resolve().parent / "test_data")

from ml.data import config
from ml.data.prepare import discover_and_split
from ml.data.dataset import DeepVisionDataset
from ml.data.dataloaders import get_dataloader

def test_manifest_creation():
    discover_and_split()
    assert config.MANIFEST_PATH.exists(), "Manifest was not created"
    
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # 100 unique valid images + 1 header
    assert len(lines) == 101, f"Expected 101 lines in manifest (header + 100 records), got {len(lines)}"
    
    # Check splits exist
    content = "".join(lines)
    assert "train" in content
    assert "val" in content
    assert "test" in content
    assert "midjourney" in content

def test_dataset_loading():
    train_dataset = DeepVisionDataset(split="train")
    
    # Should be 70% of 100 = 70
    assert len(train_dataset) == 70
    
    # Get one item
    img_tensor, label = train_dataset[0]
    
    assert img_tensor.shape == (3, 224, 224), "Tensor shape mismatch"
    assert label in [0, 1], "Invalid label"
    
def test_dataloaders():
    train_loader = get_dataloader("train", batch_size=8, num_workers=0)
    
    batch = next(iter(train_loader))
    images, labels = batch
    
    assert images.shape == (8, 3, 224, 224), "Batch image shape mismatch"
    assert labels.shape == (8,), "Batch label shape mismatch"
