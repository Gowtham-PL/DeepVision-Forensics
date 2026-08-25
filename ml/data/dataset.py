import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from ml.data import config


class DeepVisionDataset(Dataset):
    def __init__(self, split: str = "train", transform: Optional[transforms.Compose] = None):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: Optional torchvision transforms. If None, uses standard RGB preprocessing.
        """
        super().__init__()
        self.split = split
        self.records = self._load_manifest(split)
        
        # Default standard RGB preprocessing if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.TARGET_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
            ])
        else:
            self.transform = transform

    def _load_manifest(self, target_split: str) -> List[Dict]:
        records = []
        if not config.MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Manifest not found at {config.MANIFEST_PATH}. Run prepare.py first.")
            
        with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == target_split:
                    records.append(row)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = Path(record["image_path"])
        
        # Open and ensure RGB
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            
        tensor_image = self.transform(image)
        if not isinstance(tensor_image, torch.Tensor):
            raise TypeError("Expected transform to return a torch.Tensor")
            
        label = int(record["label"])
        return tensor_image, label
