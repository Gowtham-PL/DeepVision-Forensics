import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ml.data import config


class DeepVisionDataset(Dataset):
    """
    PyTorch Dataset backed by the manifest CSV produced by prepare.py.

    Manifest columns used:
        image_path      — absolute path to the image file
        label           — 0 (nature/real) or 1 (ai/generated)
        split           — logical split: "train", "val", or "test"
        generator       — canonical generator name (informational)
        orig_split      — original dataset folder "train" or "val"
        is_cross_gen_dup — "True" / "False" (informational; not filtered here)

    Args:
        split:      One of "train", "val", or "test".
        transform:  Optional torchvision transforms.Compose.  If None, uses
                    standard ImageNet resize + normalisation.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.split   = split
        self.records = self._load_manifest(split)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.TARGET_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ])
        else:
            self.transform = transform

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_manifest(self, target_split: str) -> List[Dict]:
        if not config.MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Manifest not found at {config.MANIFEST_PATH}. Run prepare.py first."
            )
        records: List[Dict] = []
        with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == target_split:
                    records.append(row)
        return records

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record     = self.records[index]
        image_path = Path(record["image_path"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        tensor_image = self.transform(image)
        if not isinstance(tensor_image, torch.Tensor):
            raise TypeError("Expected transform to return a torch.Tensor")

        label = int(record["label"])
        return tensor_image, label

    # ── Convenience ────────────────────────────────────────────────────────────

    def get_record(self, index: int) -> Dict:
        """Return the raw manifest row dict for the given index."""
        return self.records[index]
