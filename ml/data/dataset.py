import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

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
        split:                 One of "train", "val", "test", a collection of split names,
                               or None to include all splits.
        transform:             Optional torchvision transforms.Compose. If None, uses
                               standard ImageNet resize + normalisation.
        generators:            Optional iterable of generator names to include.
        exclude_generators:    Optional iterable of generator names to exclude.
        manifest_path:         Optional path to manifest CSV (defaults to config.MANIFEST_PATH).
        exclude_cross_gen_dups: Whether to filter out cross-generator duplicates.
    """

    def __init__(
        self,
        split: Optional[Union[str, List[str], set]] = "train",
        transform: Optional[transforms.Compose] = None,
        generators: Optional[Union[List[str], set]] = None,
        exclude_generators: Optional[Union[List[str], set]] = None,
        manifest_path: Optional[Path] = None,
        exclude_cross_gen_dups: bool = False,
    ) -> None:
        super().__init__()
        self.split = split
        self.generators = set(generators) if generators is not None else None
        self.exclude_generators = set(exclude_generators) if exclude_generators is not None else None
        self.manifest_path = manifest_path or config.MANIFEST_PATH
        self.exclude_cross_gen_dups = exclude_cross_gen_dups

        self.records = self._load_manifest()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.TARGET_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ])
        else:
            self.transform = transform

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_manifest(self) -> List[Dict]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {self.manifest_path}. Run prepare.py first."
            )
        target_splits = None
        if self.split is not None:
            if isinstance(self.split, str):
                target_splits = {self.split}
            else:
                target_splits = set(self.split)

        records: List[Dict] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if target_splits is not None and row["split"] not in target_splits:
                    continue
                gen = row.get("generator")
                if self.generators is not None and gen not in self.generators:
                    continue
                if self.exclude_generators is not None and gen in self.exclude_generators:
                    continue
                if self.exclude_cross_gen_dups and row.get("is_cross_gen_dup") == "True":
                    continue
                records.append(row)
        return records

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        record     = self.records[index]
        image_path = Path(record["image_path"])

        if not image_path.exists():
            path_str = str(image_path).replace("\\", "/")
            if "genimage/" in path_str:
                rel_suffix = path_str.split("genimage/", 1)[1]
                candidate = config.GENIMAGE_DIR / rel_suffix
                if candidate.exists():
                    image_path = candidate
            elif not image_path.is_absolute():
                candidate = config.DATA_DIR / image_path
                if candidate.exists():
                    image_path = candidate

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
