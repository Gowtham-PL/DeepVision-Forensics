"""
fixtures.py — Create a minimal synthetic dataset in tests/test_data/
that mirrors the real data/genimage/ structure.

Layout created
──────────────
test_data/
  genimage/
    imagenet_ai_0508_adm/      ← TRAIN generator
      train/ai/     (5 images)
      train/nature/ (5 images — image_0.JPEG is byte-identical to BigGAN's dup)
      val/ai/       (3 images)
      val/nature/   (3 images)
    imagenet_glide/            ← TRAIN generator
      train/ai/     (5 images)
      train/nature/ (5 images)
      val/ai/       (3 images)
      val/nature/   (3 images)
    imagenet_ai_0419_biggan/   ← TEST generator
      train/ai/     (4 images)
      train/nature/ (4 images — dup_0.JPEG is byte-identical to ADM's image_0)
      val/ai/       (2 images)
      val/nature/   (2 images)

Totals (used by tests)
──────────────────────
  manifest split=train  : ADM.train/* + GLIDE.train/*  = 10 + 10 = 20
  manifest split=val    : ADM.val/*   + GLIDE.val/*    =  6 +  6 = 12
  manifest split=test   : BigGAN.(train+val)/*         = 12 images
  TOTAL                 : 44 records
  Cross-gen dups flagged: 2  (ADM train/nature/image_0 ≡ BigGAN train/nature/dup_0)

Run directly to regenerate fixtures:
    python tests/fixtures.py
"""

import random
import shutil
from pathlib import Path

from PIL import Image

BASE_DIR      = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = BASE_DIR / "tests" / "test_data"
GENIMAGE_DIR  = TEST_DATA_DIR / "genimage"

# Generator folder → (canonical_name, is_test_gen, counts per cell)
# counts = (train_ai, train_nature, val_ai, val_nature)
FIXTURE_GENERATORS: dict[str, tuple[str, bool, tuple[int, int, int, int]]] = {
    "imagenet_ai_0508_adm":    ("ADM",    False, (5, 5, 3, 3)),
    "imagenet_glide":          ("GLIDE",  False, (5, 5, 3, 3)),
    "imagenet_ai_0419_biggan": ("BigGAN", True,  (4, 4, 2, 2)),
}

# Pixel colour used for the shared cross-gen duplicate
_DUP_COLOUR = (42, 42, 42)


def _make_image(path: Path, colour: tuple[int, int, int] | None = None) -> None:
    """Write a tiny 32×32 RGB JPEG."""
    if colour is None:
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new("RGB", (32, 32), color=colour)
    img.save(path, format="JPEG")


def create_fixtures() -> None:
    random.seed(0)

    if GENIMAGE_DIR.exists():
        shutil.rmtree(GENIMAGE_DIR)
    GENIMAGE_DIR.mkdir(parents=True)

    for folder_name, (canonical, is_test, counts) in FIXTURE_GENERATORS.items():
        train_ai, train_nature, val_ai, val_nature = counts
        gen_dir = GENIMAGE_DIR / folder_name

        cells = [
            ("train", "ai",     train_ai),
            ("train", "nature", train_nature),
            ("val",   "ai",     val_ai),
            ("val",   "nature", val_nature),
        ]

        for orig_split, label_folder, n in cells:
            leaf = gen_dir / orig_split / label_folder
            leaf.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                fname = f"image_{i}.JPEG"
                # ADM train/nature/image_0  →  cross-gen duplicate with BigGAN
                if canonical == "ADM" and orig_split == "train" and label_folder == "nature" and i == 0:
                    _make_image(leaf / fname, colour=_DUP_COLOUR)
                else:
                    _make_image(leaf / fname)

        # Plant the BigGAN duplicate  (byte-identical to ADM train/nature/image_0)
        if canonical == "BigGAN":
            dup_path = gen_dir / "train" / "nature" / "dup_0.JPEG"
            _make_image(dup_path, colour=_DUP_COLOUR)

    print(f"Fixtures created in {GENIMAGE_DIR}")
    _print_summary()


def _print_summary() -> None:
    total = 0
    for folder_name in FIXTURE_GENERATORS:
        gen_dir = GENIMAGE_DIR / folder_name
        count = sum(1 for f in gen_dir.rglob("*") if f.is_file())
        total += count
        print(f"  {folder_name}: {count} images")
    print(f"  TOTAL: {total} images")


if __name__ == "__main__":
    create_fixtures()
