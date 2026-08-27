"""
prepare.py — Build the dataset manifest CSV.

Generator-level holdout split strategy
──────────────────────────────────────
TRAIN_GENERATORS  (ADM, VQDM, SDv5, Wukong, GLIDE):
    original train/ subdir  →  logical split "train"
    original val/   subdir  →  logical split "val"

TEST_GENERATORS  (BigGAN, Midjourney):
    ALL images (train/ + val/)  →  logical split "test"
    Never exposed during training or validation.

No images are moved, copied, resized, or otherwise modified.
The manifest alone encodes the logical split.

Manifest columns
────────────────
image_path      Absolute path to the image file.
label           0 = nature (real), 1 = ai (generated).
class_name      "nature" or "ai" (mirrors the source folder name).
generator       Canonical generator name (ADM, VQDM, …).
orig_split      Original dataset split folder: "train" or "val".
split           Logical split assigned by this strategy: "train", "val", or "test".
width           Image width in pixels.
height          Image height in pixels.
format          PIL format string (JPEG, PNG, …).
is_cross_gen_dup  True if the image has an MD5-identical copy in another generator.
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ml.data import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def calculate_hash(filepath: Path) -> str:
    """MD5 hash of a file — used for cross-generator duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def is_valid_image(filepath: Path) -> bool:
    """Return True if the file has a supported extension and passes PIL verification."""
    if filepath.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, IOError):
        return False


def get_image_info(filepath: Path):
    """Return (width, height, format) or (None, None, None) on failure."""
    try:
        with Image.open(filepath) as img:
            return img.width, img.height, img.format
    except Exception:
        return None, None, None


# ── Main entry point ───────────────────────────────────────────────────────────

def build_manifest(
    genimage_dir: Path | None = None,
    train_generators: set | None = None,
    test_generators: set | None = None,
    generator_folder_map: dict | None = None,
    manifest_path: Path | None = None,
) -> None:
    """
    Scan data/genimage/ and write a manifest CSV encoding the generator-level
    holdout split.  Accepts keyword overrides so tests can inject a smaller
    synthetic dataset without touching config.

    Args:
        genimage_dir:         Root of the genimage dataset (default: config.GENIMAGE_DIR).
        train_generators:     Set of canonical generator names used for train/val.
        test_generators:      Set of canonical generator names held out for test.
        generator_folder_map: Dict mapping folder names → canonical names.
        manifest_path:        Output CSV path (default: config.MANIFEST_PATH).
    """
    genimage_dir         = genimage_dir         or config.GENIMAGE_DIR
    train_generators     = train_generators     or config.TRAIN_GENERATORS
    test_generators      = test_generators      or config.TEST_GENERATORS
    generator_folder_map = generator_folder_map or config.GENERATOR_FOLDER_MAP
    manifest_path        = manifest_path        or config.MANIFEST_PATH

    if not genimage_dir.exists():
        print(f"[ERROR] GenImage directory not found: {genimage_dir}")
        return

    all_generators = train_generators | test_generators

    print("=" * 64)
    print("BUILD MANIFEST")
    print(f"  Source      : {genimage_dir}")
    print(f"  Train/val   : {sorted(train_generators)}")
    print(f"  Test (held) : {sorted(test_generators)}")
    print("=" * 64)

    raw_records: list[dict] = []
    invalid = 0
    skipped = 0

    # ── Pass 1: collect every image with its metadata and MD5 hash ────────────
    for folder_name, gen_canonical in generator_folder_map.items():
        gen_dir = genimage_dir / folder_name

        if not gen_dir.exists():
            print(f"  [MISSING FOLDER] {folder_name} — skipping")
            continue

        if gen_canonical not in all_generators:
            print(f"  [UNREGISTERED]   {gen_canonical} — skipping")
            skipped += 1
            continue

        is_test_gen = gen_canonical in test_generators

        for orig_split in ("train", "val"):
            split_dir = gen_dir / orig_split
            if not split_dir.exists():
                print(f"  [MISSING] {folder_name}/{orig_split}")
                continue

            for label_folder in ("ai", "nature"):
                label_dir = split_dir / label_folder
                if not label_dir.exists():
                    print(f"  [MISSING] {folder_name}/{orig_split}/{label_folder}")
                    continue

                label_value   = config.LABEL_AI if label_folder == "ai" else config.LABEL_NATURE
                logical_split = "test" if is_test_gen else orig_split

                for filepath in sorted(label_dir.rglob("*")):
                    if not filepath.is_file():
                        continue
                    if filepath.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                        skipped += 1
                        continue
                    if not is_valid_image(filepath):
                        print(f"  [CORRUPT] {filepath}")
                        invalid += 1
                        continue

                    width, height, fmt = get_image_info(filepath)
                    if width is None:
                        invalid += 1
                        continue

                    raw_records.append({
                        "image_path":  str(filepath.absolute()),
                        "label":       label_value,
                        "class_name":  label_folder,
                        "generator":   gen_canonical,
                        "orig_split":  orig_split,
                        "split":       logical_split,
                        "width":       width,
                        "height":      height,
                        "format":      fmt or "",
                        "_hash":       calculate_hash(filepath),
                    })

    print(f"\nCollected {len(raw_records)} valid images "
          f"({invalid} corrupt, {skipped} unsupported/skipped).")

    if not raw_records:
        print("No records found — aborting manifest generation.")
        return

    # ── Pass 2: detect cross-generator duplicates by MD5 ─────────────────────
    hash_to_indices: dict[str, list[int]] = {}
    for i, rec in enumerate(raw_records):
        hash_to_indices.setdefault(rec["_hash"], []).append(i)

    cross_gen_dup_hashes: set[str] = set()
    for h, indices in hash_to_indices.items():
        if len(indices) > 1:
            gens = {raw_records[i]["generator"] for i in indices}
            if len(gens) > 1:          # same bytes, different generators
                cross_gen_dup_hashes.add(h)
                paths = [raw_records[i]["image_path"] for i in indices]
                print(f"  [CROSS-GEN DUP] hash={h}")
                for p in paths:
                    print(f"      {p}")

    # ── Build final records and write manifest ────────────────────────────────
    fieldnames = [
        "image_path", "label", "class_name", "generator",
        "orig_split", "split", "width", "height", "format", "is_cross_gen_dup",
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in raw_records:
            h = rec.pop("_hash")
            rec["is_cross_gen_dup"] = str(h in cross_gen_dup_hashes)
            writer.writerow(rec)

    # ── Summary ───────────────────────────────────────────────────────────────
    split_counts = Counter(r["split"]      for r in raw_records)
    gen_counts   = Counter(r["generator"]  for r in raw_records)
    label_counts = Counter(r["class_name"] for r in raw_records)
    dup_count    = sum(1 for r in raw_records if r["is_cross_gen_dup"] == "True")

    print(f"\nManifest written -> {manifest_path}")
    print(f"  Total records  : {len(raw_records)}")
    print(f"  Splits         : train={split_counts['train']}  "
          f"val={split_counts['val']}  test={split_counts['test']}")
    print(f"  Labels         : nature={label_counts['nature']}  "
          f"ai={label_counts['ai']}")
    print(f"  Dup records    : {dup_count}")
    print("  Per generator  :")
    for gen in sorted(gen_counts):
        print(f"    {gen:<14}: {gen_counts[gen]}")


def discover_and_split() -> None:
    """Deprecated alias -> build_manifest().  Kept for backwards compatibility."""
    build_manifest()


if __name__ == "__main__":
    build_manifest()
