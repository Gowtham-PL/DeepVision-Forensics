import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
# Override with DEEPVISION_DATA_DIR env variable for testing or alternate mounts.
DATA_DIR      = Path(os.environ.get("DEEPVISION_DATA_DIR", BASE_DIR / "data"))
GENIMAGE_DIR  = DATA_DIR / "genimage"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_NATURE = 0   # natural / real photographs  (folder: nature/)
LABEL_AI     = 1   # AI-generated images          (folder: ai/)

# Backwards-compatible aliases (do not use in new code)
LABEL_REAL         = LABEL_NATURE
LABEL_AI_GENERATED = LABEL_AI

# ── Supported file extensions ──────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Generator → canonical name mapping ────────────────────────────────────────
# Maps each folder name inside data/genimage/ to a short, human-readable label.
# Add new generator folders here if the dataset is extended.
GENERATOR_FOLDER_MAP: dict[str, str] = {
    "imagenet_ai_0508_adm":    "ADM",
    "imagenet_ai_0419_vqdm":   "VQDM",
    "imagenet_ai_0424_sdv5":   "SDv5",
    "imagenet_ai_0424_wukong": "Wukong",
    "imagenet_glide":          "GLIDE",
    "imagenet_ai_0419_biggan": "BigGAN",
    "imagenet_midjourney":     "Midjourney",
}

# ── Generator-level holdout split ─────────────────────────────────────────────
#
# TRAIN_GENERATORS  →  used for training and validation.
#     original train/ subdir  →  logical split "train"
#     original val/   subdir  →  logical split "val"
#
# TEST_GENERATORS   →  completely held-out, never seen during train/val.
#     ALL images (both train/ and val/ subdirs)  →  logical split "test"
#
# This strategy tests cross-generator generalization: the model must detect
# AI-generated images from generators it has never encountered before.
TRAIN_GENERATORS: set[str] = {"ADM", "VQDM", "SDv5", "Wukong", "GLIDE"}
TEST_GENERATORS:  set[str] = {"BigGAN", "Midjourney"}

# ── Image processing ───────────────────────────────────────────────────────────
TARGET_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]

# ── DataLoader ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 4
