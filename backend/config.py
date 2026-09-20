"""
Configuration management for DeepVision-Forensics Backend.
"""

import os
from pathlib import Path
from typing import List, Set
import torch

# Project Root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model configuration
DEFAULT_CHECKPOINT_REL = "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
MODEL_CHECKPOINT_ENV = os.getenv("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_REL)

# Resolve path relative to PROJECT_ROOT if not absolute
MODEL_CHECKPOINT_PATH = (
    Path(MODEL_CHECKPOINT_ENV)
    if Path(MODEL_CHECKPOINT_ENV).is_absolute()
    else PROJECT_ROOT / MODEL_CHECKPOINT_ENV
)

# Device configuration (CUDA auto-detection with explicit CPU fallback)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

# Upload & Validation Constraints
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/octet-stream",  # Fallback for multipart streams
}

# Inference & Classification Constants
CLASSIFICATION_THRESHOLD: float = 0.50
MODEL_NAME: str = "DeepVision-E6-C Multi-View Forensics"
BACKBONE_NAME: str = "EfficientNet-B3 + Frequency CNN"

# Multi-model catalog and selection configuration
DEFAULT_MODEL_KEY: str = os.getenv("DEFAULT_MODEL_KEY", "e6c_multiview")

SUPPORTED_MODELS = {
    "e6c_multiview": {
        "id": "e6c_multiview",
        "name": "DeepVision-E6-C Multi-View Forensics",
        "experiment": "E6-C",
        "backbone": "EfficientNet-B3 + Frequency CNN",
        "freq_norm_strategy": "standardize",
        "checkpoint_rel": "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt",
        "checkpoint_path": PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt",
        "description": "Production model: 5-view multi-scale spatial + frequency fusion with learned attention aggregation",
        "benchmark_unseen_auc": 0.9936,
        "benchmark_acc": 0.9616,
        "num_views": 5,
    },
    "e5_generalization": {
        "id": "e5_generalization",
        "name": "DeepVision-E5-Generalization",
        "experiment": "E5",
        "backbone": "EfficientNet-B3 + 4-Block Spectral CNN (Standardized)",
        "freq_norm_strategy": "standardize",
        "checkpoint_rel": "experiments/e5_generalization/best_model.pt",
        "checkpoint_path": PROJECT_ROOT / "experiments/e5_generalization/best_model.pt",
        "description": "Dual-branch spatial + frequency fusion trained for modern generator generalization (Single View)",
        "benchmark_unseen_auc": 0.9820,
        "benchmark_acc": 0.9276,
    },
    "e1_spatial": {
        "id": "e1_spatial",
        "name": "DeepVision-E1-Spatial",
        "experiment": "E1",
        "backbone": "EfficientNet-B3",
        "freq_norm_strategy": "minmax",
        "checkpoint_rel": "experiments/e1_spatial/best_model.pt",
        "checkpoint_path": PROJECT_ROOT / "experiments/e1_spatial/best_model.pt",
        "description": "Spatial-only baseline (EfficientNet-B3 backbone)",
        "benchmark_unseen_auc": 0.8991,
        "benchmark_acc": 0.8131,
    },
    "e3_std": {
        "id": "e3_std",
        "name": "DeepVision-E3-Std",
        "experiment": "E3",
        "backbone": "EfficientNet-B3 + 4-Block Spectral CNN (Standardized)",
        "freq_norm_strategy": "standardize",
        "checkpoint_rel": "experiments/candidate_standardize/best_model.pt",
        "checkpoint_path": PROJECT_ROOT / "experiments/candidate_standardize/best_model.pt",
        "description": "Dual-branch spatial + frequency fusion with standardized spectrum (GenImage holdout)",
        "benchmark_unseen_auc": 0.8959,
        "benchmark_acc": 0.7552,
    },
}

# API Routing
API_V1_PREFIX: str = "/api/v1"
CORS_ORIGINS: List[str] = ["*"]
