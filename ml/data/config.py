import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# We can override DATA_DIR with an environment variable if needed
DATA_DIR = Path(os.environ.get("DEEPVISION_DATA_DIR", BASE_DIR / "data"))
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Labels
LABEL_REAL = 0
LABEL_AI_GENERATED = 1

# Supported extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Image Processing (Configurable)
TARGET_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dataset Splitting
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# DataLoader Configuration
BATCH_SIZE = 32
NUM_WORKERS = 4
