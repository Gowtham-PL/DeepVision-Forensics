import csv
import hashlib
import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# Adjust import so it works if run as a script
from ml.data import config

def calculate_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file for basic duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def is_valid_image(filepath: Path) -> bool:
    if filepath.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify() # verify it is an image
        return True
    except (UnidentifiedImageError, IOError):
        return False

def get_image_info(filepath: Path):
    try:
        with Image.open(filepath) as img:
            return img.width, img.height, img.format
    except Exception:
        return None, None, None

def discover_and_split():
    data_dir = config.DATA_DIR
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return

    real_dir = data_dir / "real"
    ai_dir = data_dir / "ai_generated"

    records = []
    seen_hashes = set()
    duplicates = 0
    invalid = 0

    def process_directory(directory: Path, label: int, class_name: str):
        nonlocal duplicates, invalid
        if not directory.exists():
            return
        for filepath in directory.rglob("*"):
            if filepath.is_file():
                if not is_valid_image(filepath):
                    invalid += 1
                    continue
                
                file_hash = calculate_hash(filepath)
                if file_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(file_hash)

                # Extract generator from path if nested inside ai_generated
                generator = "unknown"
                if label == config.LABEL_AI_GENERATED and filepath.parent != directory:
                    generator = filepath.parent.name
                
                width, height, img_format = get_image_info(filepath)
                if width is None:
                    invalid += 1
                    continue

                records.append({
                    "image_path": str(filepath.absolute()),
                    "label": label,
                    "class_name": class_name,
                    "source": "unknown",
                    "generator": generator if label == config.LABEL_AI_GENERATED else "N/A",
                    "width": width,
                    "height": height,
                    "format": img_format,
                })

    print("Scanning for real images...")
    process_directory(real_dir, config.LABEL_REAL, "real")
    
    print("Scanning for AI-generated images...")
    process_directory(ai_dir, config.LABEL_AI_GENERATED, "ai_generated")

    print(f"Discovered {len(records)} valid unique images.")
    print(f"Ignored {invalid} invalid files and {duplicates} duplicates.")

    if not records:
        print("No records found, aborting manifest generation.")
        return

    # Splitting
    random.seed(config.RANDOM_SEED)
    random.shuffle(records)

    n_total = len(records)
    n_train = int(n_total * config.TRAIN_SPLIT)
    n_val = int(n_total * config.VAL_SPLIT)

    for i, record in enumerate(records):
        if i < n_train:
            record["split"] = "train"
        elif i < n_train + n_val:
            record["split"] = "val"
        else:
            record["split"] = "test"

    # Write manifest
    fieldnames = ["image_path", "label", "class_name", "source", "generator", "split", "width", "height", "format"]
    
    with open(config.MANIFEST_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    
    print(f"Manifest written to {config.MANIFEST_PATH}")

if __name__ == "__main__":
    discover_and_split()
