# Phase 1: Dataset Pipeline Documentation

This document describes the dataset pipeline for the DeepVision Forensics project.

## Expected Dataset Structure
The system expects datasets to be organized logically by class:
```text
data/
├── real/
└── ai_generated/
    └── midjourney/  (Optional generator subdirectory)
```

## Configuration
All configuration variables are defined in `ml/data/config.py`:
- **Paths**: Controls where data and manifest are located.
- **Labels**: `0` for Real, `1` for AI-generated.
- **Image Size**: Configured to `224x224`.
- **Normalization**: Standard ImageNet mean/std used by default.
- **Splits**: `70%` train, `15%` val, `15%` test.
- **Seed**: `42` for reproducible shuffling.

## Manifest Format
The script `prepare.py` scans the directory and creates a `manifest.csv` with fields:
`image_path, label, class_name, source, generator, split, width, height, format`
It tracks generator details if explicitly separated into subfolders inside `ai_generated/`.

## Validation and Duplicate Detection
- Checks extensions (jpg, png, webp).
- Ensures `PIL` can read and decode the image successfully.
- Generates an MD5 hash to prevent duplicate files across splits.

## How to Run
First, create test fixtures if you don't have data:
`.\.venv\Scripts\python.exe tests/fixtures.py`

Generate the manifest (will use test_data if DEEPVISION_DATA_DIR is exported):
`.\.venv\Scripts\python.exe ml/data/prepare.py`

Run the Audit Script:
`.\.venv\Scripts\python.exe ml/data/audit.py`

Run Pytest tests:
`.\.venv\Scripts\python.exe -m pytest tests/test_dataset.py -v`

## Limitations & TBD
- Extensive perceptual hashing is not implemented.
- Final transformations/augmentation schemes for EfficientNet-B3 are TBD.
