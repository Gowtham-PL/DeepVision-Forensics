# E14 — Independent External Robustness Dataset

This directory contains the protocol, manifests, and audit reports for **Experiment 14 (E14)**: the construction and cryptographic sealing of an independent external image pool for DeepVision-Forensics.

---

## Directory Structure

```
experiments/final_external_test/
├── DATASET_PROTOCOL.md                         # Full protocol, isolation rules, and guidelines
├── README.md                                   # This navigation document
├── manifests/
│   ├── external_pool_manifest.csv              # Full clean candidate pool manifest (N=438)
│   ├── final_train_manifest.csv                # Frozen training pool manifest (N=200: 100 Real, 100 AI)
│   ├── final_test_manifest.csv                 # Cryptographically sealed test manifest (N=200: 100 Real, 100 AI)
│   ├── final_train_compression_manifest.csv    # Training compression variants (N=800)
│   ├── final_test_degraded_manifest.csv        # Test robustness variants (N=600)
│   └── FINAL_TEST_SET_SEAL.json                # Cryptographic seal and integrity record
├── reports/
│   ├── DATASET_AUDIT.md                        # Comprehensive 15-point dataset audit report
│   └── DUPLICATE_AUDIT.md                      # Detailed candidate-by-candidate duplicate rejection log
└── scripts/
    ├── build_exclusion_index.py                # Pre-indexes 39,574 historical project hashes
    ├── acquire_and_audit_pool.py               # Stream acquisition, decoding, and duplicate filtering
    ├── supplement_pool.py                      # Supplementary sampling for exact 200/200 balance
    ├── split_and_freeze_pool.py                # Random stratified split (seed=42) & sealing
    └── generate_derivatives.py                 # Strictly isolated compression derivative generation
```

Associated Image Storage:
```
data/final_external_pool/
├── real/                     # Clean Real images from RAISE-1k (N=221)
├── ai/                       # Clean AI images from Synthbuster (N=217 across 8 generators)
├── compression_train/        # Training compression variants (N=800, derived from Train Pool ONLY)
├── final_test_degraded/      # Robustness test variants (N=600, derived from Test Pool ONLY)
└── reports/
    └── DUPLICATE_AUDIT.md    # Rejection and deduplication log
```

---

## Key Benchmark Specifications

- **Clean External Pool Size**: 438 unique images (221 Real, 217 AI)
- **`FINAL_TRAIN_POOL`**: 200 images (100 Real, 100 AI)
- **`FINAL_TEST_POOL`**: 200 images (100 Real, 100 AI)
- **Class Balance**: Exactly 1.0 : 1.0 (50% Real / 50% AI)
- **AI Stratification**: 8 distinct diffusion generators (DALL-E 2, DALL-E 3, Midjourney v5, Adobe Firefly, SD 1.3, SD 1.4, SD 2, Glide)
- **Photographic Sensors**: Optical camera sensor originals from RAISE-1k
- **Test Manifest SHA256**: `4e7d8ea0afe6ec8f401fc48aa131751eb2c27b2378e9b2f87c18b616ad794d06`
- **Aggregate Test Images SHA256**: `7031f8992270ceaa1b110277f30f52122f314c35519cf3700755868fc2d6855d`

---

## Research Integrity Warning

> [!CAUTION]
> **`FINAL_TEST_POOL` is a sealed independent evaluation benchmark.**
> 
> It must **never** be used for model training, fine-tuning, augmentation tuning, architecture search, hyperparameter optimization, or decision threshold calibration. It is reserved strictly for final single-shot evaluation of candidate forensic models.
