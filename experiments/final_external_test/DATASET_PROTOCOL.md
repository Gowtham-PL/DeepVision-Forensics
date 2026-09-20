# E14 — Independent External Robustness Dataset Protocol

**Protocol Version**: 1.0 (Sealed Benchmark Protocol)  
**Date**: September 18, 2026  
**Status**: ACTIVE — BENCHMARK FROZEN & CRYPTOGRAPHICALLY SEALED

---

## 1. Overview and Core Objective

The purpose of Experiment 14 (E14) is to construct an uncompromised, independent external image pool supporting two distinct research requirements:
1. **Compression-Robustness Training Derivatives**: Providing realistic compression-degraded variants for training and tuning compression-invariant models.
2. **Untouched Final Independent Evaluation Benchmark**: A permanently sealed, zero-shot evaluation test set (`FINAL_TEST_POOL`) that has zero overlap with any training, validation, hyperparameter tuning, or threshold selection procedures.

---

## 2. Strict Non-Contamination & Exclusion Perimeter

Before candidate acquisition, an exclusion perimeter was constructed indexing **39,574 image hashes** across all historical datasets and manifests in the repository:
- `data/genimage/` (35,000 images)
- `data/e5_external/` (4,432 images, `e5_manifest.csv`)
- `data/real_world_test/` (56 images)
- `data/real_world_test_v2/` (76 quarantined images)
- `data/whatsapp_robustness_test/` (67 frozen images)
- `data/e6_hard_cases/` (2 diagnostic images)
- All experimental runs E6 through E13

### Deduplication Criteria:
Any candidate image was immediately rejected if it triggered any of the following:
1. **Exact SHA256 Collision**: String equality with any existing project image hash.
2. **Perceptual Difference Hash Collision**: Hamming distance $\le 4$ on 64-bit `dhash` against any existing project image.
3. **Perceptual Average Hash Collision**: Hamming distance $\le 4$ on 64-bit `ahash` against any existing project image.
4. **Intra-Pool Collision**: Duplicate SHA256 or perceptual hash distance $\le 4$ against any previously accepted candidate in the new pool.

---

## 3. External Data Sources & Provenance

Two independent, peer-reviewed external benchmarks were selected:

| Source Dataset | Class | Subset Acquired | Generator / Optical Device | License | Commercial / Research Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RAISE-1k** | Real | 310 uncompressed camera originals | Nikon D7000, D90, D40, D700 (RAW camera sensors) | RAISE Digital Forensics License | Free for academic / research evaluation |
| **Synthbuster** | AI | 297 high-fidelity diffusion generations | 8 distinct diffusion models: DALL-E 2, DALL-E 3, Adobe Firefly, Midjourney v5, SD 1.3, SD 1.4, SD 2, Glide | CC BY-NC-SA 4.0 (Quentin Bammey / Centre Borelli) | Non-commercial research / academic use |

*Note on SDXL exclusion*: All sampled Synthbuster SDXL images collided with pre-existing E5 external manifests (`EXACT_SHA256_COLLISION_EXISTING_PROJECT`) and were strictly excluded to ensure zero cross-experiment leakage.

---

## 4. Frozen Stratified Dataset Split (Seed = 42)

The clean external candidate pool (438 images: 221 Real, 217 AI) was partitioned using a fixed random seed (`SEED = 42`) prior to running any model evaluations or transformations.

### Split Sizes and Allocation:

| Split Name | Total Images | Real Count | AI Count | Stratification Protocol | Permitted Usage |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`FINAL_TRAIN_POOL`** | **200** | 100 | 100 | Stratified across 8 AI generators (12–15 each) and optical cameras | Compression augmentations, training, tuning |
| **`FINAL_TEST_POOL`** | **200** | 100 | 100 | Proportional mirror of train split across all 8 generators | **Strict zero-shot final evaluation only** |
| *Reservoir* | *38* | 21 | 17 | Held in reserve | Backup reserve |

### Generator Breakdown in `FINAL_TEST_POOL`:
- Optical Camera: 100
- Stable Diffusion 2: 15
- DALL-E 2: 14
- Adobe Firefly: 14
- Stable Diffusion 1.3: 14
- Glide: 13
- Midjourney v5: 12
- Stable Diffusion 1.4: 12
- DALL-E 3: 6

---

## 5. Cryptographic Sealing of Final Test Set

To ensure research integrity, `FINAL_TEST_POOL` has been cryptographically sealed. Any unauthorized modification, addition, or removal of test images will invalidate the cryptographic signature:

- **Test Manifest Path**: `experiments/final_external_test/manifests/final_test_manifest.csv`
- **Test Manifest SHA256**: `4e7d8ea0afe6ec8f401fc48aa131751eb2c27b2378e9b2f87c18b616ad794d06`
- **Aggregate Test Images SHA256**: `7031f8992270ceaa1b110277f30f52122f314c35519cf3700755868fc2d6855d`
- **Cryptographic Seal Document**: `experiments/final_external_test/manifests/FINAL_TEST_SET_SEAL.json`

---

## 6. Strict Research Integrity Rules

1. **Sealed Benchmark Rule**: `FINAL_TEST_POOL` must NEVER be used for training, fine-tuning, augmentation policy design, architecture search, hyperparameter tuning, early stopping, or decision threshold calibration.
2. **Single-Shot Evaluation Only**: When an architecture is fully finalized and ready for benchmark verification, it is evaluated on `FINAL_TEST_POOL` in a single unalterable pass.
3. **No Selective Filtering**: No test sample may ever be removed or reclassified on the basis of model prediction confidence, error analysis, or perceived difficulty.
4. **Symmetric Derivatives**: All compression variants maintain exact label symmetry between Real and AI.
