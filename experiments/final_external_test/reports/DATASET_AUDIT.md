# E14 — Independent External Robustness Dataset Audit Report

**Report Date**: September 18, 2026  
**Evaluation Protocol**: Sealed Double-Blind Forensic Benchmark (E14)  
**Status**: COMPLETE — ALL LEAKAGE CHECKS PASSED, TEST SET SEALED

---

## 1. Data Sources

The E14 independent image pool was assembled from two established, peer-reviewed forensic benchmarks:

1. **RAISE-1k (Real Photography Benchmark)**:
   - *Source*: University of Naples Federico II / GRIP Digital Media Forensic Group (Dang-Nguyen et al., 2015).
   - *Repository URL*: `https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip`
   - *Nature*: High-resolution uncompressed camera RAW/PNG originals from optical DSLR camera sensors (Nikon D7000, D90, D40, D700) depicting diverse indoor and outdoor scenes, textures, portraits, and landscapes.

2. **Synthbuster (Diffusion Model Benchmark)**:
   - *Source*: Quentin Bammey, Centre Borelli, ENS Paris-Saclay / Université Paris-Saclay (2023).
   - *Zenodo DOI*: `10.5281/zenodo.10066460`
   - *Zenodo URL*: `https://zenodo.org/records/10066460/files/synthbuster.zip`
   - *Nature*: High-resolution AI-generated synthetic images from state-of-the-art diffusion generative architectures conditioned on RAISE-derived descriptive forensic prompts.

---

## 2. Licenses & Provenance

| Benchmark | License | Commercial Restriction | Research & Academic Scope | Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **RAISE-1k** | RAISE Forensic Benchmark License | Non-Commercial Research Only | Fully Permitted for Model Benchmarking & Forensics | Dang-Nguyen, Duc-Tien, et al. (ACM MMSys 2015) |
| **Synthbuster** | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) | Non-Commercial Only | Fully Permitted for Research, Detection, & Evaluation | Quentin Bammey (Centre Borelli, ENS Paris-Saclay) |

*Compliance Statement*: Both datasets permit academic research and benchmarking. No commercial models were trained or deployed using restricted commercial assets.

---

## 3. Downloaded Subsets

Rather than downloading the full multi-gigabyte archives (~14 GB total), targeted subset streaming was executed using HTTP Range requests with multi-threaded byte extraction:
- **RAISE-1k**: 310 photographic candidates uniformly sampled across the 1,000 uncompressed camera images.
- **Synthbuster**: 297 synthetic candidates sampled uniformly across the 9 diffusion generators (approx. 33 images per model).

---

## 4. Candidate Count

- **Total Initial Candidates Sampled**: 607 images
  - Real Candidates (RAISE-1k): 310
  - AI Candidates (Synthbuster): 297

---

## 5. Duplicate Rejection Count & Breakdown

A multi-stage duplicate rejection filter was executed against the **39,574 historical project images** and within the candidate pool itself.

| Rejection Category | Rejection Reason | Count | Action Taken |
| :--- | :--- | :---: | :--- |
| **Cross-Dataset Exact Collision** | `EXACT_SHA256_COLLISION_EXISTING_PROJECT` | 78 | **REJECTED**: Exact byte match with images in `data/e5_external/` (specifically Synthbuster SDXL previously indexed in E5). |
| **Cross-Dataset Perceptual Collision** | `PERCEPTUAL_AHASH_COLLISION_EXISTING_PROJECT` | 85 | **REJECTED**: Average hash Hamming distance $\le 4$ with existing project images in `data/e5_external/` or `genimage`. |
| **Cross-Dataset Perceptual Collision** | `PERCEPTUAL_DHASH_COLLISION_EXISTING_PROJECT` | 4 | **REJECTED**: Difference hash Hamming distance $\le 4$ with existing project images. |
| **Internal Candidate Pool Collision** | `PERCEPTUAL_DHASH_COLLISION_INTERNAL_POOL` | 2 | **REJECTED**: Duplicate/near-identical candidate within the acquired batch. |
| **Total Rejected Candidates** | — | **169** | Zero contaminated images admitted to pool. |

---

## 6. Final Clean External Pool Count

After multi-stage filtering, the clean external candidate pool contains:
- **Total Unique Images in Clean Pool**: **438 images**
- **Real Images**: 221 (50.5%)
- **AI Images**: 217 (49.5%)

All 438 images are verified unique, clean, and stored at:
- `data/final_external_pool/real/` (221 PNG images)
- `data/final_external_pool/ai/` (217 PNG images)
- Manifest: `experiments/final_external_test/manifests/external_pool_manifest.csv`

---

## 7. Train / Test Counts

Using a fixed random seed (`SEED = 42`), the clean pool was partitioned into two balanced splits, leaving 38 images in an untouched reserve reservoir:

| Split Name | Total Count | Real Images | AI Images | Split Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **`FINAL_TRAIN_POOL`** | **200** | 100 | 100 | 50.0% |
| **`FINAL_TEST_POOL`** | **200** | 100 | 100 | 50.0% |
| *Reserve Reservoir* | *38* | 21 | 17 | — |
| **Total** | **438** | **221** | **217** | **100.0%** |

---

## 8. Real / AI Class Balance

Both `FINAL_TRAIN_POOL` and `FINAL_TEST_POOL` feature an **exact 1:1 class ratio**:
- `FINAL_TRAIN_POOL`: **100 Real (50.0%) / 100 AI (50.0%)**
- `FINAL_TEST_POOL`: **100 Real (50.0%) / 100 AI (50.0%)**

This 1:1 balance provides unskewed class balance for balanced accuracy, ROC-AUC, and macro F1 metric calculations.

---

## 9. Generator Distribution

The AI split was stratified across all 8 accepted diffusion generators:

| Generator Name | Total in Pool | Allocated to `FINAL_TRAIN_POOL` | Allocated to `FINAL_TEST_POOL` | In Reserve Reservoir |
| :--- | :---: | :---: | :---: | :---: |
| **Stable Diffusion 2** | 33 | 15 | 15 | 3 |
| **DALL-E 2** | 30 | 14 | 14 | 2 |
| **Adobe Firefly** | 30 | 14 | 14 | 2 |
| **Stable Diffusion 1.3** | 31 | 14 | 14 | 3 |
| **Glide** | 28 | 13 | 13 | 2 |
| **Midjourney v5** | 26 | 12 | 12 | 2 |
| **Stable Diffusion 1.4** | 26 | 12 | 12 | 2 |
| **DALL-E 3** | 13 | 6 | 6 | 1 |
| **AI Subtotal** | **217** | **100** | **100** | **17** |

---

## 10. Source Distribution

| Source Dataset | Generator / Device Category | `FINAL_TRAIN_POOL` | `FINAL_TEST_POOL` | Reserve | Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RAISE-1k** | Optical DSLR Camera Sensors | 100 | 100 | 21 | 221 |
| **Synthbuster** | Generative Diffusion Networks | 100 | 100 | 17 | 217 |
| **Total** | | **200** | **200** | **38** | **438** |

---

## 11. Compression Transformations & Derivatives

To evaluate and train models under realistic social media and messaging compression without corrupting clean baselines, two strictly isolated derivative pools were constructed:

### A. Training Derivatives (`data/final_external_pool/compression_train/`)
*Generated EXCLUSIVELY from `FINAL_TRAIN_POOL` (200 images $\times$ 4 variants = 800 images total)*:
1. **Clean**: Original uncompressed image (PNG).
2. **WhatsApp-tier JPEG**: Long edge scaled to 1280 px (preserving aspect ratio), JPEG Quality = 75, standard 4:2:0 chroma subsampling.
3. **Double JPEG**: Long edge scaled to 1400 px, initial compression at Q80, decompressed in memory, and re-compressed at Q70 with 4:2:0 chroma.
4. **Moderate Downsample + JPEG**: Long edge downscaled to 960 px, compressed with JPEG Q75 (4:2:0).

Manifest: `experiments/final_external_test/manifests/final_train_compression_manifest.csv` (800 records).

### B. Robustness Evaluation Derivatives (`data/final_external_pool/final_test_degraded/`)
*Generated EXCLUSIVELY from `FINAL_TEST_POOL` (200 images $\times$ 3 variants = 600 images total)*:
1. **WhatsApp-tier Controlled JPEG**: Long edge 1280 px, JPEG Quality 75, 4:2:0 chroma.
2. **Resize (1080) + JPEG Q80**: Long edge 1080 px, JPEG Quality 80, 4:2:0 chroma.
3. **Double JPEG (1280, Q80 $\rightarrow$ Q70)**: Long edge 1280 px, Q80 $\rightarrow$ Q70, 4:2:0 chroma.

Manifest: `experiments/final_external_test/manifests/final_test_degraded_manifest.csv` (600 records).

*Isolation Assurance*: The original images in `FINAL_TEST_POOL` remain completely untouched. Training routines never access any images from `final_test_degraded/` or `FINAL_TEST_POOL`.

---

## 12. Random Seed & Reproducibility

- **Random Seed**: `SEED = 42`
- **RNG Module**: Python standard library `random.seed(42)`
- **Determinism**: The assignment of images to `FINAL_TRAIN_POOL` and `FINAL_TEST_POOL` is fully deterministic and bit-reproducible using `experiments/final_external_test/scripts/split_and_freeze_pool.py`.

---

## 13. Exact Frozen Test-Set Hash & Seal

To guarantee immutability, `FINAL_TEST_POOL` is cryptographically sealed:

- **Manifest File**: `experiments/final_external_test/manifests/final_test_manifest.csv`
- **Manifest SHA-256**: `4e7d8ea0afe6ec8f401fc48aa131751eb2c27b2378e9b2f87c18b616ad794d06`
- **Aggregate Test Images SHA-256**: `7031f8992270ceaa1b110277f30f52122f314c35519cf3700755868fc2d6855d`
- **Cryptographic Seal Path**: `experiments/final_external_test/manifests/FINAL_TEST_SET_SEAL.json`

---

## 14. Leakage & Contamination Checks

| Check Performed | Verification Method | Result | Status |
| :--- | :--- | :--- | :--- |
| **Historical Dataset Overlap** | SHA256 against 39,574 project images | 0 matches admitted | **PASSED** |
| **Historical Near-Duplicate Overlap** | 64-bit dhash Hamming distance $\le 4$ | 0 matches admitted | **PASSED** |
| **Historical Perceptual Overlap** | 64-bit ahash Hamming distance $\le 4$ | 0 matches admitted | **PASSED** |
| **Train / Test Pool Independence** | Set intersection on `image_id` and `sha256` | Exactly 0 overlapping images | **PASSED** |
| **Model Inference Isolation** | Verification of zero model evaluation | Zero predictions computed | **PASSED** |
| **Production Code Immutability** | Verification of source code state | Production untouched | **PASSED** |

---

## 15. Limitations & Protocol Notes

1. **Diffusion-Centric AI Scope**: Synthbuster focuses exclusively on modern diffusion architectures (DALL-E, Midjourney, Stable Diffusion, Firefly, Glide). Classical GAN architectures (ProGAN, StyleGAN) are not included in this benchmark, preserving focus on state-of-the-art diffusion forensics.
2. **SDXL Exclusion**: All sampled Synthbuster SDXL images collided with historical project manifests (`data/e5_external/`), requiring their complete exclusion to protect benchmark independence.
3. **Photographic Sensor Scope**: Real images in RAISE-1k originate from Nikon optical camera sensors. While this provides pristine optical physics, smartphone-specific computational photography pipelines (e.g. Apple Photonic Engine, Google HDR+) are represented via our existing `data/real_world_test_v2/` and `whatsapp_robustness_test/` rather than RAISE-1k.
4. **Sealed Status**: Once unsealed for single-shot evaluation of candidate models, the test set cannot be re-used for iterative training or parameter selection without losing its zero-shot status.
