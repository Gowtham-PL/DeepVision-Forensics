# E19 — Real Smartphone / ISP Diversity Benchmark Dataset Report

**Freeze Date**: 2026-09-18 13:53:50 UTC  
**Provenance Audit Date**: 2026-09-18 14:20:00 UTC  
**Status**: PERMANENTLY FROZEN  
**Integrity Status**: 100% VERIFIED — ZERO HISTORICAL LEAKAGE, ZERO INFERENCE PERFORMED

---

## 1. Dataset Objective

In Experiments 15, 17, and 18, forensic evaluations revealed that synthetic compression augmentations alone are insufficient to distinguish between generative AI diffusion artifacts and the real computational imaging signal processing (ISP) artifacts of modern smartphones (e.g., Apple Photonic Engine, Samsung tone mapping and edge sharpening, Google HDR+).

The **E19 Smartphone Benchmark** was constructed as a completely independent, pristine test benchmark specifically designed to evaluate:
1. Real smartphone ISP diversity across **Apple, Samsung, and Google Pixel**.
2. Robustness to social-media compression via a **controlled WhatsApp-style compression simulation** ($Q=75$, 4:2:0 chroma subsampling, 1280 px max dimension, stripped EXIF/ICC).
3. Cross-generator sensitivity against modern state-of-the-art AI architectures (**FLUX, Gemini, Midjourney, DALL-E 3, Stable Diffusion**).

---

## 2. Dataset Composition & Verified Provenance

| Partition | Category / Subgroup | Exact Count | Provenance Source & Dataset | Provenance Category | License / Attribution |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Real Smartphone** | Apple iPhone (iPhone X) | 35 | `marcosv/rawir` (`train/lq-iphone/`) | **SUPPORTED FACT** (Explicit `iPhoneX` tag in dataset card) | MIT License |
| **Real Smartphone** | Samsung Galaxy (Galaxy S9) | 35 | `marcosv/rawir` (`train/lq-samsung/`) | **SUPPORTED FACT** (Explicit `SamsungS9` tag in dataset card) | MIT License |
| **Real Smartphone** | Google Pixel (Pixel 7–9 family) | 30 | `marcosv/rawir` (`train/googlepx/`) | **SOURCE-DERIVED METADATA** (Series 7–9 supported; exact sub-model per file is **UNKNOWN / UNSPECIFIED**) | MIT License |
| **Subtotal Real** | **3 Camera ISP Families** | **100** | **NTIRE 2025 RAW Image Restoration Challenge** | **100% Authentic Smartphone Cameras** | Open Research License |
| | | | | | |
| **AI Generated** | FLUX (Flux LoRA Synthetic Corpus) | 20 | `HuggingFace: mramazan/Synthetic-Character-Dataset-for-FLUX-LORA` | **SOURCE-DERIVED METADATA** (Tagged for Flux LoRA training; exact underlying generation engine is **UNSPECIFIED**) | Other (Research only) |
| **AI Generated** | Google Gemini | 20 | `Wikimedia Commons (Category:Images_generated_by_Gemini)` | **SOURCE-DERIVED METADATA** (Gemini-generated per category/metadata; underlying Imagen engine is **UNSPECIFIED**) | Verified CC-Zero / CC-BY / PD |
| **AI Generated** | Midjourney | 20 | `Wikimedia Commons (Category:Images_generated_by_Midjourney)` | **SOURCE-DERIVED METADATA** (Midjourney-generated per category/metadata; exact version v5/v6 is **UNSPECIFIED**) | Verified CC-Zero / CC-BY / PD |
| **AI Generated** | OpenAI DALL-E (DALL-E 3) | 20 | `Wikimedia Commons (Category:Images_generated_by_DALL-E_3)` | **SUPPORTED FACT** (Explicitly tagged DALL-E 3 in category and file metadata) | Verified CC-Zero / CC-BY / PD |
| **AI Generated** | Stability AI Stable Diffusion | 20 | `Wikimedia Commons (Category:Images_generated_by_Stable_Diffusion)` | **SOURCE-DERIVED METADATA** (Stable Diffusion-generated per category; exact checkpoint SDXL/SD2 is **UNSPECIFIED**) | Verified CC-Zero / CC-BY / PD |
| **Subtotal AI** | **5 Generator Families** | **100** | **Verified Public & Research Corpora** | **100% Synthetic AI Imagery** | Open / Permissive Attribution |
| | | | | | |
| **TOTAL BENCHMARK** | — | **200** | — | — | **100 Real, 100 AI** |

---

## 3. WhatsApp Compression / Domain-Shift Transformation

### Transformation Audit: Controlled Simulation (Not Live Network Transit)

> [!IMPORTANT]
> The WhatsApp subset images did **not** travel through the proprietary WhatsApp mobile application or cellular servers.
> Instead, they were processed via a **controlled WhatsApp-style compression simulation** reproducing WhatsApp's known production image processing pipeline:
> - Long edge rescaled to 1280 px (preserving aspect ratio)
> - JPEG compression with Quality Factor $Q=75$
> - Standard 4:2:0 chroma subsampling
> - Complete removal of EXIF metadata and ICC color profiles

### Exact WhatsApp Subset Stratification:
- **Total WhatsApp-Style Processed**: Exactly **60 images (30.0%)**
  - **Real**: 30 images (11 iPhone X, 10 Samsung S9, 9 Google Pixel)
  - **AI**: 30 images (6 FLUX, 6 Gemini, 6 Midjourney, 6 DALL-E 3, 6 Stable Diffusion)
- **Total Pristine / Lossless ISP Crops**: Exactly **140 images (70.0%)**
  - **Real**: 70 images (24 iPhone X, 25 Samsung S9, 21 Google Pixel)
  - **AI**: 70 images (14 FLUX, 14 Gemini, 14 Midjourney, 14 DALL-E 3, 14 Stable Diffusion)

---

## 4. Cryptographic Leakage & Exclusion Audit

Every candidate was audited against the comprehensive historical exclusion index covering **41,172 historical project images** (including GenImage, E5 external, Real World Test v1/v2, WhatsApp robustness test, E6 hard cases, E14 train/test pools, and degraded variants).

| Audit Criterion | Protocol Rule | Result | Verification Status |
| :--- | :--- | :---: | :---: |
| **Exact SHA256 Match** | Zero match against historical index | 0 matches | **PASSED** (0.00%) |
| **Perceptual dhash** | Hamming distance > 4 against historical index | 0 matches | **PASSED** (0.00%) |
| **Perceptual ahash** | Hamming distance > 4 against historical index | 0 matches | **PASSED** (0.00%) |
| **Internal SHA256** | Zero duplicate hashes within E19 | 0 duplicates | **PASSED** (200 unique) |
| **Internal Perceptual** | Zero near-duplicates within E19 | 0 duplicates | **PASSED** (0 near-duplicates) |
| **Disk Integrity** | All 200 files exist and match SHA256 | 200 / 200 | **PASSED** (0 mismatches) |
| **Model Inference** | Zero inference during construction/audit | 0 inferences | **PASSED** (Strictly enforced) |

---

## 5. Storage Layout & Artifacts

- **Real Images**: `data/e19_smartphone_benchmark/real/` (100 files, `e19_real_phone_001_wa.jpeg` ... `e19_real_phone_100_raw.png`)
- **AI Images**: `data/e19_smartphone_benchmark/ai/` (100 files, `e19_ai_001_wa.jpeg` ... `e19_ai_100_raw.png`)
- **Test Manifest**: `data/e19_smartphone_benchmark/manifests/e19_test_manifest.csv` (200 rows with verified metadata)
- **Integrity Report**: `data/e19_smartphone_benchmark/e19_integrity_report.json`
- **Provenance Audit**: `data/e19_smartphone_benchmark/E19_PROVENANCE_AUDIT.md`

---

## 6. Scientific Limitations & Boundaries

1. **Camera Model Specificity**:
   - iPhone X and Samsung Galaxy S9 are verified at the exact model level.
   - Google Pixel images are verified as belonging to the Google Pixel 7–9 series, but the exact model (Pixel 7 vs 8 vs 9) for individual crop files is unknown/unspecified in the upstream dataset.
2. **AI Generator Engine Specificity**:
   - DALL-E 3 is verified at the exact model level.
   - FLUX, Gemini, Midjourney, and Stable Diffusion are verified at the model family level; exact internal checkpoints or underlying engines (e.g. Imagen version or SDXL vs SD2) are not individually recorded in community repository metadata.
3. **Compression Representation**:
   - The 60-image social-media subset represents a controlled WhatsApp-style simulation and must be evaluated as such.
4. **Sealed Benchmark**:
   - E19 must remain permanently frozen as a zero-leakage evaluation-only benchmark and never incorporated into training or threshold optimization.
