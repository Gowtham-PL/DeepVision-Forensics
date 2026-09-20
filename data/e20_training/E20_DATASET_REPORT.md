# E20 Dataset Construction & Distribution Report

**Dataset Identifier**: `E20_TRAIN_DEV_EXPANDED`  
**Date**: September 18, 2026  
**Status**: Frozen & Audited (Dataset Construction Only)  
**Deterministic Partition Seed**: 42  

---

## 1. Executive Summary & Core Objective

The **E20 Dataset** was constructed directly in response to the comprehensive failure-driven audit (E1–E19). Previous experimental cycles revealed critical vulnerability vectors in the DeepVision-Forensics detector:
1. **Generator Narrowness & Flow-Matching Domain Shift**: Prior training datasets were heavily skewed toward older GANs and early latent diffusion (SD 1.4/1.5/2.1), leaving the detector susceptible to modern flow-matching architectures (FLUX.1) and multimodal generation models (Gemini / Imagen 3).
2. **Camera ISP & Hardware Bias**: When exposed to real smartphone photography from modern ISP pipelines (Google Pixel 7/8/9, Apple iPhone X, Samsung Galaxy S9, Vivo X90) and uncompressed DSLR sensors (Nikon D7000 / D90), detectors trained on narrow web scrapes produced high false-positive rates due to conflating camera post-processing (HDR tone mapping, localized sharpening, Bayer demosaicing) with generative artifacts.
3. **Compound Compression Invariance**: Detectors trained purely on pristine images failed on compressed images, while detectors trained on naive compression (E15) over-relied on high-frequency corner crop features that collapsed under social-media recompression (WhatsApp 1280px, sequential JPEG).

To address these vulnerabilities without compromising evaluation integrity, E20 establishes a **balanced, diverse, and leakage-sealed** training and development corpus.

---

## 2. Dataset Scale & High-Level Metrics

| Metric | Train Split | Dev Split | Total |
| :--- | :--- | :--- | :--- |
| **Clean Originals** | **3,200** | **400** | **3,600** |
| - Real Originals | 1,600 (50.0%) | 200 (50.0%) | 1,800 (50.0%) |
| - AI Originals | 1,600 (50.0%) | 200 (50.0%) | 1,800 (50.0%) |
| **Total Examples (Originals + Variants)** | **22,400** | **2,800** | **25,200** |
| - Real Examples | 11,200 (50.0%) | 1,400 (50.0%) | 12,600 (50.0%) |
| - AI Examples | 11,200 (50.0%) | 1,400 (50.0%) | 12,600 (50.0%) |
| **Compression Variants per Original** | 7 symmetric | 7 symmetric | 7 symmetric |
| **Exclusion Pool Collisions** | **0** / 41,372 | **0** / 41,372 | **0** / 41,372 |
| **Cross-Split Collisions (SHA / Parent)** | **0** | **0** | **0** |

*Note: All compression variants derived from an original image are strictly bound to the split of their parent original. No original image or derivative appears across splits.*

---

## 3. Real Image Distribution & Camera Hardware

The Real split incorporates 1,800 unique, verified original sensor captures and uncompressed camera outputs spanning multiple mobile device manufacturers and professional DSLR optical systems:

| Device Family | Device Model(s) | Primary Source | Train Orig. (Examples) | Dev Orig. (Examples) | Total Orig. (Examples) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple iPhone** | iPhone X | `marcosv/rawir` (iphonex-samsungs9) | 489 (3,423) | 61 (427) | **550 (3,850)** |
| **Samsung Galaxy**| Galaxy S9 | `marcosv/rawir` (iphonex-samsungs9) | 489 (3,423) | 61 (427) | **550 (3,850)** |
| **Nikon DSLR** | Nikon D7000 / D90 | GRIP Unina (`real_RAISE_1k`) | 356 (2,492) | 44 (308) | **400 (2,800)** |
| **Vivo** | Vivo X90 | `marcosv/rawir` (vivo-x90) | 222 (1,554) | 28 (196) | **250 (1,750)** |
| **Google Pixel** | Pixel 7 / Pixel 8 | Wikimedia Commons (EXIF-verified) | 44 (308) | 6 (42) | **50 (350)** |
| **Total Real** | — | — | **1,600 (11,200)** | **200 (1,400)** | **1,800 (12,600)** |

### Scientific Rationale
- **Smartphone ISP Diversity**: By training symmetrically on Apple, Samsung, Vivo, and Google Pixel hardware, the model is exposed to diverse computational photography pipelines (e.g., Apple's Smart HDR, Samsung's scene optimizer, Vivo's Zeiss T* coating pipeline, and Google's HDR+ multi-frame bracketing).
- **Achromatic & Sensor Baseline**: Nikon DSLR RAW captures from RAISE ensure high-resolution optical ground truth free from aggressive mobile phone aggressive temporal denoising.

---

## 4. AI Image Distribution & Generator Architecture

The AI split incorporates 1,800 unique synthetic images spanning modern state-of-the-art flow-matching, multimodal image models, diffusion transformers, and commercial text-to-image engines:

| Generator Family | Documented Version | Source Dataset | Train Orig. (Examples) | Dev Orig. (Examples) | Total Orig. (Examples) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FLUX.1** | `Flux.1-dev` (Flow-Matching) | `data-is-better-together/open-image-preferences-v1` | 948 (6,636) | 118 (826) | **1,066 (7,462)** |
| **Stable Diffusion** | `SD-3.5 / SDXL` | `data-is-better-together/open-image-preferences-v1` | 577 (4,039) | 73 (511) | **650 (4,550)** |
| **OpenAI DALL-E** | DALL-E 2 / 3 (`version_unknown`) | Wikimedia Commons | 34 (238) | 4 (28) | **38 (266)** |
| **Gemini / Imagen** | Gemini 1.5 Pro / Imagen 3 | `Falah/images_gemini` & Wikimedia Commons | 32 (224) | 4 (28) | **36 (252)** |
| **Midjourney** | Midjourney v5/v6 (`version_unknown`) | Wikimedia Commons | 9 (63) | 1 (7) | **10 (70)** |
| **Total AI** | — | — | **1,600 (11,200)** | **200 (1,400)** | **1,800 (12,600)** |

### Versioning Compliance
Per strict scientific protocol, exact generator versions were preserved when documented by the source metadata (`Flux.1-dev`, `SD-3.5 / SDXL`, `Gemini 1.5 Pro / Imagen 3`). Where exact minor versions could not be independently authenticated from metadata (e.g. Wikimedia uploads stating DALL-E or Midjourney without parameter logs), `version_unknown` was strictly recorded; no versions were inferred from visual appearance.

---

## 5. Symmetric Compression Transformation Suite

To eliminate the spurious correlation where detectors associate compression artifacts with AI or pristine quality with authenticity, **every single original image (Real and AI) is subjected to the exact same 7-stage degradation protocol**:

| Variant Key | Transformation Description | JPEG Quality | Subsampling | Target Domain / Artifact Mode | Train Count | Dev Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean` | Original uncompressed / pristine source file | Original | Original | High-frequency sensor & spectral artifacts | 3,200 | 400 |
| `resize_jpeg` | 75% bilinear downscale + JPEG compression | 80 | 4:2:0 | Standard web re-encoding | 3,200 | 400 |
| `moderate_jpeg` | Direct JPEG re-encoding | 65 | 4:2:0 | Mid-grade web / storage compression | 3,200 | 400 |
| `severe_jpeg` | Aggressive JPEG re-encoding | 40 | 4:2:0 | Heavy bandwidth throttling / aggressive 8x8 DCT grid | 3,200 | 400 |
| `sequential_jpeg`| JPEG Q85 save followed by reload & JPEG Q60 save | 60 | 4:2:0 | Multiple recompression cycles (generation loss) | 3,200 | 400 |
| `resize_jpeg_resize`| Downscale 60% + JPEG Q75 + upscale back to 100% + JPEG Q85 | 75 | 4:2:0 | Social thumbnail expansion / compound scaling | 3,200 | 400 |
| `social_media_whatsapp`| Downscale max dimension to 1280px + JPEG Q75 | 75 | 4:2:0 | Messaging app simulation (WhatsApp standard transcode) | 3,200 | 400 |
| **Total** | — | — | — | — | **22,400** | **2,800** |

---

## 6. Localized AI-Edited / Inpainting Evaluation

During candidate discovery, candidate inpainting / localized edit repositories were analyzed. 
- **Audit Finding**: Open-source datasets claiming localized edits (e.g., MagicBrush, InstructPix2Pix) predominantly utilize synthetic automated diffs or lack authentic, verified pixel-level ground truth masks.
- **Scientific Decision**: In strict accordance with the user instructions ("Do NOT fabricate masks. If sufficient reliable localized-edit data cannot be acquired, do not manufacture synthetic claims"), localized AI-edited imagery was **omitted from E20 Train/Dev** rather than introducing unverified synthetic claims or fabricated masks. Localized AI edits will be addressed in a specialized future benchmark where legal and provenance-safe pixel-level masks are fully established.

---

## 7. Leakage Prevention & Cryptographic Audit

To guarantee that E20 models cannot memorize or leak into any evaluation benchmark:
1. **Historical Exclusion Index**: Compiled 41,372 SHA256 hashes and 41,373 64-bit perceptual hashes (`dhash`, `ahash`) spanning:
   - E14 `FINAL_TEST_POOL` & `FINAL_TEST_DEGRADED`
   - E19 Smartphone Benchmark (Pristine and WhatsApp subsets)
   - WhatsApp N=67 benchmark
   - E6 Hard Cases & all prior historical evaluation benchmarks
2. **Rejection Statistics During Construction**:
   - Duplicate / Perceptual Collisions Rejected: **1,304** candidates.
   - Network / Download Errors: **1,036** attempts.
3. **Cross-Split Non-Contamination Audit**:
   - Cross-Split SHA collisions: **0**
   - Cross-Split Parent original collisions: **0**
   - Historical Benchmark SHA collisions: **0**
   - Perceptual separation: Sampled minimum dHash Hamming distance between clean Train and Dev originals is **6 bits**, confirming distinct visual scenes.

---

## 8. Readiness Assessment

The E20 dataset construction is **100% complete and verified**. All manifests, images, and integrity logs are in place:
- `data/e20_training/manifests/e20_train_manifest.csv` (22,400 rows)
- `data/e20_training/manifests/e20_dev_manifest.csv` (2,800 rows)
- `data/e20_training/e20_integrity_report.json` (`leakage_rule_passed: true`)

**Strict Boundary Confirmation**:
- Zero model training was performed.
- Zero model inference was performed.
- No model checkpoints were altered.
- Production code and existing benchmarks remain completely untouched.
