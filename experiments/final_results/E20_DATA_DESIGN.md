# Experiment 20 (E20) Data Architecture & Acquisition Specification

**A Rigorous Protocol for Balanced Generative Sampling, Camera ISP Stratification, and Zero-Leakage Dataset Partitioning**

- **Project**: DeepVision-Forensics
- **Status**: Pre-Acquisition Architectural Blueprint
- **Target Dataset Size**: 30,000 Total Images (24,000 Train / 3,000 Dev-Val / 3,000 Final Test)
- **Primary Objective**: Provide balanced multi-generator and multi-ISP training coverage to directly address the empirical failure modes identified in the E20 Failure Audit.

---

## 1. Quantitative Data Stratification Table

The target data pool is partitioned into six distinct technical strata designed to eliminate specific historical failure modes:

| Stratum ID | Stratum Name | Target Share | Target Image Count | Target Real Sources | Target AI Generator Sources | Specific Failure Mode Addressed |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **S1** | **Modern Flow-Matching AI** | 20.0% | 6,000 (3,000 Real / 3,000 AI) | Authentic camera photography | FLUX.1 [dev], FLUX.1 [schnell], Stable Diffusion 3.5 | Eliminates E6-C's 90% miss rate on FLUX (E19: 18/20 FNs) |
| **S2** | **Modern LLM Multimodal AI** | 15.0% | 4,500 (2,250 Real / 2,250 AI) | Authentic camera photography | Google Gemini (Imagen 3), OpenAI ChatGPT-4o | Eliminates E6-C's 70% miss rate on Gemini (E19: 14/20 FNs) |
| **S3** | **Commercial & Classical Diffusion** | 15.0% | 4,500 (2,250 Real / 2,250 AI) | Authentic camera photography | Midjourney v5.2/v6/v6.1, DALL-E 3, SDXL | Preserves high baseline sensitivity on established generators |
| **S4** | **Diverse Smartphone Camera ISPs** | 25.0% | 7,500 (3,750 Real / 3,750 AI) | Google Pixel 6–9, Samsung S20–S24, iPhone 11–16 | Balanced across S1–S3 generator pools | Resolves Google Pixel false-alarm surge (E19: 43.3% FPR) |
| **S5** | **Uncompressed RAW DSLRs** | 10.0% | 3,000 (1,500 Real / 1,500 AI) | Nikon, Canon, Sony uncompressed RAWs (RAISE/Flickr) | High-res lossless synthetic PNGs | Resolves the 100% false-alarm trap on RAISE-1k sensor noise |
| **S6** | **Compound Social Media Channels** | 15.0% | 4,500 (2,250 Real / 2,250 AI) | WhatsApp, Telegram, WebP-transmitted captures | Social-media transmitted AI imagery | Bridges the channel attenuation gap (WhatsApp AI FNR: 72.7%) |

---

## 2. Dataset Partitioning & Quarantine Protocol

To guarantee absolute scientific integrity and prevent data leakage:

```
[Candidate Acquisition Pool: ~30,000 Images]
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   [E20_TRAIN]   [E20_DEV]   [E20_TEST]
     (80.0%)      (10.0%)     (10.0%)
    N=24,000     N=3,000     N=3,000
    12kR / 12kA  1.5kR/1.5kA 1.5kR/1.5kA
```

### Protocol Rules:
1. **Cryptographic Exclusion**: All candidate images must be verified against the Master Historical Exclusion Index (`GenImage`, `E5`, `V2`, `WhatsApp_67`, `E14`, `E19`). Any exact SHA-256 match or perceptual hash collision ($d_H \le 3$, $a_H \le 3$) is permanently rejected.
2. **Device / Subject Splitting**: All crops or bursts from the same camera or scene must reside strictly in the same partition. Never split images of the same subject across Train and Test.
3. **Sealed Test Set**: `E20_TEST` must be cryptographically hashed and permanently sealed before training begins. Zero model forward passes may be executed on `E20_TEST` during optimization, validation, or checkpoint selection.
4. **Permanent Holdouts**: Historical benchmarks (`Dataset V2`, `WhatsApp Benchmark N=67`, `FINAL_TEST_POOL N=200`, and `E19 N=200`) remain permanently quarantined as external blind evaluation assets.

---

## 3. Dynamic Compound Channel Augmentation Specification

For Stratum S6 and 40% of standard training batches, augmentations must strictly mirror the physical characteristics of real messaging networks (derived from the E10-A WhatsApp diagnostic):

```python
# Calibrated Compound Degradation Parameters (Post-E10-A)
DEGRADATION_SPEC = {
    "jpeg_quality_range": (65, 85),      # Matches real WhatsApp/Telegram quantization
    "chroma_subsampling": "4:2:0",        # Mandatory YUV color compression
    "max_dimension": 1280,                # Mobile messaging resize limit
    "resample_filter": "bicubic",         # Standard platform downsampler
    "additive_gaussian_noise": False,     # Strictly FALSE (proven destructive in E4)
    "random_blurring": False              # Strictly FALSE (masks subtle synthetic grids)
}
```

---
*End of E20 Data Architecture & Acquisition Specification.*
