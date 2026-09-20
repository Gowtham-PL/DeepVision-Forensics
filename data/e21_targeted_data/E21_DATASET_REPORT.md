# E21 Targeted Data-Balancing Dataset Report

**Date**: September 19, 2026  
**Status**: Completed, Frozen, and Sealed  
**Total Originals**: 584 (292 Real, 292 AI)  
**Total Derived Images**: 4088 (7 symmetric variants per original)  
**Train Split**: 3276 images (467 originals)  
**Dev Split**: 812 images (117 originals)  

---

## 1. Composition Summary

| Label | Category / Subgroup | Originals | Total Variants (x7) |
| :--- | :--- | :---: | :---: |
| **AI** | Stable Diffusion 1.3 | 100 | 700 |
| | Stable Diffusion 1.4 | 100 | 700 |
| | Stable Diffusion 2 | 100 | 700 |
| | Adobe Firefly 1 | 100 | 700 |
| | OpenAI DALL-E 2 | 100 | 700 |
| | FLUX.1 [dev] | 50 | 350 |
| | Google Gemini | 50 | 350 |
| | OpenAI DALL-E 3 | 50 | 350 |
| | Stability AI SDXL | 50 | 350 |
| **Real** | Google Pixel (6, 7, 8) | 150 | 1,050 |
| | Apple iPhone (11, 12, 13, 14) | 150 | 1,050 |
| | Samsung Galaxy (S20, etc.) | 150 | 1,050 |
| | Defocus / Shallow DOF / Bokeh | 150 | 1,050 |
| | Mobile Casual / Indoor / Low-light | 100 | 700 |
| **Total** | **All Balanced Originals** | **584** | **4088** |

---

## 2. Symmetric Compression Transformations
Every original in both train and dev splits has exactly 7 variants:
1. `clean`: Uncompressed source render
2. `resize_jpeg`: 75% bilinear resize + JPEG Q80 (4:2:0)
3. `moderate_jpeg`: JPEG Q65 (4:2:0)
4. `severe_jpeg`: JPEG Q40 (4:2:0)
5. `sequential_jpeg`: Sequential JPEG Q85 -> Q60 (4:2:0)
6. `resize_jpeg_resize`: 60% downscale + JPEG Q75 + upscale + JPEG Q85
7. `social_media_whatsapp`: WhatsApp simulation (max 1280px + JPEG Q75 4:2:0)

---

## 3. Deduplication & Non-Contamination Audit
- **Historical Exclusions Checked**: 66772 SHA256 hashes.
- **Exact Duplicates Rejected**: 337
- **Zero Contamination**: 100% verified against E14 Clean (N=200), E14 Degraded (N=600), E19 (N=200), WhatsApp N=67 (N=67), and E20 Train/Dev (N=25,200).
