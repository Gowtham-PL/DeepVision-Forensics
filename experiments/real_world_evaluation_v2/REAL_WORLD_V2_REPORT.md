# DeepVision Forensics — Real-World Generalization Evaluation Report

This report provides an empirical evaluation of **E1 Spatial Baseline** versus **E3-Std Dual-Domain Candidate** on completely unconstrained, real-world images (camera/smartphone photographs, web images, and state-of-the-art AI-synthesized media).

---

## 1. Dataset Overview

- **Dataset Path:** `data\real_world_test_v2`
- **Total Images Evaluated:** **76**
- **Authentic Photographs (`real/`):** **38** (Ground Truth = 0)
- **AI-Generated Images (`ai/`):** **38** (Ground Truth = 1, e.g., Gemini, Stable Diffusion, Midjourney, Flux)
- **Decision Threshold:** `0.50` (Fixed, non-tuned)

---

## 2. Summary Leaderboard

| Model ID | Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positive Rate (Real) | False Negative Rate (AI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 Spatial Baseline** | EfficientNet-B3 (Pretrained) | **39.47%** | 16.67% | 5.26% | 0.0800 | **0.2621** | 0.3621 | 26.32% (10/38) | 94.74% (36/38) |
| **E3-Std Dual-Domain Candidate** | Dual-Domain (Spatial + Frequency Standardization) | **48.68%** | 40.00% | 5.26% | 0.0930 | **0.4619** | 0.4540 | 7.89% (3/38) | 94.74% (36/38) |

---

## 3. Confusion Matrix Breakdown

| Model | True Positive (AI detected as AI) | False Positive (Real flagged as AI) | True Negative (Real detected as Real) | False Negative (AI missed as Real) |
| :--- | :---: | :---: | :---: | :---: |
| **E1 Spatial** | 2 (5.3%) | 10 (26.3%) | 28 (73.7%) | 36 (94.7%) |
| **E3-Std Dual** | 2 (5.3%) | 3 (7.9%) | 35 (92.1%) | 36 (94.7%) |

---

## 4. Performance Deltas & Comparative Insights

- **Accuracy Delta (E3-Std - E1):** `+9.21%` (48.68% vs 39.47%)
- **Precision Delta (E3-Std - E1):** `+23.33%` (40.00% vs 16.67%)
- **Recall Delta (E3-Std - E1):** `+0.00%` (5.26% vs 5.26%)
- **F1-Score Delta (E3-Std - E1):** `+0.0130` (0.0930 vs 0.0800)
- **ROC-AUC Delta (E3-Std - E1):** `+0.1998` (0.4619 vs 0.2621)
- **False Positive Rate on Real Images (E3-Std - E1):** `-18.42%` (7.89% vs 26.32%)
- **False Negative Rate on AI Images (E3-Std - E1):** `+0.00%` (94.74% vs 94.74%)

---

## 5. Per-Image Prediction Registry

| Filename | Ground Truth | E1 Prob (AI) | E1 Pred | E1 Correct | E3-Std Prob (AI) | E3-Std Pred | E3-Std Correct |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `aadhi.jpeg` | **REAL** | `0.3191` | REAL | PASS | `0.0101` | REAL | PASS |
| `aswin.jpeg` | **REAL** | `0.0714` | REAL | PASS | `0.0006` | REAL | PASS |
| `aswin2.jpeg` | **REAL** | `0.4302` | REAL | PASS | `0.1688` | REAL | PASS |
| `dev1.jpeg` | **REAL** | `0.2925` | REAL | PASS | `0.4417` | REAL | PASS |
| `dev2.jpeg` | **REAL** | `0.9385` | AI | FAIL | `0.8818` | AI | FAIL |
| `Scene1.jpeg` | **REAL** | `0.2430` | REAL | PASS | `0.2217` | REAL | PASS |
| `scene2.jpeg` | **REAL** | `0.4983` | REAL | PASS | `0.0646` | REAL | PASS |
| `scene3.jpeg` | **REAL** | `0.6978` | AI | FAIL | `0.5449` | AI | FAIL |
| `scene4.jpeg` | **REAL** | `0.0024` | REAL | PASS | `0.0012` | REAL | PASS |
| `scene5.jpeg` | **REAL** | `0.0001` | REAL | PASS | `0.0001` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.11.35 PM (1).jpeg` | **REAL** | `0.5771` | AI | FAIL | `0.4846` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.11.35 PM.jpeg` | **REAL** | `0.8672` | AI | FAIL | `0.4827` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.12.08 PM.jpeg` | **REAL** | `0.0195` | REAL | PASS | `0.0034` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.12.09 PM (1).jpeg` | **REAL** | `0.0023` | REAL | PASS | `0.0047` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.12.09 PM.jpeg` | **REAL** | `0.0626` | REAL | PASS | `0.2690` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.12.30 PM.jpeg` | **REAL** | `0.6157` | AI | FAIL | `0.1962` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.13.02 PM.jpeg` | **REAL** | `0.0945` | REAL | PASS | `0.1770` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.13.06 PM.jpeg` | **REAL** | `0.4453` | REAL | PASS | `0.0311` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.13.16 PM.jpeg` | **REAL** | `0.2306` | REAL | PASS | `0.1346` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.13.41 PM.jpeg` | **REAL** | `0.8916` | AI | FAIL | `0.4973` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.14.51 PM.jpeg` | **REAL** | `0.4607` | REAL | PASS | `0.0403` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.15.05 PM.jpeg` | **REAL** | `0.7583` | AI | FAIL | `0.5791` | AI | FAIL |
| `WhatsApp Image 2026-08-31 at 10.15.46 PM.jpeg` | **REAL** | `0.1605` | REAL | PASS | `0.0353` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.16.11 PM.jpeg` | **REAL** | `0.4070` | REAL | PASS | `0.0276` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.16.21 PM.jpeg` | **REAL** | `0.1589` | REAL | PASS | `0.0225` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.16.32 PM (1).jpeg` | **REAL** | `0.5649` | AI | FAIL | `0.3462` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.16.32 PM.jpeg` | **REAL** | `0.5820` | AI | FAIL | `0.0244` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 10.16.54 PM.jpeg` | **REAL** | `0.0860` | REAL | PASS | `0.0037` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.00.40 PM.jpeg` | **REAL** | `0.0947` | REAL | PASS | `0.0368` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.01.28 PM.jpeg` | **REAL** | `0.3086` | REAL | PASS | `0.0608` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.01.51 PM.jpeg` | **REAL** | `0.3013` | REAL | PASS | `0.0295` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.02.11 PM.jpeg` | **REAL** | `0.0022` | REAL | PASS | `0.0003` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.02.37 PM.jpeg` | **REAL** | `0.0005` | REAL | PASS | `0.0012` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.03.13 PM.jpeg` | **REAL** | `0.2761` | REAL | PASS | `0.1277` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.03.34 PM.jpeg` | **REAL** | `0.2734` | REAL | PASS | `0.0226` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.03.45 PM.jpeg` | **REAL** | `0.6118` | AI | FAIL | `0.0703` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.04.11 PM.jpeg` | **REAL** | `0.0001` | REAL | PASS | `0.0001` | REAL | PASS |
| `WhatsApp Image 2026-08-31 at 11.04.43 PM.jpeg` | **REAL** | `0.0663` | REAL | PASS | `0.0126` | REAL | PASS |
| `0d28d39e-c5b6-4436-b476-e5d257ff4f96.jpg` | **AI** | `0.0014` | REAL | FAIL | `0.0000` | REAL | FAIL |
| `0e5c8694-f927-47cd-9206-b12af52c1c5b.jpg` | **AI** | `0.0368` | REAL | FAIL | `0.0279` | REAL | FAIL |
| `11185668-7b1c-4c51-b326-073b7a2624ab.jpg` | **AI** | `0.0151` | REAL | FAIL | `0.0138` | REAL | FAIL |
| `63b038f4-b540-4f60-ad08-d984826c48a8.jpg` | **AI** | `0.0120` | REAL | FAIL | `0.0854` | REAL | FAIL |
| `706d4ecc-bc97-4f47-b591-e5590b97203f.jpg` | **AI** | `0.0342` | REAL | FAIL | `0.0995` | REAL | FAIL |
| `830c25d9-bfc6-4cf9-abe4-7477c408ebdd.jpg` | **AI** | `0.0030` | REAL | FAIL | `0.0003` | REAL | FAIL |
| `bac7477a-e896-4c25-875e-d320e147d7e2.jpg` | **AI** | `0.0049` | REAL | FAIL | `0.0288` | REAL | FAIL |
| `c03c2063-adb4-489a-bc49-9185590b95da.jpg` | **AI** | `0.2474` | REAL | FAIL | `0.0349` | REAL | FAIL |
| `c28779be-ca74-4e18-bce1-4f66fac5338a.jpg` | **AI** | `0.0023` | REAL | FAIL | `0.0047` | REAL | FAIL |
| `c4ba4a96-2767-4796-be63-366623b61669.jpg` | **AI** | `0.1274` | REAL | FAIL | `0.0243` | REAL | FAIL |
| `e8d62f29-ec23-4a15-aa37-93b880293c09.jpg` | **AI** | `0.0320` | REAL | FAIL | `0.1416` | REAL | FAIL |
| `fcdc787f-8aa8-4821-829a-640024bf3893.jpg` | **AI** | `0.1821` | REAL | FAIL | `0.7563` | AI | PASS |
| `Gemini_Generated_Image_117fps117fps117f.png` | **AI** | `0.4526` | REAL | FAIL | `0.4766` | REAL | FAIL |
| `Gemini_Generated_Image_4iugfr4iugfr4iug.png` | **AI** | `0.0047` | REAL | FAIL | `0.0062` | REAL | FAIL |
| `Gemini_Generated_Image_4tm47s4tm47s4tm4.png` | **AI** | `0.0101` | REAL | FAIL | `0.0104` | REAL | FAIL |
| `Gemini_Generated_Image_5qg2pg5qg2pg5qg2.png` | **AI** | `0.2925` | REAL | FAIL | `0.0483` | REAL | FAIL |
| `Gemini_Generated_Image_6uqymu6uqymu6uqy.png` | **AI** | `0.0270` | REAL | FAIL | `0.3350` | REAL | FAIL |
| `Gemini_Generated_Image_6wg3ub6wg3ub6wg3.png` | **AI** | `0.0100` | REAL | FAIL | `0.0103` | REAL | FAIL |
| `Gemini_Generated_Image_7hjd827hjd827hjd.png` | **AI** | `0.0057` | REAL | FAIL | `0.0031` | REAL | FAIL |
| `Gemini_Generated_Image_9zfx8e9zfx8e9zfx.png` | **AI** | `0.0028` | REAL | FAIL | `0.0031` | REAL | FAIL |
| `Gemini_Generated_Image_aakexpaakexpaake.png` | **AI** | `0.7593` | AI | PASS | `0.1675` | REAL | FAIL |
| `Gemini_Generated_Image_am35wnam35wnam35.png` | **AI** | `0.0344` | REAL | FAIL | `0.0067` | REAL | FAIL |
| `Gemini_Generated_Image_fg4hufg4hufg4huf.png` | **AI** | `0.1761` | REAL | FAIL | `0.0444` | REAL | FAIL |
| `Gemini_Generated_Image_he9veuhe9veuhe9v.png` | **AI** | `0.6201` | AI | PASS | `0.2444` | REAL | FAIL |
| `Gemini_Generated_Image_i387pvi387pvi387.png` | **AI** | `0.0149` | REAL | FAIL | `0.0332` | REAL | FAIL |
| `Gemini_Generated_Image_i576uxi576uxi576.png` | **AI** | `0.0435` | REAL | FAIL | `0.0296` | REAL | FAIL |
| `Gemini_Generated_Image_kduz3lkduz3lkduz.png` | **AI** | `0.1990` | REAL | FAIL | `0.1833` | REAL | FAIL |
| `Gemini_Generated_Image_n84ugan84ugan84u.png` | **AI** | `0.0031` | REAL | FAIL | `0.0402` | REAL | FAIL |
| `Gemini_Generated_Image_nbsk7vnbsk7vnbsk.png` | **AI** | `0.1803` | REAL | FAIL | `0.1041` | REAL | FAIL |
| `Gemini_Generated_Image_qf4qlgqf4qlgqf4q.png` | **AI** | `0.0198` | REAL | FAIL | `0.1879` | REAL | FAIL |
| `Gemini_Generated_Image_so466wso466wso46.png` | **AI** | `0.0026` | REAL | FAIL | `0.0037` | REAL | FAIL |
| `Gemini_Generated_Image_ssgk4vssgk4vssgk.png` | **AI** | `0.0036` | REAL | FAIL | `0.0268` | REAL | FAIL |
| `Gemini_Generated_Image_tcnp2ltcnp2ltcnp.png` | **AI** | `0.0171` | REAL | FAIL | `0.5903` | AI | PASS |
| `Gemini_Generated_Image_tw0h6ptw0h6ptw0h.png` | **AI** | `0.1373` | REAL | FAIL | `0.0754` | REAL | FAIL |
| `Gemini_Generated_Image_u0p8xru0p8xru0p8.png` | **AI** | `0.0002` | REAL | FAIL | `0.0014` | REAL | FAIL |
| `Gemini_Generated_Image_xb4tusxb4tusxb4t.png` | **AI** | `0.0330` | REAL | FAIL | `0.0366` | REAL | FAIL |
| `Gemini_Generated_Image_yrcx5dyrcx5dyrcx.png` | **AI** | `0.0488` | REAL | FAIL | `0.1584` | REAL | FAIL |
| `images.jpg` | **AI** | `0.0189` | REAL | FAIL | `0.0136` | REAL | FAIL |
