# Paired Hard Case Diagnostic

This document presents a controlled diagnostic investigation evaluating the performance of the **DeepVision-E5 Production Model** (`experiments/e5_generalization/best_model.pt`) on a paired real-world hard case: an original AI-edited photograph versus its WhatsApp-downloaded (JPEG compressed) counterpart.

---

## 1. Input Images

The paired test set consists of two versions of the same photographic subject located under `data/e6_hard_cases/`:

1. **Original AI-Edited Image (`data/e6_hard_cases/original_ai_edit.png`):**
   - **Format:** Uncompressed Portable Network Graphics (PNG)
   - **Dimensions:** $941 \times 1672$ pixels (Aspect Ratio $9:16 = 0.5628$)
   - **File Size:** $2,229,907$ bytes ($2.127\text{ MB}$ / $2,177.64\text{ KB}$)
   - **Visual Content:** High-resolution portrait photograph containing localized generative AI editing.

2. **WhatsApp-Downloaded Version (`data/e6_hard_cases/whatsapp_download.jpeg`):**
   - **Format:** Joint Photographic Experts Group (JPEG)
   - **Dimensions:** $900 \times 1600$ pixels (Aspect Ratio $9:16 = 0.5625$, slightly resampled by WhatsApp)
   - **File Size:** $170,945$ bytes ($0.163\text{ MB}$ / $166.94\text{ KB}$)
   - **Compression Reduction:** **$13.04\times$ file size reduction** due to WhatsApp lossy JPEG re-encoding and chroma subsampling.

---

## 2. E5 Results

Both images were evaluated independently through the exact production inference pipeline (PyTorch model `DeepVisionFusionModel`, CUDA acceleration, standard bilinear resizing to $224 \times 224$, and decision threshold $0.50$).

| Diagnostic Metric | Original AI Edit (`original_ai_edit.png`) | WhatsApp Download (`whatsapp_download.jpeg`) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Image Format** | PNG | JPEG | Format shift |
| **Dimensions ($W \times H$)** | $941 \times 1672$ | $900 \times 1600$ | $41 \times 72$ downsampled |
| **File Size** | $2,177.64\text{ KB}$ ($2.127\text{ MB}$) | $166.94\text{ KB}$ ($0.163\text{ MB}$) | $-92.3\%$ size reduction |
| **Preprocessed Tensor Shape** | `[1, 3, 224, 224]` | `[1, 3, 224, 224]` | Identical |
| **Inference Latency** | $26.035\text{ ms}$ | $26.605\text{ ms}$ | $+0.57\text{ ms}$ |
| **Raw Model Logit** | **-2.899407** | **-6.544741** | **-3.645334** |
| **E5 AI Probability** | **0.052183 (5.22%)** | **0.001436 (0.14%)** | **-0.050747 (-5.07 pp)** |
| **Predicted Class (@ 0.50)** | **Real** ❌ *(False Negative)* | **Real** ❌ *(False Negative)* | Both classified Real |
| **Spatial Embedding $L_2$ Norm** | 11.0226 | 11.0768 | $+0.0542$ |
| **Frequency Embedding $L_2$ Norm** | 16.6604 | 16.6610 | $+0.0006$ |

---

## 3. Probability Difference

* **Original AI Edit Probability:** $0.052183$ ($5.22\%$)
* **WhatsApp Download Probability:** $0.001436$ ($0.14\%$)
* **Absolute Probability Difference:** **$0.050747$**
* **Percentage-Point Change:** **$-5.0747\text{ percentage points}$** (AI score **decreased**)
* **Relative Confidence Collapse:** WhatsApp JPEG re-encoding caused a **$-97.25\%$ relative drop** in synthetic probability, driving the model logit from $-2.8994 \to -6.5447$.

---

## 4. Diagnosis

### A. Does E5 detect the original AI-edited image?
**NO.** E5 assigns an AI probability of only **$5.22\%$** ($0.052183$) to `original_ai_edit.png`, misclassifying it as **Real** (False Negative).

### B. Does E5 detect the WhatsApp version?
**NO.** E5 assigns an AI probability of only **$0.14\%$** ($0.001436$) to `whatsapp_download.jpeg`, misclassifying it as **Real** (False Negative).

### C. Did WhatsApp/JPEG processing materially reduce the AI probability?
**YES.** WhatsApp JPEG re-encoding and spatial resampling materially reduced the AI probability from $5.22\% \to 0.14\%$ (a **$-5.07$ percentage-point drop** and a **$-97.25\%$ relative confidence collapse**). The decision logit shifted further into the negative (authentic) domain by $-3.645$ units.

### D. Is the dominant failure mode AI-editing recognition, compression/transmission robustness, preprocessing/downsampling, or a combination?
**The failure is caused by a COMBINATION OF ALL THREE FACTORS, structured as follows:**

1. **Primary Baseline Driver — Localized AI Editing vs. Full-Image Generation:**
   * E5 was trained primarily on whole-image generative outputs (GenImage + Synthbuster SDXL/FLUX).
   * Localized AI editing (in-painting, background editing, or localized face/object manipulation) leaves large authentic photographic regions intact.
   * Because the spatial branch (EfficientNet-B3) aggregates global features via Global Average Pooling (GAP) and the 2D FFT spectral branch computes global log-magnitude energy across the full image, the dominant authentic background features mask localized synthetic traces. Consequently, even the uncompressed PNG original receives an AI score of only **$5.22\%$**.

2. **Structural Amplifier — Preprocessing Aspect Ratio Distortion & Downsampling:**
   * Both images have a tall portrait aspect ratio ($9:16 \approx 0.56$).
   * Standard production preprocessing applies direct bilinear resizing to $224 \times 224$ without aspect-ratio preservation or cropping.
   * This squashes the image vertically by a factor of $7.46\times$, distorting high-frequency spatial structures and blurring boundary artifacts at localized edit seams.

3. **Secondary Degrader — WhatsApp JPEG Re-Compression:**
   * WhatsApp downsamples the pixel grid ($941 \times 1672 \to 900 \times 1600$) and applies lossy JPEG quantization ($2.13\text{ MB} \to 0.16\text{ MB}$, a $13\times$ compression ratio).
   * High-frequency residual traces are attenuated by JPEG quantization matrices, shifting the 2D FFT spectral embedding and driving the logit from $-2.899 \to -6.545$ ($0.14\%$ AI probability).

---

## 5. Recommendation for E6

Based strictly on this diagnostic finding, the recommended research directions for **Experiment E6** (ranked by expected value) are:

1. **Multi-Scale Tiling / Patch-Based Preprocessing (Highest EV):**
   * Process $224 \times 224$ crops extracted at native resolution in addition to global downsampled views. Patch-based evaluation isolates localized AI edits (preventing background pooling dilution) and avoids high-ratio downsampling distortion.
2. **Aspect-Ratio-Preserving Preprocessing & High-Res Input ($512 \times 512$):**
   * Replace forced $224 \times 224$ distortion with center-crop or padded resizing at $512 \times 512$ resolution to retain $4\times$ more high-frequency spatial and spectral detail.
3. **Robustness Fine-Tuning with Social Media Re-Compression Data:**
   * Include WhatsApp and Web-compressed paired images during training to insulate the spectral branch against JPEG quantization shifts.

*(Do not implement these recommendations yet. This document is diagnostic only.)*
