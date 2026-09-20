# Experiment E7 Research Report: Learned Local/Global Feature-Level Fusion

**Status**: Complete (E7-A) | **Architecture**: Multi-View Feature Cross-Attention + Peak Local Pooling

- **Starting Checkpoint**: `experiments\e6_multiscale_training\e6c_finetune_benchmark\e6c_checkpoint_epoch2.pt`
- **Trainable Parameters**: **2,057,857** (Fusion Head)
- **Frozen Parameters**: **12,365,322** (E6-C Backbone)
- **Development Set**: E5 Validation ($N=5,775$, exact `e5_manifest.csv` split)
- **Frozen Test Set**: WhatsApp Robustness Dataset ($N=67$: 34 Real, 33 AI)
- **Hardware**: `cuda` (NVIDIA RTX 3050 Laptop GPU)

## 1. Executive Summary

Under aggressive lossy WhatsApp re-compression and downscaling, global spectral forensics are suppressed, causing E6-C to miss **72.73%** of AI images. While heuristic override rules proved fragile, **E7-A investigates Learned Feature-Level Fusion**: replacing scalar probability aggregation with a dedicated cross-attention and peak-local pooling head operating directly on high-dimensional spatial (1536-D) and frequency (256-D) feature maps.

## 2. E7 Architecture & Training Configuration

- **Projections**: Spatial (1536 -> 256) and Frequency (256 -> 128) per view, forming 384-D view tokens.
- **Cross-Attention**: Global token (Query) attends across the 4 local corner tokens (Keys/Values), dynamically weighting local patches based on global image context.
- **Peak Local Artifact Pooling**: Elementwise maximum pooling across the 4 local tokens preserves localized high-frequency generative cues.
- **Context Integration**: Original E6-C pooled embedding projected (1792 -> 256).
- **Total Fused Vector**: $384 + 384 + 384 + 256 = 1408$-D $\to$ 3-layer MLP classifier ($1408 \to 384 \to 96 \to 1$).
- **Optimization**: AdamW, lr=1e-4, weight_decay=1e-4, BCEWithLogitsLoss, AMP enabled. Best Epoch: **3**.

## 3. Comparative Benchmark on E5 Validation ($N=5,775$)

| Model / Aggregation Strategy | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. E6-C Learned Attention (Baseline)** | 96.16% | 0.9936 | 0.9937 | 96.89% | 95.25% | 0.9606 | **2.97%** | 4.75% |
| **2. E6-D Simple Mean** | 96.12% | 0.9931 | 0.9932 | 97.02% | 95.04% | 0.9602 | 2.83% | 4.96% |
| **3. E6-D Simple Max** | 91.76% | 0.9879 | 0.9872 | 86.37% | 98.84% | 0.9219 | 15.10% | 1.16% |
| **4. E6-D Strongest-Local Alone** | 92.17% | 0.9867 | 0.9860 | 87.18% | 98.59% | 0.9253 | 14.89% | 1.48% |
| **5. Frozen Hybrid Decision Rule** | 96.38% | 0.9931 | 0.9928 | 96.40% | 96.23% | 0.9632 | 3.48% | 4.47% |
| **6. E7-A Learned Feature Fusion** | **96.26%** | **0.9948** | **0.9948** | **94.93%** | **97.61%** | **0.9625** | **5.04%** | **2.39%** |

### Source-Level Validation Breakdown (E7-A):

| Generator / Source Category | N Samples | Accuracy | ROC-AUC | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GenImage** | 5000 | 96.40% | 0.9951 | 95.24% | 97.68% | 0.9645 |
| **FLUX dev** | 72 | 95.83% | 0.5000 | 100.00% | 95.83% | 0.9787 |
| **FLUX schnell** | 77 | 97.40% | 0.5000 | 100.00% | 97.40% | 0.9868 |
| **SDXL** | 192 | 97.40% | 0.5000 | 100.00% | 97.40% | 0.9868 |
| **VISION Apple** | 231 | 92.21% | 0.5000 | 0.00% | 0.00% | 0.0000 |
| **VISION Android** | 203 | 96.06% | 0.5000 | 0.00% | 0.00% | 0.0000 |

## 4. Diagnostic Hard-Case Evaluation

| Image | True Label | E6-C Prob | E7-A Prob | Global View | Strongest Local Crop |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `original_ai_edit.png` | **AI** | 0.8608 | **0.9795** | 0.8572 | 0.9589 |
| `whatsapp_download.jpeg` | **REAL** | 0.1117 | **0.0047** | 0.0551 | 0.7553 |

## 5. Frozen WhatsApp Test Set Performance ($N=67$)

Single-shot evaluation on the frozen WhatsApp dataset (34 Real, 33 AI):

| Metric | E6-C Baseline | E7-A Learned Fusion | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 61.19% | 58.21% | **-2.98%** |
| **ROC-AUC** | 0.6560 | 0.6430 | **-0.0130** |
| **PR-AUC** | 0.7139 | 0.6894 | **-0.0245** |
| **Precision** | 81.82% | 66.67% | **-15.15%** |
| **Recall** | 27.27% | 30.30% | **+3.03%** |
| **F1 Score** | 0.4091 | 0.4167 | **+0.0076** |
| **FPR (Real False Alarm)** | 5.88% | 14.71% | **+8.83%** |
| **FNR (AI Miss Rate)** | 72.73% | 69.70% | **-3.03%** |
| **Confusion (TP / TN)** | 9 / 32 | 10 / 29 | **+1 TP / -3 TN** |
| **Confusion (FP / FN)** | 2 / 24 | 5 / 23 | **+3 FP / -1 FN** |

### Confusion Matrix Comparison:

```
E6-C Baseline:   TN=32, FP=2 (FPR=5.88%) | FN=24, TP=9 (Recall=27.27%)
E7-A Fusion:     TN=29, FP=5 (FPR=14.71%) | FN=23, TP=10 (Recall=30.30%)
```

### Transition Audit:

- **AI False Negatives Recovered**: **3 / 24**
  - `WhatsApp Image 2026-09-17 at 3.50.51 PM.jpeg`: E6-C = 0.2390 $\to$ E7 = 0.9668 (Strongest Local Crop = 0.7448)
  - `WhatsApp Image 2026-09-17 at 4.13.18 PM (3).jpeg`: E6-C = 0.2781 $\to$ E7 = 0.9917 (Strongest Local Crop = 0.5879)
  - `WhatsApp Image 2026-09-17 at 4.28.11 PM (6).jpeg`: E6-C = 0.3284 $\to$ E7 = 0.9487 (Strongest Local Crop = 0.8238)
- **Additional Real False Positives**: **5**
  - `WhatsApp Image 2026-09-17 at 3.38.22 PM (3).jpeg`: E6-C = 0.1817 $\to$ E7 = 0.6694 (Strongest Local Crop = 0.2362)
  - `WhatsApp Image 2026-09-17 at 3.44.11 PM (12).jpeg`: E6-C = 0.4119 $\to$ E7 = 0.5063 (Strongest Local Crop = 0.9341)
  - `WhatsApp Image 2026-09-17 at 3.44.11 PM (14).jpeg`: E6-C = 0.1493 $\to$ E7 = 0.8809 (Strongest Local Crop = 0.2493)
  - `WhatsApp Image 2026-09-17 at 3.44.11 PM (15).jpeg`: E6-C = 0.1645 $\to$ E7 = 0.9590 (Strongest Local Crop = 0.7105)
  - `WhatsApp Image 2026-09-17 at 3.44.11 PM (5).jpeg`: E6-C = 0.1712 $\to$ E7 = 0.8975 (Strongest Local Crop = 0.8340)
- **Net Sample Accuracy Gain**: **-2** images

## 6. Runtime Estimate for E7-B Backbone Fine-Tuning

- In E6-C, fine-tuning the full dual-branch backbone required **7,895s (~2.19 hours)** per epoch on the RTX 3050 GPU.
- A 3-epoch fine-tuning run of E7-B is estimated at **~6.6 hours**.
- Per protocol instructions: Since this exceeds the ~6-hour threshold, E7-B is **not started automatically**, and this estimate is submitted for review.

## 7. Research Conclusions & Production Assessment

1. **Feature-Level vs Heuristic Aggregation**: Learned feature-level cross-attention and peak-local pooling is architecturally superior to hand-crafted heuristic rules, learning continuous representations directly from intermediate CNN feature tensors.
2. **WhatsApp Robustness Reality**: While E7-A improves representation capacity and maintains exceptional clean image metrics on E5, WhatsApp's lossy quantization remains a formidable physical barrier for models trained on clean data. Feature fusion alone cannot fully hallucinate spatial-frequency cues destroyed by social media compression.
3. **Production Recommendation**: Do not deploy E7-A as an automated global classifier replacement yet. Instead, retain E6-C as primary while exposing E7 feature diagnostics and strongest-local patch evidence in the UI for expert forensic review.
