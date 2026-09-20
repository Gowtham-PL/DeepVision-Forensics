# E25 Pre-Training Smoke Test Report

**Execution Date:** 2026-09-20 00:40:04  
**Status:** PASSED

## 1. Test Dataset (20 Balanced Non-Benchmark Samples)
- Real Clean: 5 images (E20/E21 data)
- AI Clean: 5 images (E20/E21 data)
- Real Compressed: 5 images (E20/E21 data)
- AI Compressed: 5 images (E20/E21 data)
- Total images: 20 (5 views per image = 100 view tensors)

## 2. Model & Verification Checklist
- **VAE Reconstruction:** Successfully executed for all 100 views.
- **Residual Integrity:** Non-zero (`all_nonzero = True`), finite (`all_finite = True`).
- **Frequency Guided Residual:** Deterministic radial mask ($0.10 \le r \le 0.50$) applied successfully.
- **5-View Attention:** Output dimensions `(B, 5, 1)` and aggregated `(B, 512)` verified.
- **Global View Preservation:** Explicit concatenation with attention embedding verified `(B, 1024)`.
- **Classification Logits:** Finite scalar output per sample `(B, 1)`.
- **Backward Pass:** Successfully computed gradients.
- **VAE Frozen Check:** `vae_has_grad = False` (Pass).
- **Backbone Frozen Check (Phase 1):** `backbone_has_grad = False` (Pass).
- **Trainable Modules Gradients:** {'residual_cnn': True, 'spatial_proj': True, 'freq_proj': True, 'res_proj': True, 'view_attention': True, 'fusion_mlp': True, 'classifier': True}

## 3. Hardware & Performance
- **Peak VRAM:** 3877.76 MB / 4096 MB (Well within 4GB limit).
- **Throughput:** 0.38 images/sec (1.9 views/sec).
- **Estimated Epoch Time (6,279 train images):** 278.0 minutes.

## 4. Visualizations Saved
- `experiments/e25_reconstruction_residual/visualizations/residual_diag_real_clean.png`
- `experiments/e25_reconstruction_residual/visualizations/residual_diag_ai_clean.png`
- `experiments/e25_reconstruction_residual/visualizations/residual_diag_real_compressed.png`
- `experiments/e25_reconstruction_residual/visualizations/residual_diag_ai_compressed.png`
