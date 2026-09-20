# E21 Training & Validation Experiment Report

**Experiment**: E21 Targeted Data-Balancing Fine-Tuning  
**Date**: September 19, 2026  
**Status**: Completed and Frozen  
**Starting Checkpoint**: `experiments/e20_training/checkpoints/e20_best_model.pt`  
**Best Checkpoint Selected**: `experiments/e21_targeted_data/checkpoints/e21_best_model.pt` (Epoch 4)  
**Evaluation Threshold**: Strictly 0.50  

---

## 1. Executive Summary & Training Progression

The objective of **Experiment 21 (E21)** was to fine-tune the high-performing E20 multi-scale forensic detector using a targeted data-balancing curriculum:
1. **Restoring Legacy Generative Model Sensitivity**: Direct exposure to early latent diffusion models (Stable Diffusion 1.3, 1.4, 2, Adobe Firefly 1, DALL-E 2).
2. **Defocus & Low-Edge Invariance**: Incorporating natural shallow depth-of-field, indoor casual, and low-light smartphone photography.
3. **Preserving Modern Smartphone Invariance**: Retaining high sensitivity on FLUX.1 [dev], Google Gemini, DALL-E 3, SDXL, and Pixel/iPhone/Samsung mobile captures.

### Training Progression Table

| Epoch | Phase | Train Loss | Dev Acc | Dev ROC-AUC | Dev F1 | Dev FPR | Dev AI Recall | SD 1.3 Rec | Firefly Rec | Pixel FPR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | phase1 | 0.6602 | 77.96% | 0.8635 | 0.7861 | 25.12% | 81.03% | 88.1% | 67.1% | 11.2% |
| 2 | phase1 | 0.4122 | 81.40% | 0.9061 | 0.8286 | 27.09% | 89.90% | 94.0% | 81.4% | 18.4% |
| 3 | phase2 | 0.3333 | 86.21% | 0.9434 | 0.8654 | 16.26% | 88.67% | 94.0% | 85.7% | 5.1% |
| 4 | phase2 | 0.2662 | 88.18% | 0.9529 | 0.8884 | 17.73% | 94.09% | 92.9% | 97.1% | 4.1% |
| 5 | phase2 | 0.2277 | 87.81% | 0.9563 | 0.8828 | 16.26% | 91.87% | 94.0% | 92.9% | 2.0% |

---

## 2. Checkpoint Selection

Selected **Epoch 4** as `e21_best_model.pt` strictly following pre-registered E21 Dev selection criteria:
- **Primary (Dev F1)**: **0.8884**
- **Tie-breaker (Dev ROC-AUC)**: **0.9529**
- **False Alarm Rate (Dev FPR)**: **17.73%**
- **Detection Sensitivity (Dev Recall)**: **94.09%**
