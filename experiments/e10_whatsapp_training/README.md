# Experiment E10-B: WhatsApp-Aware Multi-View Attention Training

## Overview
Experiment E10-B trains a WhatsApp-aware version of the E6-C multi-view attention head and classifier with frozen feature extraction backbones, using in-memory symmetric social-media compression augmentation.

## Key Results (Best Epoch: 2)
- **Clean E5 Val Accuracy**: 96.62% (ROC-AUC: 0.9936, FPR: 3.27%)
- **WhatsApp Test Accuracy**: 62.69% (ROC-AUC: 0.7001, PR-AUC: 0.7519)
- **WhatsApp Real FPR**: 17.65% (2 FP vs. 5 FP in E8-B)
- **WhatsApp AI Recall**: 42.42%

## Deliverables in this Directory
- `train_and_evaluate_e10b.py`: Master reproducible training script.
- `training_config.json`: Configuration and hyperparameters.
- `training_history.csv`: Per-epoch metrics across the 3 epochs.
- `e5_val_predictions.csv`: Clean validation predictions.
- `per_image_predictions_whatsapp.csv`: WhatsApp test benchmark per-image audit.
- `hard_case_diagnostics.json`: Diagnostic outputs on hard cases.
- `study_results.json`: Full machine-readable experimental metrics.
- `E10B_RESEARCH_REPORT.md`: Comprehensive formal research report answering all 8 mandatory questions.
- `checkpoints/`: Epoch checkpoints and `e10b_best_model.pt`.
