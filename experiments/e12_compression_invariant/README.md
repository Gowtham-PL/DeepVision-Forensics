# Experiment E12-A: Compression-Invariant Feature Regularization

## Summary
Experiment E12-A trained the multi-view attention and classification heads of `MultiViewE5Model` using paired clean/compressed inputs and a feature-consistency regularization loss ($\lambda = 0.05$).

## Key Findings
- **Clean E5 Validation ($N=5,775$)**: Accuracy = 96.48%, ROC-AUC = 0.9929, PR-AUC = 0.9920, FPR = 4.19%, F1 = 0.9645.
- **Frozen WhatsApp Benchmark ($N=67$)**: Accuracy = 61.19%, ROC-AUC = 0.6840, PR-AUC = 0.7418, Recall = 39.39%, FPR = 17.65%, F1 = 0.5000.

## Artifacts in this Directory
- `train_and_evaluate_e12a.py`: Master reproducible training script.
- `training_config.json`: Hyperparameters and configuration.
- `training_history.csv`: Per-epoch loss and clean validation metrics.
- `e5_val_predictions.csv`: Predictions on clean E5 validation set for winning epoch.
- `whatsapp_predictions.csv`: Single-shot WhatsApp evaluation per-image predictions.
- `whatsapp_transition_analysis.csv`: Detailed transition audit against E10-B.
- `hard_case_diagnostics.json`: Diagnostics for benchmark hard cases.
- `E12A_RESEARCH_REPORT.md`: Comprehensive formal research report.
- `checkpoints/`: Checkpoints saved for each epoch and `e12a_best_model.pt`.
