# Experiment E7: Learned Local/Global Feature-Level Fusion

This directory contains the code, checkpoints, evaluation results, and research report for **Experiment E7**.

## Directory Layout
- [`models/e7_fusion_model.py`](./models/e7_fusion_model.py): Model definition for E7LearnedFusionModel.
- [`checkpoints/e7a_best_model.pt`](./checkpoints/e7a_best_model.pt): Best model checkpoint from E7-A training.
- [`results/e7_study_results.json`](./results/e7_study_results.json): Structured JSON results.
- [`results/per_image_predictions_whatsapp.csv`](./results/per_image_predictions_whatsapp.csv): Per-image WhatsApp test predictions.
- [`E7_RESEARCH_REPORT.md`](./E7_RESEARCH_REPORT.md): Comprehensive research report.
- [`smoke_test_e7.py`](./smoke_test_e7.py): Pre-training verification smoke test.
- [`train_and_evaluate_e7a.py`](./train_and_evaluate_e7a.py): Full reproducible pipeline script.
