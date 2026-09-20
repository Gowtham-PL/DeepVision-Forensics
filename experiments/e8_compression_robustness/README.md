# Experiment E8: Compression-Robust DeepVision Forensics

Contains training pipeline, checkpoints, and evaluation results for compression-aware multi-view training.

## Directory Layout
- [`augmentations.py`](./augmentations.py): On-the-fly in-memory realistic social media compression pipeline.
- [`smoke_test_e8.py`](./smoke_test_e8.py): Pre-training verification and runtime pilot.
- [`train_and_evaluate_e8.py`](./train_and_evaluate_e8.py): Complete training and evaluation runner.
- [`checkpoints/e8b_best_model.pt`](./checkpoints/e8b_best_model.pt): Best E8-B checkpoint.
- [`results/e8_study_results.json`](./results/e8_study_results.json): Full experimental metrics.
- [`results/per_image_predictions_whatsapp.csv`](./results/per_image_predictions_whatsapp.csv): WhatsApp test predictions.
- [`E8_RESEARCH_REPORT.md`](./E8_RESEARCH_REPORT.md): Comprehensive scientific research report.
