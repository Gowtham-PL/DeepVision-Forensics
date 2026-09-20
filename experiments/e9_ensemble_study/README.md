# Experiment E9: E6-C + E8-B Compression-Robust Ensemble Study

## Overview
Experiment E9 investigates the combination of the production model E6-C (trained on clean multiscale views) and E8-B (fine-tuned with realistic social-media compression augmentation) via continuous probability ensembling and conservative gated rules.

## Protocol Highlights
- **Inference-Only**: No model training or fine-tuning. Checkpoints frozen.
- **Development on E5 Validation**: All weights, candidate rules, and FPR-constrained thresholds were chosen on the clean E5 validation set ($N=5,775$).
- **Frozen Single-Shot WhatsApp Benchmark**: Evaluated once after permanently freezing the final rule.

## Frozen Winning Rule
- `Ensemble (w_e8=0.20)`
- Parameters: `['ensemble', 0.2, 0.5]`

## Artifacts in this Directory
- `run_e9_study.py`: Reproducible experimental runner.
- `e5_val_predictions.csv`: Model predictions on all 5,775 E5 validation samples.
- `candidate_ensemble_results.csv`: Continuous probability ensemble sweep.
- `candidate_gated_results.csv`: Systematic gated rule family sweep.
- `per_image_predictions_whatsapp.csv`: Single-shot WhatsApp test results ($N=67$).
- `hard_case_diagnostics.json`: Diagnostic outputs on hard-case images.
- `study_results.json`: Full machine-readable metrics and transition analysis.
- `E9_RESEARCH_REPORT.md`: Comprehensive formal research report.
