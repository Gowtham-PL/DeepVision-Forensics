# Experiment E10-A: WhatsApp Forensic Domain-Shift Diagnostic

## Overview
Experiment E10-A is an empirical, inference-only diagnostic study investigating the forensic and signal transformations induced by social-media transmission (specifically WhatsApp).

## Key Deliverables
- `E10A_DIAGNOSTIC_REPORT.md`: Comprehensive formal research report answering all 10 mandatory scientific questions.
- `diagnostic_summary.json`: Complete machine-readable statistics, dimensions, frequency profiles, and correlations.
- `image_metadata.csv`: Full format and container forensics for all 67 benchmark images.
- `frequency_statistics.csv`: 2D FFT spectral metrics (HF energy, spectral centroid, roll-off).
- `model_response_analysis.csv`: Correlated model probabilities and forensic metrics.
- `transformation_ablation.csv`: Controlled ablation of resizing, JPEG, and compounding stages on hard cases.
- `error_analysis.csv`: Deep-dive audit into false negatives and false positives.
- `hard_case_decomposition.csv`: Step-by-step diagnostic breakdown of the two hard cases.
- `plots/`: 7 publication-quality diagnostic visualizations.
