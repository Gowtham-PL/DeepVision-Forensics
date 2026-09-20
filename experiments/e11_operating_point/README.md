# Experiment E11: E10-B Architecture Audit + Operating-Point Study

## Overview
Experiment E11 performs an exhaustive architectural audit of E10-B (`e10b_best_model.pt`) and evaluates threshold operating points established strictly on the clean E5 validation dataset ($N=5,775$), followed by a single-shot test on the frozen WhatsApp benchmark ($N=67$).

## Predefined Frozen Operating Points (from Clean E5 Val)
- **Candidate A (FPR <= 3%)**: Tau = 0.60
- **Candidate B (FPR <= 5%)**: Tau = 0.30
- **Candidate C (FPR <= 7.5%)**: Tau = 0.20
- **Candidate D (Max F1)**: Tau = 0.55

## Artifacts in this Directory
- `run_e11_study.py`: Master reproducible evaluation script.
- `E10B_ARCHITECTURE_AUDIT.md`: In-depth parameter breakdown and state dict integrity audit.
- `E11_OPERATING_POINT_REPORT.md`: Formal operating-point research report answering all 10 mandatory questions.
- `architecture_parameter_audit.json`: Machine-readable parameter counts and module breakdown.
- `e10b_e5_predictions.csv`: E10-B predictions on clean E5 validation ($N=5,775$).
- `e5_threshold_sweep.csv`: Full E5 validation threshold sweep across tau in [0.20, 0.80].
- `frozen_operating_points.json`: Formal definitions of frozen operating-point candidates.
- `whatsapp_operating_point_results.csv`: WhatsApp test results for all thresholds and candidates.
- `whatsapp_transition_analysis.csv`: Detailed transition analysis relative to tau=0.50.
- `hard_case_threshold_analysis.csv`: Hard-case diagnostic outputs across thresholds.
