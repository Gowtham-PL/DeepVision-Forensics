# WhatsApp Robustness: Hybrid Decision Rule Aggregation Study

This directory contains the complete artifacts, methodology, candidate search grid, and single-shot evaluation results for the **Controlled Hybrid Decision Rule Aggregation Study** on DeepVision E6-C.

## Key Documents & Data

1. [`HYBRID_AGGREGATION_STUDY_REPORT.md`](./HYBRID_AGGREGATION_STUDY_REPORT.md): Comprehensive report with executive summary, methodology, full metrics tables, confusion matrices, and error analysis.
2. [`candidate_rules_e5_val.csv`](./candidate_rules_e5_val.csv): Tabular evaluation of all candidate hybrid rules evaluated on the E5 validation set ($N=5,775$).
3. [`per_image_predictions_whatsapp.csv`](./per_image_predictions_whatsapp.csv): Per-image prediction audit on the frozen WhatsApp dataset ($N=67$).
4. [`study_results.json`](./study_results.json): Structured JSON data containing all development and test results.
5. [`run_aggregation_study.py`](./run_aggregation_study.py): Executable evaluation script reproducing all findings.
