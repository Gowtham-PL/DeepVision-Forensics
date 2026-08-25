# DeepVision Forensics - Development Roadmap

## Roadmap Overview
The project will be built incrementally, verifying every major stage to ensure a genuinely working research prototype. 

### Phase 0 — Architecture and Project Setup (CURRENT)
- Create architecture, API, ML, and development plan documentation.
- Establish initial folder structure.
- Define core tech stack and dependencies.

### Phase 1 — Dataset Pipeline
- **Goal:** Set up secure, reproducible data loading and preprocessing.
- **Tasks:** 
  - Write PyTorch Dataset/DataLoader classes.
  - Implement strict Train/Val/Test splits (70/15/15).
  - Test data pipeline with a small subset before full scaling.

### Phase 2 — EfficientNet-B3 Baseline Detector
- **Goal:** Implement the spatial branch of the model.
- **Tasks:** 
  - Load pretrained EfficientNet-B3.
  - Modify classification head for our task.
  - Set up initial training loop and validation metrics.

### Phase 3 — Frequency-Domain Feature Extraction
- **Goal:** Implement FFT/DCT extraction.
- **Tasks:**
  - Create the frequency analysis module.
  - Test spectral output on known real/fake samples.

### Phase 4 — Spatial + Frequency Feature Fusion
- **Goal:** Combine both branches into the final architecture.
- **Tasks:**
  - Implement concatenation and linear layers.
  - Train the combined architecture.

### Phase 5 — Model Evaluation
- **Goal:** Thoroughly test the baseline and fused models.
- **Tasks:**
  - Generate evaluation metrics (Accuracy, F1, ROC-AUC).
  - Compare spatial-only vs. fused model performance.

### Phase 6 — Grad-CAM Explainability
- **Goal:** Provide spatial evidence visualization.
- **Tasks:**
  - Implement Grad-CAM hooking into EfficientNet-B3.
  - Generate and save heatmaps.

### Phase 7 — Robustness Experiments
- **Goal:** Test robustness against JPEG compression, resizing, blur and noise. Optional screenshot/social-media simulation may be evaluated if time permits.
- **Tasks:**
  - Apply JPEG compression, resizing, blur, and noise.
  - Evaluate and document model degradation.
  - *Future Scope:* Adversarial perturbation defense is explicitly kept as future scope.

### Phase 8 — Cross-Generator Evaluation
- **Goal:** Evaluate generalization to unseen AI generators.
- **Tasks:**
  - Run inference on the hold-out generator dataset.
  - Document generalization performance.

### Phase 9 — FastAPI Backend
- **Goal:** Serve the model via a REST API.
- **Tasks:**
  - Build `POST /analyze` endpoint.
  - Integrate inference script.
  - Generate structured JSON reports.

### Phase 10 — React Frontend
- **Goal:** Build the forensic UI workspace.
- **Tasks:**
  - Initialize React + TS + Tailwind.
  - Build image upload component.
  - Build results dashboard (Classification, Heatmaps).

### Phase 11 — Forensic Report and Risk Indicator
- **Goal:** Finalize the user-facing output logic.
- **Tasks:**
  - Implement Risk Indicator thresholds (LOW/MEDIUM/HIGH/CRITICAL).
  - Format textual evidence summaries.

### Phase 12 — Integration
- **Goal:** End-to-end system test.
- **Tasks:**
  - Connect Frontend to Backend.
  - Ensure heatmaps and data flow seamlessly.

### Phase 13 — Testing and Optimization
- **Goal:** Refine and optimize.
- **Tasks:**
  - Write unit/integration tests for API and ML pipelines.
  - Optimize inference time.

### Phase 14 — Deployment (Optional)
- **Goal:** Prepare for production/demo.
- **Tasks:**
  - Dockerize backend and frontend.

## Technical Risks & Open Decisions
1. **Feature Dimensionality Alignment:** How to best align the highly dense spatial features of EfficientNet-B3 with the potentially distinct dimensionality of FFT/DCT features before fusion.
2. **Dataset Bias:** Ensuring the dataset does not contain unintended shortcuts (e.g., all fake images being a specific resolution or format).
3. **Training Resources:** The fused model might require significant VRAM; batch sizes and potential gradient accumulation need to be managed.

## Next Step Execution (Implementation Order)
The immediate next step after approval is **Phase 1: Dataset Pipeline**. This will involve defining the PyTorch `Dataset` classes and preprocessing logic, ensuring we can load images reliably before touching any model architecture.
