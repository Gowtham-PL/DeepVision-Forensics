# DeepVision Forensics Architecture

## System Overview
DeepVision Forensics is a single-image AI-generated image forensic analysis system. It leverages a dual-branch machine learning architecture combining spatial and frequency domain analysis to detect AI-generated images, provide visual explainability, and assess security risks.

## Core Components

### 1. Frontend (React / TypeScript / Tailwind CSS)
A modern, dark-first interface providing an analytical workspace for forensic image analysis.
- **Image Workspace:** Upload and visualize the single input image.
- **Forensic Report Dashboard:** Presents the classification result, AI probability, and evidence summaries.
- **Explainability View:** Overlays Grad-CAM heatmaps on the original image to explain spatial evidence.
- **Risk Indicator:** Clearly displays the application-level security risk (LOW, MEDIUM, HIGH, CRITICAL).

### 2. Backend (FastAPI / Python)
A REST API handling requests from the frontend, orchestrating the ML inference, and formatting the forensic report.
- **Image Ingestion:** Validates and stores the uploaded image temporarily for processing.
- **Inference Orchestration:** Passes the image to the PyTorch inference pipeline and gathers outputs.
- **Report Generation:** Compiles classification, probability, and spatial/frequency evidence into a structured JSON response.
- **Database (PostgreSQL - Optional):** Future integration for storing analysis history, reports, and experiment records.

### 3. ML Pipeline (PyTorch)
The core analytical engine based on a dual-branch architecture.
- **Spatial Branch:** EfficientNet-B3 extracting deep visual features.
- **Frequency Branch:** FFT/DCT-based feature extractor to identify spectral artifacts introduced by AI generators.
- **Feature Fusion:** Concatenates or otherwise combines spatial and frequency features.
- **Classifier:** A fully connected network producing the final "Real vs. AI-generated" probabilities.
- **Explainability Module:** Applies Grad-CAM to the EfficientNet-B3 branch to generate heatmaps.

## Data Flow
1. **User Upload:** User uploads a single image via the React frontend.
2. **API Reception:** FastAPI receives the image and runs basic validation.
3. **Preprocessing:** Image is transformed (resized, normalized) for the ML model.
4. **Dual Analysis:** 
   - Image passes through EfficientNet-B3 (Spatial).
   - Image passes through FFT/DCT extraction (Frequency).
5. **Fusion & Classification:** Features are fused, and the classifier outputs the AI probability.
6. **Explanation Generation:** Grad-CAM generates a heatmap based on the spatial branch.
7. **Report Construction:** Backend constructs an authenticity assessment and risk indicator based on the probability and evidence.
8. **Result Display:** Frontend visualizes the report, evidence, and Grad-CAM overlay to the user.
