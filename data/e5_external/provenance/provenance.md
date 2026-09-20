# E5 External Dataset Provenance and Licensing Documentation

This document records the exact sources, licensing, provenance, and acquisition parameters for the external dataset subsets designed for **Experiment E5: Modern-Generator Domain Generalization**.

---

## 1. Modern AI Sources

### A. FLUX.1 [dev]
- **Source Repository:** Hugging Face Datasets — `LukasT9/Flux-1-Dev-Images-1k`
- **URL:** [https://huggingface.co/datasets/LukasT9/Flux-1-Dev-Images-1k](https://huggingface.co/datasets/LukasT9/Flux-1-Dev-Images-1k)
- **Model Architecture:** Black Forest Labs FLUX.1 [dev] (12B-parameter 24-layer MMDiT Rectified Flow Transformer).
- **Generation Parameters:** 50 inference steps, guidance scale 3.5, native $1024 \times 1024$ resolution.
- **License:** Apache License 2.0 (Verified permissive open license).
- **Academic / Model Training Rights:** Fully permitted.
- **Acquisition Plan:**
  - 500 images allocated to `train` split (`data/e5_external/modern_ai/flux_dev/`).
  - 250 images allocated to `val` split (`data/e5_external/modern_ai/flux_dev/`).

### B. FLUX.1 [schnell]
- **Source Repository:** Hugging Face Datasets — `LukasT9/Flux-1-Schnell-Images-1k`
- **URL:** [https://huggingface.co/datasets/LukasT9/Flux-1-Schnell-Images-1k](https://huggingface.co/datasets/LukasT9/Flux-1-Schnell-Images-1k)
- **Model Architecture:** Black Forest Labs FLUX.1 [schnell] (Distilled 4-step Rectified Flow Transformer).
- **Generation Parameters:** 4 inference steps, native $1024 \times 1024$ resolution.
- **License:** Apache License 2.0 (Verified permissive open license).
- **Academic / Model Training Rights:** Fully permitted.
- **Acquisition Plan:**
  - 500 images allocated to `train` split (`data/e5_external/modern_ai/flux_schnell/`).
  - 250 images allocated to `val` split (`data/e5_external/modern_ai/flux_schnell/`).

### C. Stable Diffusion XL (SDXL)
- **Source Repository:** Zenodo Record 10066460 (Synthbuster benchmark)
- **URL:** [https://zenodo.org/records/10066460](https://zenodo.org/records/10066460) (DOI: `10.5281/zenodo.10066460`)
- **Citation:** Quentin Bammey, *"Synthbuster: Towards Detection of Diffusion Model Generated Images"*, IEEE Open Journal of Signal Processing, 2023.
- **Model Architecture:** Stability AI Stable Diffusion XL 1.0 (Base 3.5B LDM).
- **Generation Parameters:** Prompts derived via CLIP interrogator on RAISE-1k raw image dataset with manual curation for photorealism. Uncompressed PNG, $1024 \times 1024$.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Academic / Model Training Rights:** Permitted for academic research and non-commercial model development.
- **Acquisition Plan:**
  - 1,000 images allocated to `train` split (`data/e5_external/modern_ai/sdxl/`).

### D. DALL-E 3 (Strictly Evaluation-Only)
- **Source Repository:** Zenodo Record 10066460 (Synthbuster benchmark)
- **URL:** [https://zenodo.org/records/10066460](https://zenodo.org/records/10066460) (DOI: `10.5281/zenodo.10066460`)
- **Citation:** Quentin Bammey, *"Synthbuster: Towards Detection of Diffusion Model Generated Images"*, 2023.
- **Model Architecture:** OpenAI DALL-E 3 (Autoregressive prompt upsampler + diffusion decoder).
- **Generation Parameters:** Same standardized RAISE-1k prompts. Uncompressed PNG, $1024 \times 1024$.
- **License:** CC BY-NC-SA 4.0.
- **Academic / Model Training Rights:** Permitted for research evaluation.
- **CRITICAL ISOLATION RULE:** DALL-E 3 is **strictly quarantined as an unseen modern evaluation benchmark**. Zero DALL-E 3 images will be present in training or validation splits.
- **Acquisition Plan:**
  - 500 images allocated to `eval_modern` split (`data/e5_external/modern_ai/dalle3_eval/`).

---

## 2. Real Camera Sources

### VISION Smartphone Dataset
- **Institution:** University of Florence — Media Integration and Communication Center (MICC) & LESC Laboratory.
- **Citation:** Dasara Shullani, Marco Fontani, Massimo Iuliani, Omar Al Shaya, Alessandro Piva, *"VISION: a video and image dataset for source identification"*, EURASIP Journal on Information Security, 2017.
- **URL:** [https://lesc.dinfo.unifi.it/VISION/](https://lesc.dinfo.unifi.it/VISION/)
- **License:** Open Academic Research License (scientific community research dataset).
- **Characteristics:** Captured across 35 portable devices (11 brands). Each device includes:
  - `nat/`: Native pristine camera JPEG captures.
  - `natWA/`: Social-media recompressed versions downloaded from WhatsApp.
- **Device Selection Plan:**
  - **Apple Devices (1,250 total):**
    - `D02_Apple_iPhone4s`
    - `D05_Apple_iPhone5c`
    - `D06_Apple_iPhone6`
    - `D14_Apple_iPhone5c`
    - `D19_Apple_iPhone6Plus`
    - Partition: 625 native (`nat/`), 625 WhatsApp (`natWA/`).
    - Allocation: 1,000 to `train`, 250 to `val`.
  - **Samsung / Android Devices (1,250 total):**
    - `D01_Samsung_GalaxyS3Mini`
    - `D11_Samsung_GalaxyS3`
    - `D24_Xiaomi_RedmiNote3`
    - `D25_OnePlus_A3000`
    - `D27_Samsung_GalaxyS5`
    - `D31_Samsung_GalaxyS4Mini`
    - Partition: 625 native (`nat/`), 625 WhatsApp (`natWA/`).
    - Allocation: 1,000 to `train`, 250 to `val`.
