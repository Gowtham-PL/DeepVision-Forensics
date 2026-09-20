# WhatsApp Robustness Evaluation Report: DeepVision E6-C

**Evaluation Type**: Controlled WhatsApp Robustness Evaluation (No Training, Frozen Weights)

- **Evaluated Checkpoint**: `experiments\e6_multiscale_training\e6c_finetune_benchmark\e6c_checkpoint_epoch2.pt`
- **Dataset**: `data\whatsapp_robustness_test`
- **Classification Threshold**: `0.5`
- **Pipeline Architecture**: 5-View Multi-Scale Spatial + Frequency Fusion with Learned Attention
- **Execution Hardware**: `cuda`

## 1. Executive Summary & Core Metrics

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Sample Size (N)** | **67** | 34 Real + 33 AI WhatsApp images |
| **Overall Accuracy** | **61.19%** | (41 / 67 correct) |
| **Precision** | **81.82%** | Precision on AI detection |
| **Recall (Sensitivity)** | **27.27%** | Recall on AI detection (9 / 33) |
| **F1-Score** | **0.4091** | Harmonic mean of Precision & Recall |
| **ROC-AUC** | **0.6560** | Area under ROC Curve (Ranking discriminability) |
| **PR-AUC** | **0.7139** | Area under Precision-Recall Curve |
| **False Positive Rate (FPR)** | **5.88%** | Real images misclassified as AI (2 / 34) |
| **False Negative Rate (FNR)** | **72.73%** | AI images misclassified as Real (24 / 33) |

## 2. Confusion Matrix

| | Predicted Real | Predicted AI | Total |
| :--- | :--- | :--- | :--- |
| **Actual Real** | **TN = 32** (94.1%) | **FP = 2** (5.9%) | 34 |
| **Actual AI** | **FN = 24** (72.7%) | **TP = 9** (27.3%) | 33 |
| **Total** | 56 | 11 | 67 |

## 3. Probability Distribution Analysis

| Class | Mean AI Probability | Median AI Probability | Min AI Probability | Max AI Probability |
| :--- | :--- | :--- | :--- | :--- |
| **REAL Images** (N=34) | 0.1080 (10.80%) | 0.0446 (4.46%) | 0.0000 | 0.5367 |
| **AI Images** (N=33) | 0.2781 (27.81%) | 0.1694 (16.94%) | 0.0001 | 0.9235 |

## 4. Strongest Local Evidence Statistics

The strongest local evidence reflects the highest AI probability among the 4 deterministic 60% corner crops (Top-Left, Top-Right, Bottom-Left, Bottom-Right):

| Class | Mean Local Evidence | Median Local Evidence | Min Local Evidence | Max Local Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **REAL Images** | 0.3076 | 0.1596 | 0.0000 | 0.9341 |
| **AI Images** | 0.5083 | 0.6406 | 0.0004 | 0.9926 |

## 5. Subgroup Breakdown

### A. REAL WhatsApp Images
- Total Evaluated: **34**
- Correctly Identified as Real: **32** (94.12%)
- False Positives: **2** (5.88%)
- Mean AI Probability: **0.1080**

### B. AI WhatsApp Images
- Total Evaluated: **33**
- Correctly Identified as AI: **9** (27.27%)
- False Negatives: **24** (72.73%)
- Mean AI Probability: **0.2781**

## 6. Detailed Error Inspection (False Positives & False Negatives)

### False Positives (Real misclassified as AI: 2)

| Filename | Overall Prob | Strongest Local Prob | View Probs [Global, TL, TR, BL, BR] | Attn Weights [Global, TL, TR, BL, BR] |
| :--- | :--- | :--- | :--- | :--- |
| `WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg` | **0.5367** | 0.8095 | [0.010788, 0.523389, 0.809515, 0.446523, 0.300154] | [0.2186, 0.2025, 0.1807, 0.2052, 0.1929] |
| `WhatsApp Image 2026-09-17 at 3.44.11 PM (3).jpeg` | **0.5136** | 0.8067 | [0.581229, 0.806707, 0.289071, 0.176652, 0.392831] | [0.2201, 0.1976, 0.1954, 0.1818, 0.2052] |

### False Negatives (AI misclassified as Real: 24)

| Filename | Overall Prob | Strongest Local Prob | View Probs [Global, TL, TR, BL, BR] | Attn Weights [Global, TL, TR, BL, BR] |
| :--- | :--- | :--- | :--- | :--- |
| `WhatsApp Image 2026-09-17 at 3.50.51 PM.jpeg` | **0.2390** | 0.7448 | [0.270872, 0.179102, 0.251088, 0.225147, 0.744794] | [0.2397, 0.1937, 0.1756, 0.2165, 0.1746] |
| `WhatsApp Image 2026-09-17 at 3.56.02 PM (1).jpeg` | **0.0686** | 0.6646 | [0.004296, 0.234961, 0.003446, 0.664588, 0.020311] | [0.2033, 0.2393, 0.222, 0.1786, 0.1569] |
| `WhatsApp Image 2026-09-17 at 3.56.02 PM (2).jpeg` | **0.1800** | 0.6123 | [0.65527, 0.020474, 0.042867, 0.144914, 0.612314] | [0.2117, 0.1987, 0.2032, 0.1811, 0.2053] |
| `WhatsApp Image 2026-09-17 at 3.56.02 PM.jpeg` | **0.0796** | 0.0967 | [0.701114, 0.05567, 0.032058, 0.096705, 0.012389] | [0.2194, 0.213, 0.2359, 0.1894, 0.1422] |
| `WhatsApp Image 2026-09-17 at 4.13.17 PM (1).jpeg` | **0.0029** | 0.0096 | [0.012047, 0.000128, 0.000988, 0.009556, 0.004297] | [0.2233, 0.1904, 0.201, 0.198, 0.1873] |
| `WhatsApp Image 2026-09-17 at 4.13.17 PM (4).jpeg` | **0.1699** | 0.3291 | [0.442627, 0.019971, 0.329149, 0.003042, 0.05108] | [0.2201, 0.2254, 0.2024, 0.1966, 0.1555] |
| `WhatsApp Image 2026-09-17 at 4.13.17 PM.jpeg` | **0.0001** | 0.0004 | [0.000577, 1e-05, 0.000359, 0.00025, 0.000184] | [0.2167, 0.2017, 0.2038, 0.1511, 0.2268] |
| `WhatsApp Image 2026-09-17 at 4.13.18 PM (2).jpeg` | **0.1694** | 0.8371 | [0.052485, 0.017052, 0.015624, 0.802806, 0.837102] | [0.2209, 0.2052, 0.1997, 0.1872, 0.187] |
| `WhatsApp Image 2026-09-17 at 4.13.18 PM (3).jpeg` | **0.2781** | 0.5879 | [0.727577, 0.133524, 0.54541, 0.077653, 0.587888] | [0.2091, 0.1954, 0.1838, 0.2221, 0.1896] |
| `WhatsApp Image 2026-09-17 at 4.13.18 PM (4).jpeg` | **0.0566** | 0.1307 | [0.109949, 0.130712, 0.04038, 0.068682, 0.006696] | [0.2234, 0.2084, 0.2135, 0.1662, 0.1885] |
| `WhatsApp Image 2026-09-17 at 4.13.18 PM.jpeg` | **0.4772** | 0.7315 | [0.908649, 0.731482, 0.197674, 0.155419, 0.001841] | [0.2509, 0.2002, 0.2185, 0.1429, 0.1875] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM (2).jpeg` | **0.0047** | 0.6406 | [0.01014, 5.5e-05, 0.003219, 0.640588, 0.006714] | [0.249, 0.265, 0.1932, 0.1718, 0.121] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM (3).jpeg` | **0.0016** | 0.0118 | [0.000903, 0.011406, 0.011845, 4.9e-05, 0.001694] | [0.2193, 0.2307, 0.1947, 0.1783, 0.1771] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM (4).jpeg` | **0.0021** | 0.0032 | [0.018599, 0.003236, 0.000135, 0.002204, 0.001701] | [0.202, 0.2169, 0.2002, 0.1896, 0.1913] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM (5).jpeg` | **0.0837** | 0.8199 | [0.007369, 0.219249, 0.000443, 0.819882, 0.567911] | [0.2101, 0.1959, 0.2247, 0.1715, 0.1978] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM (6).jpeg` | **0.3284** | 0.8238 | [0.479747, 0.231011, 0.823782, 0.426188, 0.012754] | [0.2187, 0.2024, 0.2289, 0.1717, 0.1783] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM (7).jpeg` | **0.0060** | 0.0595 | [0.00061, 0.022852, 0.004486, 0.05954, 0.003619] | [0.2038, 0.1969, 0.213, 0.1788, 0.2075] |
| `WhatsApp Image 2026-09-17 at 4.28.11 PM.jpeg` | **0.0149** | 0.0195 | [0.368212, 0.009519, 0.002359, 0.01949, 0.000989] | [0.2092, 0.2147, 0.2118, 0.1822, 0.1822] |
| `WhatsApp Image 2026-09-17 at 4.31.09 PM.jpeg` | **0.1063** | 0.2092 | [0.331918, 0.209222, 0.025422, 0.148566, 0.016575] | [0.2135, 0.2214, 0.2152, 0.1854, 0.1645] |
| `WhatsApp Image 2026-09-17 at 4.37.12 PM (1).jpeg` | **0.0022** | 0.1756 | [0.042077, 1e-05, 0.000686, 0.000124, 0.175625] | [0.244, 0.1603, 0.1989, 0.1969, 0.2] |
| `WhatsApp Image 2026-09-17 at 4.37.12 PM (2).jpeg` | **0.3809** | 0.7785 | [0.782312, 0.016123, 0.057474, 0.778521, 0.731609] | [0.2021, 0.2214, 0.2115, 0.1883, 0.1767] |
| `WhatsApp Image 2026-09-17 at 4.37.12 PM (3).jpeg` | **0.0019** | 0.0075 | [0.000577, 0.007549, 0.002076, 0.003242, 0.001821] | [0.2256, 0.2007, 0.2366, 0.1607, 0.1763] |
| `WhatsApp Image 2026-09-17 at 4.37.12 PM.jpeg` | **0.0007** | 0.0495 | [0.001153, 0.001095, 1.4e-05, 0.049546, 0.003177] | [0.1815, 0.218, 0.2303, 0.2111, 0.1591] |
| `WhatsApp Image 2026-09-17 at 4.39.57 PM.jpeg` | **0.0400** | 0.2531 | [0.025505, 0.005742, 0.253136, 0.159466, 0.006778] | [0.2274, 0.192, 0.2153, 0.2022, 0.1631] |

## 7. Full Per-Image Predictions Table

Complete per-image tabular evaluation results:

| # | Filename | Ground Truth | Prediction | AI Prob | Strongest Local | Correct? | Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `WhatsApp Image 2026-09-17 at 3.38.18 PM (1).jpeg` | REAL | AI | 0.5367 | 0.8095 | FAIL | FP |
| 2 | `WhatsApp Image 2026-09-17 at 3.38.18 PM (2).jpeg` | REAL | REAL | 0.0248 | 0.0659 | PASS | - |
| 3 | `WhatsApp Image 2026-09-17 at 3.38.18 PM.jpeg` | REAL | REAL | 0.0000 | 0.0006 | PASS | - |
| 4 | `WhatsApp Image 2026-09-17 at 3.38.19 PM (1).jpeg` | REAL | REAL | 0.0463 | 0.2501 | PASS | - |
| 5 | `WhatsApp Image 2026-09-17 at 3.38.19 PM (2).jpeg` | REAL | REAL | 0.0095 | 0.0530 | PASS | - |
| 6 | `WhatsApp Image 2026-09-17 at 3.38.19 PM.jpeg` | REAL | REAL | 0.0002 | 0.0040 | PASS | - |
| 7 | `WhatsApp Image 2026-09-17 at 3.38.21 PM (1).jpeg` | REAL | REAL | 0.0060 | 0.0893 | PASS | - |
| 8 | `WhatsApp Image 2026-09-17 at 3.38.21 PM (2).jpeg` | REAL | REAL | 0.0542 | 0.0861 | PASS | - |
| 9 | `WhatsApp Image 2026-09-17 at 3.38.21 PM.jpeg` | REAL | REAL | 0.0054 | 0.0453 | PASS | - |
| 10 | `WhatsApp Image 2026-09-17 at 3.38.22 PM (1).jpeg` | REAL | REAL | 0.3354 | 0.8748 | PASS | - |
| 11 | `WhatsApp Image 2026-09-17 at 3.38.22 PM (2).jpeg` | REAL | REAL | 0.0001 | 0.0042 | PASS | - |
| 12 | `WhatsApp Image 2026-09-17 at 3.38.22 PM (3).jpeg` | REAL | REAL | 0.1817 | 0.2362 | PASS | - |
| 13 | `WhatsApp Image 2026-09-17 at 3.38.22 PM.jpeg` | REAL | REAL | 0.0706 | 0.7106 | PASS | - |
| 14 | `WhatsApp Image 2026-09-17 at 3.40.06 PM (1).jpeg` | REAL | REAL | 0.0328 | 0.4490 | PASS | - |
| 15 | `WhatsApp Image 2026-09-17 at 3.40.06 PM.jpeg` | REAL | REAL | 0.0130 | 0.0417 | PASS | - |
| 16 | `WhatsApp Image 2026-09-17 at 3.44.10 PM (1).jpeg` | REAL | REAL | 0.0141 | 0.0844 | PASS | - |
| 17 | `WhatsApp Image 2026-09-17 at 3.44.10 PM (2).jpeg` | REAL | REAL | 0.0038 | 0.0048 | PASS | - |
| 18 | `WhatsApp Image 2026-09-17 at 3.44.10 PM (3).jpeg` | REAL | REAL | 0.0471 | 0.2177 | PASS | - |
| 19 | `WhatsApp Image 2026-09-17 at 3.44.10 PM (4).jpeg` | REAL | REAL | 0.0453 | 0.3181 | PASS | - |
| 20 | `WhatsApp Image 2026-09-17 at 3.44.10 PM (7).jpeg` | REAL | REAL | 0.0005 | 0.0373 | PASS | - |
| 21 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (1).jpeg` | REAL | REAL | 0.0439 | 0.6690 | PASS | - |
| 22 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (10).jpeg` | REAL | REAL | 0.3169 | 0.5157 | PASS | - |
| 23 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (11).jpeg` | REAL | REAL | 0.0626 | 0.4151 | PASS | - |
| 24 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (12).jpeg` | REAL | REAL | 0.4119 | 0.9341 | PASS | - |
| 25 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (13).jpeg` | REAL | REAL | 0.3496 | 0.7368 | PASS | - |
| 26 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (14).jpeg` | REAL | REAL | 0.1493 | 0.2493 | PASS | - |
| 27 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (15).jpeg` | REAL | REAL | 0.1645 | 0.7105 | PASS | - |
| 28 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (2).jpeg` | REAL | REAL | 0.0001 | 0.0005 | PASS | - |
| 29 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (3).jpeg` | REAL | AI | 0.5136 | 0.8067 | FAIL | FP |
| 30 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (4).jpeg` | REAL | REAL | 0.0000 | 0.0000 | PASS | - |
| 31 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (5).jpeg` | REAL | REAL | 0.1712 | 0.8340 | PASS | - |
| 32 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (7).jpeg` | REAL | REAL | 0.0073 | 0.0806 | PASS | - |
| 33 | `WhatsApp Image 2026-09-17 at 3.44.11 PM (8).jpeg` | REAL | REAL | 0.0468 | 0.1015 | PASS | - |
| 34 | `WhatsApp Image 2026-09-17 at 3.44.11 PM.jpeg` | REAL | REAL | 0.0073 | 0.0221 | PASS | - |
| 35 | `WhatsApp Image 2026-09-17 at 3.50.51 PM.jpeg` | AI | REAL | 0.2390 | 0.7448 | FAIL | FN |
| 36 | `WhatsApp Image 2026-09-17 at 3.56.02 PM (1).jpeg` | AI | REAL | 0.0686 | 0.6646 | FAIL | FN |
| 37 | `WhatsApp Image 2026-09-17 at 3.56.02 PM (2).jpeg` | AI | REAL | 0.1800 | 0.6123 | FAIL | FN |
| 38 | `WhatsApp Image 2026-09-17 at 3.56.02 PM (3).jpeg` | AI | AI | 0.7363 | 0.8005 | PASS | - |
| 39 | `WhatsApp Image 2026-09-17 at 3.56.02 PM.jpeg` | AI | REAL | 0.0796 | 0.0967 | FAIL | FN |
| 40 | `WhatsApp Image 2026-09-17 at 4.13.17 PM (1).jpeg` | AI | REAL | 0.0029 | 0.0096 | FAIL | FN |
| 41 | `WhatsApp Image 2026-09-17 at 4.13.17 PM (2).jpeg` | AI | AI | 0.8050 | 0.8953 | PASS | - |
| 42 | `WhatsApp Image 2026-09-17 at 4.13.17 PM (3).jpeg` | AI | AI | 0.7318 | 0.8800 | PASS | - |
| 43 | `WhatsApp Image 2026-09-17 at 4.13.17 PM (4).jpeg` | AI | REAL | 0.1699 | 0.3291 | FAIL | FN |
| 44 | `WhatsApp Image 2026-09-17 at 4.13.17 PM (5).jpeg` | AI | AI | 0.5517 | 0.9535 | PASS | - |
| 45 | `WhatsApp Image 2026-09-17 at 4.13.17 PM.jpeg` | AI | REAL | 0.0001 | 0.0004 | FAIL | FN |
| 46 | `WhatsApp Image 2026-09-17 at 4.13.18 PM (1).jpeg` | AI | AI | 0.6780 | 0.9602 | PASS | - |
| 47 | `WhatsApp Image 2026-09-17 at 4.13.18 PM (2).jpeg` | AI | REAL | 0.1694 | 0.8371 | FAIL | FN |
| 48 | `WhatsApp Image 2026-09-17 at 4.13.18 PM (3).jpeg` | AI | REAL | 0.2781 | 0.5879 | FAIL | FN |
| 49 | `WhatsApp Image 2026-09-17 at 4.13.18 PM (4).jpeg` | AI | REAL | 0.0566 | 0.1307 | FAIL | FN |
| 50 | `WhatsApp Image 2026-09-17 at 4.13.18 PM (5).jpeg` | AI | AI | 0.9235 | 0.9926 | PASS | - |
| 51 | `WhatsApp Image 2026-09-17 at 4.13.18 PM (6).jpeg` | AI | AI | 0.6134 | 0.9321 | PASS | - |
| 52 | `WhatsApp Image 2026-09-17 at 4.13.18 PM.jpeg` | AI | REAL | 0.4772 | 0.7315 | FAIL | FN |
| 53 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (1).jpeg` | AI | AI | 0.5728 | 0.8046 | PASS | - |
| 54 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (2).jpeg` | AI | REAL | 0.0047 | 0.6406 | FAIL | FN |
| 55 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (3).jpeg` | AI | REAL | 0.0016 | 0.0118 | FAIL | FN |
| 56 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (4).jpeg` | AI | REAL | 0.0021 | 0.0032 | FAIL | FN |
| 57 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (5).jpeg` | AI | REAL | 0.0837 | 0.8199 | FAIL | FN |
| 58 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (6).jpeg` | AI | REAL | 0.3284 | 0.8238 | FAIL | FN |
| 59 | `WhatsApp Image 2026-09-17 at 4.28.11 PM (7).jpeg` | AI | REAL | 0.0060 | 0.0595 | FAIL | FN |
| 60 | `WhatsApp Image 2026-09-17 at 4.28.11 PM.jpeg` | AI | REAL | 0.0149 | 0.0195 | FAIL | FN |
| 61 | `WhatsApp Image 2026-09-17 at 4.31.09 PM.jpeg` | AI | REAL | 0.1063 | 0.2092 | FAIL | FN |
| 62 | `WhatsApp Image 2026-09-17 at 4.37.12 PM (1).jpeg` | AI | REAL | 0.0022 | 0.1756 | FAIL | FN |
| 63 | `WhatsApp Image 2026-09-17 at 4.37.12 PM (2).jpeg` | AI | REAL | 0.3809 | 0.7785 | FAIL | FN |
| 64 | `WhatsApp Image 2026-09-17 at 4.37.12 PM (3).jpeg` | AI | REAL | 0.0019 | 0.0075 | FAIL | FN |
| 65 | `WhatsApp Image 2026-09-17 at 4.37.12 PM.jpeg` | AI | REAL | 0.0007 | 0.0495 | FAIL | FN |
| 66 | `WhatsApp Image 2026-09-17 at 4.39.57 PM (1).jpeg` | AI | AI | 0.8717 | 0.9586 | PASS | - |
| 67 | `WhatsApp Image 2026-09-17 at 4.39.57 PM.jpeg` | AI | REAL | 0.0400 | 0.2531 | FAIL | FN |
