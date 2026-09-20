# Experiment 23 (E23): Confidence-Aware E20/E21 Model Routing Report
**DeepVision-Forensics Research Progression**  
**Date:** September 19, 2026  
**Artifact:** `experiments/e23_model_routing/e23_routing_report.md`  
**Base Models:**  
- **E20:** `experiments/e20_training/checkpoints/e20_best_model.pt`  
- **E21:** `experiments/e21_targeted_data/checkpoints/e21_best_model.pt`  
- **E6-C Baseline:** `experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt`  
- **E22 Fixed Blend:** $0.10 \times \text{E20} + 0.90 \times \text{E21}$  
**Winning E21 Dev Rule:** `C_E22_blend` (Rank 1, F1: 0.8941, Real FPR: 15.76%)  
**Top Pure Routing Rule:** `D9_disagree_e21` (Rank 3, F1: 0.8884, Real FPR: 17.73%)  
**Protocol:** Strictly Frozen Checkpoints, Zero Retraining, Fixed Threshold $\tau = 0.50$ across all benchmarks.

---

## 1. Executive Summary & Core Objective

Experiment 23 (E23) was designed to determine whether a lightweight, confidence-aware routing strategy—either a deterministic confidence rule, a logistic regression router, or a small decision tree—could dynamically exploit the complementary forensic strengths of the frozen **E20** (ultra-conservative, 1% real FPR on smartphones) and **E21** (high-recall, 95% detection on legacy diffusion models) detectors without inheriting E21's severe false-positive penalty on uncompressed real imagery.

### Scientific Protocols Enforced:
1. **Zero Model Retraining / Fine-Tuning:** All detector weights (`E20`, `E21`, `E6-C`) remained frozen and strictly read-only.
2. **Strict Data Policy:** All router candidate selection and training occurred **strictly on the independent E21 Dev split ($N=812$)**. No external test labels were seen, touched, or optimized against during router design.
3. **No Pixel / Deep Feature Extraction:** The router operated solely on 9 interpretable prediction-derived features ($p_{e20}, p_{e21}, |p_{e20}-p_{e21}|$, mean, min, max, confidence distances $|p-0.5|$, and binary agreement).
4. **Fixed Operational Threshold:** A fixed threshold $\tau = 0.50$ was maintained for all evaluations.
5. **No Production Modifications & Zero Git Commits:** Unmodified production backend and zero Git commits/pushes.

### Primary Experimental Conclusions:
1. **Fixed Continuous Blend Outperforms All Discrete Routers:** On the independent E21 Dev split, the fixed linear blend `0.1*E20 + 0.9*E21` (`C_E22_blend`) ranked **#1** across all 15 candidates ($F1 = 0.8941$, $Real FPR = 15.76\%$, $Accuracy = 88.92\%$). It outperformed all deterministic routing rules, logistic regression, and decision trees.
2. **Failure of Simple Confidence-Distance Routing (`D1_conf_max`):** The intuitive heuristic "route to whichever model is further from 0.50" severely degraded performance ($F1 = 0.7029$, AI Recall dropped from 94.09% to 60.59%, ranking #9). The mechanism is profound: E20 is **extremely confident** that legacy AI images are real ($p_{e20} \approx 0.01$, distance $|0.01 - 0.5| = 0.49$). When E21 detects the AI image with moderate-to-high confidence ($p_{e21} \approx 0.85$, distance $|0.85 - 0.5| = 0.35$), the confidence router selects E20, directly routing into E20's blind spot.
3. **Discrete Routing Suffers From an All-or-Nothing Dilemma:** On external benchmarks, discrete routing rules that pick either E20 or E21 per image cannot escape the underlying calibration disparity:
   - Routing to E21 on disagreement preserves legacy AI detection (95.0% recall on E14 Clean), but immediately inherits E21's **67.00% Real FPR** on uncompressed RAW imagery.
   - Routing to E20 on disagreement suppresses real false positives (4.00% FPR on E14 Clean), but immediately collapses AI recall down to **34.00%**.
   - Continuous blending (`C_E22_blend`) smoothly suppresses marginal false positives, reducing E1 Clean Real FPR from 67.00% to 59.00% and E14 Degraded Real FPR from 63.00% to 57.67% while keeping AI Recall above 88–93%.
4. **Conclusion on Model Routing:** Confidence-aware routing based purely on output probabilities without domain/metadata context cannot bridge the gap between two detectors operating in fundamentally different calibration regimes. The evidence indicates that further heuristic or prediction-level routing will not achieve the 90%+ real-world target without domain-aware metadata or joint calibration.

---

## 2. Phase A: Router Selection Strictly on E21 Dev Split ($N=812$)

All 15 candidate strategies were evaluated on the 812 validation instances of E21 Dev (406 Real, 406 AI).

### Candidate Protocol:
- **A_E20_only:** Pure E20 baseline.
- **B_E21_only:** Pure E21 baseline.
- **C_E22_blend:** Linear blend $0.10 \times p_{20} + 0.90 \times p_{21}$.
- **D1_conf_max:** If $|p_{20} - 0.5| \ge |p_{21} - 0.5| \implies$ E20, else E21.
- **D2_conf_min:** If $|p_{20} - 0.5| < |p_{21} - 0.5| \implies$ E20, else E21.
- **D3_e21_override_p80:** E20 default; override to E21 if $p_{21} \ge 0.80$.
- **D4_e21_override_p70:** E20 default; override to E21 if $p_{21} \ge 0.70$.
- **D5_e20_override_p10:** E21 default; override to E20 if $p_{20} \le 0.10$.
- **D6_e20_override_p20:** E21 default; override to E20 if $p_{20} \le 0.20$.
- **D7_disagree_max_conf:** If agree use E21; if disagree use max confidence.
- **D8_disagree_e20:** If agree use E21; if disagree use conservative E20.
- **D9_disagree_e21:** If agree use E20; if disagree use sensitive E21.
- **E_logistic_regression:** Scikit-learn LogisticRegression fit on 9 features predicting optimal model assignment.
- **F1_decision_tree_d2:** DecisionTreeClassifier (`max_depth=2`) fit on 9 features.
- **F2_decision_tree_d3:** DecisionTreeClassifier (`max_depth=3`) fit on 9 features.

### Selection Hierarchy:
1. **Primary:** Highest F1-Score
2. **Tie-Break 1:** Lowest Real False Positive Rate (FPR)
3. **Tie-Break 2:** Highest ROC-AUC
4. **Tie-Break 3:** Highest AI Recall

### E21 Dev Candidate Performance Table:

| Rank | Candidate ID | Description | Accuracy | F1-Score | ROC-AUC | AI Recall | Real FPR | % Routed E20 | % Routed E21 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **C_E22_blend** | **Fixed blend (0.1\*E20 + 0.9\*E21)** | **88.92%** | **0.8941** | **0.9498** | **93.60%** | **15.76%** | **0.0%** | **0.0%** |
| 2 | B_E21_only | Pure E21 baseline | 88.18% | 0.8884 | 0.9530 | 94.09% | 17.73% | 0.0% | 100.0% |
| 3 | D9_disagree_e21 | Disagree $\rightarrow$ use E21 | 88.18% | 0.8884 | 0.9261 | 94.09% | 17.73% | 54.3% | 45.7% |
| 4 | F1_decision_tree_d2 | Decision Tree (depth $\le 2$) | 88.05% | 0.8860 | 0.9250 | 92.86% | 16.75% | 20.9% | 79.1% |
| 5 | E_logistic_regression| Logistic Regression on 9 feats | 88.18% | 0.8835 | 0.9209 | 89.66% | 13.30% | 40.9% | 59.1% |
| 6 | D4_e21_override_p70 | E20 default, E21 $\ge 0.70$ override | 87.07% | 0.8730 | 0.9107 | 88.92% | 14.78% | 51.0% | 49.0% |
| 7 | F2_decision_tree_d3 | Decision Tree (depth $\le 3$) | 87.56% | 0.8723 | 0.9244 | 84.98% | 9.85% | 50.1% | 49.9% |
| 8 | D3_e21_override_p80 | E20 default, E21 $\ge 0.80$ override | 84.61% | 0.8412 | 0.8767 | 81.53% | 12.32% | 56.5% | 43.5% |
| 9 | D1_conf_max | Max distance from 0.50 | 74.38% | 0.7029 | 0.7938 | 60.59% | 11.82% | 51.4% | 48.6% |
| 10 | D7_disagree_max_conf| Disagree $\rightarrow$ max confidence | 74.38% | 0.7029 | 0.7595 | 60.59% | 11.82% | 22.4% | 77.6% |
| 11 | D5_e20_override_p10 | E21 default, E20 $\le 0.10$ override | 70.94% | 0.6499 | 0.7546 | 53.94% | 12.07% | 56.0% | 44.0% |
| 12 | D2_conf_min | Min distance from 0.50 | 70.32% | 0.6471 | 0.7952 | 54.43% | 13.79% | 48.6% | 51.4% |
| 13 | D6_e20_override_p20 | E21 default, E20 $\le 0.20$ override | 63.42% | 0.5008 | 0.7059 | 36.70% | 9.85% | 70.0% | 30.0% |
| 14 | A_E20_only | Pure E20 baseline | 56.53% | 0.3250 | 0.6559 | 20.94% | 7.88% | 100.0% | 0.0% |
| 15 | D8_disagree_e20 | Disagree $\rightarrow$ use E20 | 56.53% | 0.3250 | 0.6458 | 20.94% | 7.88% | 45.7% | 54.3% |

### Selection Outcome:
- **Winning Router Selected & Frozen:** `C_E22_blend` ($p = 0.10 p_{20} + 0.90 p_{21}$) is the clear winner on the Dev split, achieving the highest F1 score (**0.8941**) and reducing Real FPR from 17.73% to 15.76%.
- **Top Discrete Pure Router:** `D9_disagree_e21` achieved Rank 3 ($F1 = 0.8884$), followed by `F1_decision_tree_d2` at Rank 4 ($F1 = 0.8860$). Both discrete models are evaluated side-by-side with the winning blend across all external benchmarks.

---

## 3. Phase B: Sealed External Benchmark Evaluation

All sealed test sets were evaluated strictly post-freeze under fixed threshold $\tau = 0.50$. Full results are saved in [`experiments/e23_model_routing/e23_external_results.csv`](file:///d:/DeepVision-Forensics/experiments/e23_model_routing/e23_external_results.csv).

### Benchmark Comparison Matrix:

```
====================================================================================================================================================
Benchmark Dataset          N    Model                              Accuracy   ROC-AUC    PR-AUC    Precision  AI Recall  Real FPR   F1-Score   TP / FP
====================================================================================================================================================
E14 Clean External        200   E6-C Baseline                       75.50%     0.8427    0.8493     76.29%     74.00%     23.00%     0.7513    74 / 23
                                E20 Baseline                        65.00%     0.7765    0.7746     89.47%     34.00%      4.00%     0.4928    34 /  4
                                E21 Baseline                        64.00%     0.8369    0.8448     58.64%     95.00%     67.00%     0.7252    95 / 67
                                E22 Ensemble (0.1/0.9)              67.00%     0.8448    0.8426     61.18%     93.00%     59.00%     0.7381    93 / 59
                                E23 Router (Winning C_E22_blend)    67.00%     0.8448    0.8426     61.18%     93.00%     59.00%     0.7381    93 / 59
                                E23 Pure Router (D9_disagree_e21)   64.00%     0.7686    0.7825     58.64%     95.00%     67.00%     0.7252    95 / 67
----------------------------------------------------------------------------------------------------------------------------------------------------
E14 Degraded External     600   E6-C Baseline                       65.83%     0.7640    0.7723     78.44%     43.67%     12.00%     0.5610   131 / 36
                                E20 Baseline                        64.17%     0.7980    0.8013     89.72%     32.00%      3.67%     0.4717    96 / 11
                                E21 Baseline                        63.17%     0.7699    0.7749     58.64%     89.33%     63.00%     0.7081   268 / 189
                                E22 Ensemble (0.1/0.9)              65.17%     0.7817    0.7918     60.41%     88.00%     57.67%     0.7164   264 / 173
                                E23 Router (Winning C_E22_blend)    65.17%     0.7817    0.7917     60.41%     88.00%     57.67%     0.7164   264 / 173
                                E23 Pure Router (D9_disagree_e21)   63.17%     0.7256    0.7303     58.64%     89.33%     63.00%     0.7081   268 / 189
----------------------------------------------------------------------------------------------------------------------------------------------------
E19 Smartphone Benchmark  200   E6-C Baseline                       69.50%     0.7853    0.7705     77.46%     55.00%     16.00%     0.6433    55 / 16
                                E20 Baseline                        86.50%     0.9770    0.9708     98.67%     74.00%      1.00%     0.8457    74 /  1
                                E21 Baseline                        80.50%     0.8835    0.8852     75.63%     90.00%     29.00%     0.8219    90 / 29
                                E22 Ensemble (0.1/0.9)              80.00%     0.9153    0.9174     76.79%     86.00%     26.00%     0.8113    86 / 26
                                E23 Router (Winning C_E22_blend)    80.00%     0.9153    0.9174     76.79%     86.00%     26.00%     0.8113    86 / 26
                                E23 Pure Router (D9_disagree_e21)   80.50%     0.8748    0.8632     75.63%     90.00%     29.00%     0.8219    90 / 29
----------------------------------------------------------------------------------------------------------------------------------------------------
E19 WhatsApp Subset        60   E6-C Baseline                       36.67%     0.3044    0.3746     27.78%     16.67%     43.33%     0.2083     5 / 13
                                E20 Baseline                        73.33%     0.8911    0.8687     93.75%     50.00%      3.33%     0.6522    15 /  1
                                E21 Baseline                        61.67%     0.6189    0.6168     57.45%     90.00%     66.67%     0.7013    27 / 20
                                E22 Ensemble (0.1/0.9)              56.67%     0.6656    0.7025     54.76%     76.67%     63.33%     0.6389    23 / 19
                                E23 Router (Winning C_E22_blend)    56.67%     0.6656    0.7025     54.76%     76.67%     63.33%     0.6389    23 / 19
                                E23 Pure Router (D9_disagree_e21)   61.67%     0.5311    0.4722     57.45%     90.00%     66.67%     0.7013    27 / 20
----------------------------------------------------------------------------------------------------------------------------------------------------
WhatsApp N=67 Benchmark    67   E6-C Baseline                       61.19%     0.6551    0.6828     81.82%     27.27%      5.88%     0.4091     9 /  2
                                E20 Baseline                        52.24%     0.5321    0.5367     52.00%     39.39%     35.29%     0.4483    13 / 12
                                E21 Baseline                        55.22%     0.5642    0.5512     53.66%     66.67%     55.88%     0.5946    22 / 19
                                E22 Ensemble (0.1/0.9)              53.73%     0.5633    0.5451     52.50%     63.64%     55.88%     0.5753    21 / 19
                                E23 Router (Winning C_E22_blend)    53.73%     0.5633    0.5451     52.50%     63.64%     55.88%     0.5753    21 / 19
                                E23 Pure Router (D9_disagree_e21)   55.22%     0.5517    0.5404     53.66%     66.67%     55.88%     0.5946    22 / 19
====================================================================================================================================================
```

---

## 4. Routing Frequency & Subgroup Diagnostics

### 4.1 Router Selection Frequencies Across Benchmarks

| Benchmark Dataset | Total $N$ | Winning Blend ($0.1/0.9$) | Pure Router % Routed E20 | Pure Router % Routed E21 |
| :--- | :---: | :---: | :---: | :---: |
| **E21 Dev Split (Validation)** | 812 | Continuous Blend | 54.3% | 45.7% |
| **E14 Clean External** | 200 | Continuous Blend | 37.0% | 63.0% |
| **E14 Degraded External** | 600 | Continuous Blend | 40.7% | 59.3% |
| **E19 Smartphone Benchmark** | 200 | Continuous Blend | 73.0% | 27.0% |
| **E19 WhatsApp Subset** | 60 | Continuous Blend | 41.7% | 58.3% |
| **WhatsApp N=67 Benchmark** | 67 | Continuous Blend | 52.2% | 47.8% |

*Key Insight:* On E19 Smartphone images, the pure router correctly favored E20 (73.0% routed to E20), whereas on E14 diffusion images, it favored E21 (63.0% routed to E21). However, because routing was discrete, any mistake in routing an uncompressed RAISE-1k photo to E21 immediately triggered a false positive.

### 4.2 AI Generator Sensitivity (E14 Clean External Test, $N=100$)

| Generator | $N$ | E6-C Recall | E20 Recall | E21 Recall | **E23 Blend** | **E23 Pure** | Pure % Routed E21 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DALL-E 2** | 14 | 42.86% | 14.29% | 92.86% | **92.86%** | **92.86%** | 78.6% |
| **DALL-E 3** | 6 | 100.00% | 100.00% | 100.00% | **100.00%** | **100.00%** | 0.0% (both agree) |
| **Adobe Firefly 1** | 14 | 42.86% | 21.43% | 85.71% | **85.71%** | **85.71%** | 64.3% |
| **OpenAI GLIDE** | 13 | 92.31% | 38.46% | 100.00% | **100.00%** | **100.00%** | 61.5% |
| **Midjourney v5** | 12 | 91.67% | 66.67% | 100.00% | **91.67%** | **100.00%** | 33.3% |
| **Stable Diffusion 1.3** | 14 | 71.43% | 14.29% | 92.86% | **85.71%** | **92.86%** | 78.6% |
| **Stable Diffusion 1.4** | 12 | 100.00% | 33.33% | 100.00% | **100.00%** | **100.00%** | 66.7% |
| **Stable Diffusion 2** | 15 | 73.33% | 26.67% | 93.33% | **93.33%** | **93.33%** | 66.7% |
| **Total AI Pool** | **100** | **74.00%** | **34.00%** | **95.00%** | **93.00%** | **95.00%** | **63.0%** |

### 4.3 Device Family Specificity (E19 Smartphone Benchmark, $N=100$ Reals)

| Device Family | $N$ | E6-C FPR | E20 FPR | E21 FPR | **E23 Blend** | **E23 Pure** | Pure % Routed E20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple iPhone (iPhone X)** | 35 | 5.71% | 0.00% | 5.71% | **5.71%** | **5.71%** | 94.3% |
| **Samsung Galaxy (Galaxy S9)** | 35 | 2.86% | 0.00% | 20.00% | **14.29%** | **20.00%** | 80.0% |
| **Google Pixel (Pixel 7–9)** | 30 | 43.33% | 3.33% | 66.67% | **63.33%** | **66.67%** | 36.7% |

*Observation:* While the pure router routed 80.0% of Samsung Galaxy S9 images to E20, the remaining 20.0% routed to E21 were all false positives, resulting in a 20.00% FPR. The continuous blend softened these decisions, reducing Samsung FPR to 14.29%.

---

## 5. Error Transitions & Complementarity

Across the 1,127 benchmark samples, cross-model agreement and transitions for the winning E23 configuration demonstrate persistent orthogonality:

```
========================================================================================================================
Benchmark Dataset         Total N   Both Correct   E20 Wrong + E21 Correct   E20 Correct + E21 Wrong   Both Simultaneous Wrong
========================================================================================================================
E14 Clean                   200          66                  62                        64                   8
E14 Degraded                600         204                 175                       181                  40
E19 Smartphone              200         140                  21                        33                   6
E19 WhatsApp                 60          23                  14                        21                   2
WhatsApp N=67                67          20                  17                        15                  15
------------------------------------------------------------------------------------------------------------------------
TOTAL                      1,127        453                 289                       314                  71
PERCENTAGE                 100%       40.20%               25.64%                    27.86%              6.30%
========================================================================================================================
```

### Transition Breakdown for E23 vs Base Models:
- **E20 Wrong $\rightarrow$ E23 Correct (Recovered):** 274 samples (massively recovering legacy AI generators).
- **E20 Correct $\rightarrow$ E23 Wrong (Regressed):** 286 samples (predominantly uncompressed RAISE-1k photos where E21 output high false-positive probabilities).
- **E21 Wrong $\rightarrow$ E23 Correct (Recovered):** 28 samples.
- **E21 Correct $\rightarrow$ E23 Wrong (Regressed):** 15 samples.

---

## 6. Critical Scientific Analysis & Stop Conditions

In accordance with experimental instructions, we analyze why routing did not surpass the fixed blend and evaluate whether model development should stop.

### 6.1 Why Simple Confidence Routing and Learned Feature Routers Failed
1. **The Overconfidence Inversion Paradox:**  
   Simple confidence routing (`|p - 0.5|`) is predicated on the assumption that extreme probabilities indicate reliable predictions. In forensic detection across out-of-domain distributions, this assumption is fundamentally false:
   - E20 outputs $p_{20} < 0.02$ on Stable Diffusion 1.3 because it has never seen that generator's spectral fingerprint during specialized training. It is *wrong with near-certainty*.
   - E21 outputs $p_{21} > 0.85$ on uncompressed RAW RAISE-1k images because its training balance was heavily weighted toward legacy artifact detection. It is *wrong with near-certainty*.
   - A router comparing confidence distances $|p - 0.5|$ simply picks the more overconfident model, amplifying catastrophic failures.
2. **Prediction-Space Insufficiency:**  
   The 9 routing features derived purely from $p_{e20}$ and $p_{e21}$ contain no semantic information about whether the input image is an uncompressed RAW DSLR capture, an iPhone portrait, or a social media transcode. Without metadata or image-level domain representations, the prediction vector alone cannot tell *which* model is hallucinating.
3. **Discrete Switching vs Continuous Regularization:**  
   Discrete routing forces a hard choice between E20 and E21. Continuous blending ($0.10 \times p_{20} + 0.90 \times p_{21}$) acts as a margin regularizer: it allows E21's high probabilities ($p > 0.90$) to trigger AI detections while allowing E20's low probabilities ($p < 0.05$) to pull borderline real images below the 0.50 threshold.

### 6.2 The Remaining Limiting Domains
1. **Uncompressed DSLR / RAW Sensors (RAISE-1k):** E21 suffers from a 67.0% FPR on clean camera photos. No linear combination or prediction-level router can suppress this without suppressing legacy AI recall.
2. **Social-Media Double Compression (WhatsApp):** Re-transcoded images remain at ~53–56% accuracy. Aggressive transcoding destroys subtle sensor noise and high-frequency generator artifacts simultaneously.

### 6.3 Evidence Assessment for Stopping Model Development
- **Checkpoints Tested:** E1 through E22 covering single-backbone CNNs, multi-scale spatial/frequency fusion (E6-C), global constrained gating (E18), failure-targeted retraining (E20), legacy rebalancing (E21), linear ensembling (E22), and confidence routing (E23).
- **Fundamental Trade-off Reached:** The experimental progression has mapped an empirical Pareto frontier:
  - High Smartphone Specificity (E20): 86.5% Acc, 1.0% FPR on phones, but 34.0% Recall on legacy AI.
  - High Generator Generalization (E21): 95.0% Recall on legacy AI, but 67.0% FPR on RAW photos.
  - Optimal Linear Compromise (E22 / E23): 67.0% Clean Acc, 80.0% Smartphone Acc, 93.0% AI Recall, 59.0% RAW FPR.
- **Scientific Verdict:** The evidence demonstrates that **further prediction-level heuristic routing, ensembling, or threshold manipulation on frozen weights has reached diminishing returns**. To achieve the theoretical 93.7% oracle potential without massive false positives, the system would require **domain-conditioned feature representation (e.g. metadata-informed camera ISP classifiers or end-to-end multi-task learning)** rather than post-hoc prediction routing.
- **Recommendation:** Model architecture and routing experiments should conclude. The current frozen benchmarks and findings provide a complete, honest, and rigorous empirical characterization of cross-generator generalization in digital image forensics.

---

## 7. Verification & Repository Integrity Audit

- **Checkpoints Unchanged (SHA256 verified):**
  - E20 Checkpoint: `f96eec36ffea15c7...` (Unmodified)
  - E21 Checkpoint: `8d6d9a0ae8fd6e1e...` (Unmodified)
  - E6-C Baseline Checkpoint: `2866365f28bc08e2...` (Unmodified)
- **External Benchmarks Unchanged:**
  - E14 Clean (`experiments/final_external_test/manifests/final_test_manifest.csv`)
  - E14 Degraded (`experiments/final_external_test/manifests/final_test_degraded_manifest.csv`)
  - E19 Smartphone (`data/e19_smartphone_benchmark/manifests/e19_test_manifest.csv`)
  - E19 WhatsApp & WhatsApp N=67
- **Production Integrity:**
  - Production inference code (`backend/inference.py`, `backend/main.py`) remained 100% unaltered.
  - Production decision threshold remained strictly at 0.50.
- **Git Hygiene:** Clean working tree status; zero commits or pushes created.
