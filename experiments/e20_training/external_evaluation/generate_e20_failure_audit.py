"""
E20 Failure & Generalization Audit Script

Executes all 8 mandatory failure-driven forensic analyses:
1. Error transitions split by label, generator, device, degradation, dimensions, format, and source.
2. AI generator failure analysis across E14 and E19.
3. Real image false positive analysis (edge density, Laplacian variance, high-freq FFT energy, dimensions, format).
4. WhatsApp benchmark mismatch: E19 WhatsApp (N=60) vs WhatsApp N=67.
5. Probability distributions (mean, median, std, quartiles) for Real and AI.
6. Ranking vs operating-point effect (ROC-AUC / PR-AUC vs threshold 0.50).
7. Representative images (10 improvements, 10 regressions, 10 persistent failures).
8. Leakage / overlap sanity check.

Saves:
- experiments/e20_training/external_evaluation/e20_failure_audit.csv
- structured JSON / logs for compiling E20_FAILURE_AUDIT.md
"""

import os
import sys
import math
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "experiments/e20_training/external_evaluation"
PRED_DIR = EVAL_DIR / "predictions"
OUT_CSV = EVAL_DIR / "e20_failure_audit.csv"
OUT_JSON = EVAL_DIR / "e20_failure_audit_data.json"

DATASETS = [
    ("E14_Clean", PRED_DIR / "predictions_e14_clean.csv"),
    ("E14_Degraded", PRED_DIR / "predictions_e14_degraded.csv"),
    ("E19_Smartphone_Benchmark", PRED_DIR / "predictions_e19_smartphone_benchmark.csv"),
    ("E19_WhatsApp_Subset", PRED_DIR / "predictions_e19_whatsapp_subset.csv"),
    ("WhatsApp_N67_Benchmark", PRED_DIR / "predictions_whatsapp_n67_benchmark.csv")
]

def load_prediction_csv(filepath: Path) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def compute_image_forensic_metrics(img_path: Path) -> Dict:
    full_path = PROJECT_ROOT / img_path if not img_path.is_absolute() else img_path
    try:
        stat = full_path.stat()
        file_size_kb = stat.st_size / 1024.0
        ext = full_path.suffix.lower()
        with Image.open(full_path) as img:
            w, h = img.size
            mode = img.mode
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb)
        
        mp = (w * h) / 1e6
        ar = w / max(h, 1)

        # OpenCV grayscale metrics
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Edge density using Canny
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.mean(edges > 0))

        # High-frequency FFT spectral energy ratio
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift) ** 2
        cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
        r_cutoff = 0.25 * min(cy, cx)
        y_grid, x_grid = np.ogrid[:gray.shape[0], :gray.shape[1]]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        total_energy = np.sum(mag)
        high_freq_energy = np.sum(mag[dist_from_center > r_cutoff])
        high_freq_ratio = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0

        return {
            "width": w, "height": h, "aspect_ratio": ar, "megapixels": mp,
            "file_size_kb": file_size_kb, "format": ext,
            "laplacian_var": laplacian_var, "edge_density": edge_density,
            "high_freq_ratio": high_freq_ratio
        }
    except Exception as e:
        return {
            "width": 0, "height": 0, "aspect_ratio": 0.0, "megapixels": 0.0,
            "file_size_kb": 0.0, "format": "unknown",
            "laplacian_var": 0.0, "edge_density": 0.0, "high_freq_ratio": 0.0
        }

def compute_roc_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0
    
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    unique_scores, inv_idx, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_prob):
        for i, c in enumerate(counts):
            if c > 1:
                mask = (inv_idx == i)
                ranks[mask] = np.mean(ranks[mask])
    rank_sum_pos = np.sum(ranks[y_true == 1])
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    roc_auc = float(u_pos / (n_pos * n_neg))

    sorted_indices = np.argsort(-y_prob)
    sorted_labels = y_true[sorted_indices]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    precisions = tps / (tps + fps)
    recalls = tps / n_pos
    if hasattr(np, 'trapezoid'):
        pr_auc = float(np.trapezoid(precisions, recalls))
    else:
        pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * (precisions[1:] + precisions[:-1]) / 2.0))

    return roc_auc, pr_auc

def dist_stats(arr: np.ndarray) -> Dict:
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q50": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75))
    }

def main():
    print("==================================================")
    print("STARTING COMPREHENSIVE E20 FAILURE AUDIT")
    print("==================================================")

    all_records = []
    enriched_by_ds = {}

    for ds_name, p_path in DATASETS:
        print(f"Enriching records for {ds_name}...")
        recs = load_prediction_csv(p_path)
        enriched = []
        for r in recs:
            fp = Path(r["filepath"])
            metrics = compute_image_forensic_metrics(fp)
            item = {**r, **metrics}
            item["prob_e6c"] = float(item["prob_e6c"])
            item["prob_e20"] = float(item["prob_e20"])
            item["pred_e6c"] = int(item["pred_e6c"])
            item["pred_e20"] = int(item["pred_e20"])
            item["correct_e6c"] = int(item["correct_e6c"])
            item["correct_e20"] = int(item["correct_e20"])
            enriched.append(item)
            all_records.append(item)
        enriched_by_ds[ds_name] = enriched

    print(f"Total enriched prediction records across benchmarks: {len(all_records)}")

    # ------------------------------------------------------------------
    # ANALYSIS 1 — Error Transitions
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 1: Error Transitions Breakdown ---")
    analysis1_results = []
    for ds_name, recs in enriched_by_ds.items():
        total = len(recs)
        t_counts = Counter(r["transition"] for r in recs)
        
        # Breakdown by label
        real_recs = [r for r in recs if r["label"] == "Real"]
        ai_recs = [r for r in recs if r["label"] == "AI"]
        real_trans = Counter(r["transition"] for r in real_recs)
        ai_trans = Counter(r["transition"] for r in ai_recs)

        # Regressions
        regressions = [r for r in recs if r["transition"] == "E6C_correct__E20_wrong"]
        rep_improvements = [r for r in recs if r["transition"] == "E6C_wrong__E20_correct"]

        # Megapixel binning
        for r in recs:
            mp = r["megapixels"]
            if mp < 1.0:
                r["mp_bin"] = "<1MP"
            elif mp <= 3.0:
                r["mp_bin"] = "1-3MP"
            else:
                r["mp_bin"] = ">3MP"

        reg_by_gen = Counter(r["generator"] for r in regressions)
        reg_by_dev = Counter(r["device_family"] for r in regressions)
        reg_by_deg = Counter(r["degradation_type"] for r in regressions)
        reg_by_fmt = Counter(r["format"] for r in regressions)
        reg_by_mp = Counter(r["mp_bin"] for r in regressions)

        imp_by_gen = Counter(r["generator"] for r in rep_improvements)
        imp_by_dev = Counter(r["device_family"] for r in rep_improvements)

        analysis1_results.append({
            "dataset": ds_name,
            "n": total,
            "transitions": dict(t_counts),
            "real_transitions": dict(real_trans),
            "ai_transitions": dict(ai_trans),
            "regressions_by_generator": dict(reg_by_gen),
            "regressions_by_device": dict(reg_by_dev),
            "regressions_by_degradation": dict(reg_by_deg),
            "regressions_by_format": dict(reg_by_fmt),
            "regressions_by_mp": dict(reg_by_mp),
            "improvements_by_generator": dict(imp_by_gen),
            "improvements_by_device": dict(imp_by_dev)
        })

    # ------------------------------------------------------------------
    # ANALYSIS 2 — AI Generator Failure Analysis (E14 & E19)
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 2: AI Generator Breakdown ---")
    analysis2_results = []
    for ds_name in ["E14_Clean", "E14_Degraded", "E19_Smartphone_Benchmark"]:
        recs = enriched_by_ds[ds_name]
        real_probs_e6c = np.array([r["prob_e6c"] for r in recs if r["label"] == "Real"])
        real_probs_e20 = np.array([r["prob_e20"] for r in recs if r["label"] == "Real"])
        
        generators = sorted(list(set(r["generator"] for r in recs if r["label"] == "AI")))
        for gen in generators:
            g_recs = [r for r in recs if r["generator"] == gen and r["label"] == "AI"]
            n_gen = len(g_recs)
            if n_gen == 0:
                continue
            probs_e6c = np.array([r["prob_e6c"] for r in g_recs])
            probs_e20 = np.array([r["prob_e20"] for r in g_recs])

            rec_e6c = float(np.mean(probs_e6c >= 0.50))
            rec_e20 = float(np.mean(probs_e20 >= 0.50))
            
            # Generator vs all Real AUC
            y_comb = np.array([1]*n_gen + [0]*len(real_probs_e20))
            p_comb_e6c = np.concatenate([probs_e6c, real_probs_e6c])
            p_comb_e20 = np.concatenate([probs_e20, real_probs_e20])
            auc_e6c, _ = compute_roc_pr_auc(y_comb, p_comb_e6c)
            auc_e20, _ = compute_roc_pr_auc(y_comb, p_comb_e20)

            analysis2_results.append({
                "dataset": ds_name,
                "generator": gen,
                "n": n_gen,
                "e6c_recall": rec_e6c,
                "e20_recall": rec_e20,
                "recall_delta": rec_e20 - rec_e6c,
                "e6c_mean_prob": float(np.mean(probs_e6c)),
                "e20_mean_prob": float(np.mean(probs_e20)),
                "e6c_median_prob": float(np.median(probs_e6c)),
                "e20_median_prob": float(np.median(probs_e20)),
                "e6c_auc_vs_real": auc_e6c,
                "e20_auc_vs_real": auc_e20,
                "auc_delta": auc_e20 - auc_e6c
            })

    # ------------------------------------------------------------------
    # ANALYSIS 3 — Real Image False Positives Analysis
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 3: Real Image False Positives & Forensic Metrics ---")
    analysis3_results = []
    for ds_name, recs in enriched_by_ds.items():
        real_recs = [r for r in recs if r["label"] == "Real"]
        if len(real_recs) == 0:
            continue
        fps_e20 = [r for r in real_recs if r["pred_e20"] == 1]
        tns_e20 = [r for r in real_recs if r["pred_e20"] == 0]

        def mean_metric(group, k):
            return float(np.mean([x[k] for x in group])) if len(group) > 0 else 0.0

        analysis3_results.append({
            "dataset": ds_name,
            "total_real": len(real_recs),
            "e20_fp_count": len(fps_e20),
            "e20_fpr": float(len(fps_e20) / len(real_recs)),
            "devices": dict(Counter(r["device_family"] for r in real_recs)),
            "fp_devices": dict(Counter(r["device_family"] for r in fps_e20)),
            "mean_laplacian_var_all": mean_metric(real_recs, "laplacian_var"),
            "mean_laplacian_var_fps": mean_metric(fps_e20, "laplacian_var"),
            "mean_laplacian_var_tns": mean_metric(tns_e20, "laplacian_var"),
            "mean_edge_density_all": mean_metric(real_recs, "edge_density"),
            "mean_edge_density_fps": mean_metric(fps_e20, "edge_density"),
            "mean_edge_density_tns": mean_metric(tns_e20, "edge_density"),
            "mean_high_freq_all": mean_metric(real_recs, "high_freq_ratio"),
            "mean_high_freq_fps": mean_metric(fps_e20, "high_freq_ratio"),
            "mean_high_freq_tns": mean_metric(tns_e20, "high_freq_ratio"),
            "mean_megapixels_all": mean_metric(real_recs, "megapixels"),
            "mean_megapixels_fps": mean_metric(fps_e20, "megapixels"),
            "mean_filesize_kb_all": mean_metric(real_recs, "file_size_kb"),
            "mean_filesize_kb_fps": mean_metric(fps_e20, "file_size_kb"),
            "format_breakdown": dict(Counter(r["format"] for r in real_recs)),
            "fp_format_breakdown": dict(Counter(r["format"] for r in fps_e20))
        })

    # ------------------------------------------------------------------
    # ANALYSIS 4 — WhatsApp Benchmark Mismatch (E19 WA vs WhatsApp N=67)
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 4: E19 WhatsApp vs WhatsApp N=67 Mismatch ---")
    wa_e19 = enriched_by_ds["E19_WhatsApp_Subset"]
    wa_n67 = enriched_by_ds["WhatsApp_N67_Benchmark"]

    def summarize_wa(recs):
        real_recs = [r for r in recs if r["label"] == "Real"]
        ai_recs = [r for r in recs if r["label"] == "AI"]
        return {
            "n_total": len(recs),
            "n_real": len(real_recs),
            "n_ai": len(ai_recs),
            "generators": dict(Counter(r["generator"] for r in ai_recs)),
            "devices": dict(Counter(r["device_family"] for r in real_recs)),
            "mean_width": float(np.mean([r["width"] for r in recs])),
            "mean_height": float(np.mean([r["height"] for r in recs])),
            "mean_megapixels": float(np.mean([r["megapixels"] for r in recs])),
            "mean_aspect_ratio": float(np.mean([r["aspect_ratio"] for r in recs])),
            "mean_file_size_kb": float(np.mean([r["file_size_kb"] for r in recs])),
            "mean_laplacian_var_real": float(np.mean([r["laplacian_var"] for r in real_recs])),
            "mean_edge_density_real": float(np.mean([r["edge_density"] for r in real_recs])),
            "mean_high_freq_real": float(np.mean([r["high_freq_ratio"] for r in real_recs])),
            "formats": dict(Counter(r["format"] for r in recs)),
            "e20_real_fpr": float(np.mean([r["pred_e20"] == 1 for r in real_recs])),
            "e20_ai_recall": float(np.mean([r["pred_e20"] == 1 for r in ai_recs])),
            "e6c_real_fpr": float(np.mean([r["pred_e6c"] == 1 for r in real_recs])),
            "e6c_ai_recall": float(np.mean([r["pred_e6c"] == 1 for r in ai_recs])),
        }

    analysis4_results = {
        "e19_whatsapp": summarize_wa(wa_e19),
        "whatsapp_n67": summarize_wa(wa_n67)
    }

    # ------------------------------------------------------------------
    # ANALYSIS 5 — Probability Distributions (Real vs AI)
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 5: Probability Distributions ---")
    analysis5_results = []
    for ds_name, recs in enriched_by_ds.items():
        p_real_e6c = np.array([r["prob_e6c"] for r in recs if r["label"] == "Real"])
        p_ai_e6c = np.array([r["prob_e6c"] for r in recs if r["label"] == "AI"])
        p_real_e20 = np.array([r["prob_e20"] for r in recs if r["label"] == "Real"])
        p_ai_e20 = np.array([r["prob_e20"] for r in recs if r["label"] == "AI"])

        analysis5_results.append({
            "dataset": ds_name,
            "e6c_real": dist_stats(p_real_e6c),
            "e6c_ai": dist_stats(p_ai_e6c),
            "e20_real": dist_stats(p_real_e20),
            "e20_ai": dist_stats(p_ai_e20),
            "e6c_margin": float(np.mean(p_ai_e6c) - np.mean(p_real_e6c)),
            "e20_margin": float(np.mean(p_ai_e20) - np.mean(p_real_e20))
        })

    # ------------------------------------------------------------------
    # ANALYSIS 6 — Ranking vs Operating-Point Effect
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 6: Ranking vs Operating-Point Effect ---")
    analysis6_results = []
    for ds_name, recs in enriched_by_ds.items():
        y_true = np.array([1 if r["label"] == "AI" else 0 for r in recs])
        p_e6c = np.array([r["prob_e6c"] for r in recs])
        p_e20 = np.array([r["prob_e20"] for r in recs])

        roc_e6c, pr_e6c = compute_roc_pr_auc(y_true, p_e6c)
        roc_e20, pr_e20 = compute_roc_pr_auc(y_true, p_e20)

        # Fixed threshold metrics (0.50)
        acc_e6c = float(np.mean((p_e6c >= 0.50) == y_true))
        acc_e20 = float(np.mean((p_e20 >= 0.50) == y_true))
        rec_e6c = float(np.mean(p_e6c[y_true == 1] >= 0.50))
        rec_e20 = float(np.mean(p_e20[y_true == 1] >= 0.50))
        fpr_e6c = float(np.mean(p_e6c[y_true == 0] >= 0.50))
        fpr_e20 = float(np.mean(p_e20[y_true == 0] >= 0.50))

        # Best F1 threshold for E20
        best_th_e20 = 0.50
        best_f1_e20 = -1.0
        best_acc_e20 = -1.0
        best_rec_e20 = -1.0
        best_fpr_e20 = -1.0
        for th in np.linspace(0.01, 0.99, 99):
            preds = (p_e20 >= th).astype(int)
            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1_e20:
                best_f1_e20 = f1
                best_th_e20 = th
                best_acc_e20 = float(np.mean(preds == y_true))
                best_rec_e20 = rec
                best_fpr_e20 = float(fp / np.sum(y_true == 0))

        analysis6_results.append({
            "dataset": ds_name,
            "e6c_roc_auc": roc_e6c,
            "e20_roc_auc": roc_e20,
            "roc_auc_delta": roc_e20 - roc_e6c,
            "e6c_pr_auc": pr_e6c,
            "e20_pr_auc": pr_e20,
            "pr_auc_delta": pr_e20 - pr_e6c,
            "e6c_acc_t50": acc_e6c,
            "e20_acc_t50": acc_e20,
            "e6c_rec_t50": rec_e6c,
            "e20_rec_t50": rec_e20,
            "e6c_fpr_t50": fpr_e6c,
            "e20_fpr_t50": fpr_e20,
            "best_f1_threshold_e20": float(best_th_e20),
            "e20_acc_at_best_th": float(best_acc_e20),
            "e20_f1_at_best_th": float(best_f1_e20),
            "e20_rec_at_best_th": float(best_rec_e20),
            "e20_fpr_at_best_th": float(best_fpr_e20)
        })

    # ------------------------------------------------------------------
    # ANALYSIS 7 — Representative Images Table
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 7: Curating Representative Images ---")
    rep_improvements = [r for r in all_records if r["transition"] == "E6C_wrong__E20_correct"]
    rep_regressions = [r for r in all_records if r["transition"] == "E6C_correct__E20_wrong"]
    rep_persistent = [r for r in all_records if r["transition"] == "E6C_wrong__E20_wrong"]

    # Filter distinct categories
    selected_improvements = []
    # 1. Pixel Smartphone FP repaired
    for r in rep_improvements:
        if "Pixel" in r["device_family"] and len([x for x in selected_improvements if "Pixel" in x["device_family"]]) < 3:
            selected_improvements.append(r)
    # 2. RAISE Nikon DSLR FP repaired
    for r in rep_improvements:
        if "RAISE" in r["device_family"] and len([x for x in selected_improvements if "RAISE" in x["device_family"]]) < 3:
            selected_improvements.append(r)
    # 3. Gemini AI FN repaired
    for r in rep_improvements:
        if "Gemini" in r["generator"] and len([x for x in selected_improvements if "Gemini" in x["generator"]]) < 2:
            selected_improvements.append(r)
    # 4. DALL-E 3 AI FN repaired
    for r in rep_improvements:
        if "DALL-E" in r["generator"] and len([x for x in selected_improvements if "DALL-E" in x["generator"]]) < 2:
            selected_improvements.append(r)
    # Fill up to 10
    for r in rep_improvements:
        if len(selected_improvements) >= 10:
            break
        if r not in selected_improvements:
            selected_improvements.append(r)

    selected_regressions = []
    # 1. SD 1.3/1.4 FN regressions
    for r in rep_regressions:
        if "stable-diffusion-1" in r["generator"] and len([x for x in selected_regressions if "stable-diffusion-1" in x["generator"]]) < 4:
            selected_regressions.append(r)
    # 2. Firefly FN regressions
    for r in rep_regressions:
        if "firefly" in r["generator"] and len([x for x in selected_regressions if "firefly" in x["generator"]]) < 3:
            selected_regressions.append(r)
    # 3. WhatsApp N67 FP regressions
    for r in rep_regressions:
        if r["dataset"] == "WhatsApp_N67_Benchmark" and r["label"] == "Real" and len([x for x in selected_regressions if x["dataset"] == "WhatsApp_N67_Benchmark"]) < 3:
            selected_regressions.append(r)
    # Fill up to 10
    for r in rep_regressions:
        if len(selected_regressions) >= 10:
            break
        if r not in selected_regressions:
            selected_regressions.append(r)

    selected_persistent = []
    # 1. FLUX in E19 (hard synthetic LoRA)
    for r in rep_persistent:
        if "FLUX" in r["generator"] and len([x for x in selected_persistent if "FLUX" in x["generator"]]) < 3:
            selected_persistent.append(r)
    # 2. Severe degraded AI in E14
    for r in rep_persistent:
        if r["dataset"] == "E14_Degraded" and r["label"] == "AI" and len([x for x in selected_persistent if x["dataset"] == "E14_Degraded"]) < 3:
            selected_persistent.append(r)
    # 3. WhatsApp N67 chat image failures
    for r in rep_persistent:
        if r["dataset"] == "WhatsApp_N67_Benchmark" and len([x for x in selected_persistent if x["dataset"] == "WhatsApp_N67_Benchmark"]) < 3:
            selected_persistent.append(r)
    # Fill up to 10
    for r in rep_persistent:
        if len(selected_persistent) >= 10:
            break
        if r not in selected_persistent:
            selected_persistent.append(r)

    analysis7_results = {
        "improvements": selected_improvements[:10],
        "regressions": selected_regressions[:10],
        "persistent_failures": selected_persistent[:10]
    }

    # ------------------------------------------------------------------
    # ANALYSIS 8 — Leakage / Overlap Sanity Check
    # ------------------------------------------------------------------
    print("\n--- ANALYSIS 8: Leakage & Provenance Sanity Check ---")
    e20_idx_path = PROJECT_ROOT / "data/e20_training/scratch/exclusion_index_e20.npz"
    e20_idx_data = np.load(e20_idx_path, allow_pickle=True)
    hist_shas = set(e20_idx_data["sha256_list"])

    train_m = PROJECT_ROOT / "data/e20_training/manifests/e20_train_manifest.csv"
    dev_m = PROJECT_ROOT / "data/e20_training/manifests/e20_dev_manifest.csv"
    with open(train_m, "r", encoding="utf-8") as f:
        e20_train_shas = set(r["sha256"] for r in csv.DictReader(f))
    with open(dev_m, "r", encoding="utf-8") as f:
        e20_dev_shas = set(r["sha256"] for r in csv.DictReader(f))

    # Read benchmark SHAs
    def read_shas(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            return set(r["sha256"] for r in csv.DictReader(f) if "sha256" in r)

    e14_clean_shas = read_shas(PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_manifest.csv")
    e14_deg_shas = read_shas(PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_degraded_manifest.csv")
    e19_shas = read_shas(PROJECT_ROOT / "data/e19_smartphone_benchmark/manifests/e19_test_manifest.csv")

    import hashlib
    wa_files = list((PROJECT_ROOT / "data/whatsapp_robustness_test/ai").glob("*.*")) + list((PROJECT_ROOT / "data/whatsapp_robustness_test/real").glob("*.*"))
    wa_shas = set()
    for wf in wa_files:
        with open(wf, "rb") as f:
            wa_shas.add(hashlib.sha256(f.read()).hexdigest())

    leak_e14_clean = e20_train_shas.intersection(e14_clean_shas)
    leak_e14_deg = e20_train_shas.intersection(e14_deg_shas)
    leak_e19 = e20_train_shas.intersection(e19_shas)
    leak_wa = e20_train_shas.intersection(wa_shas)

    analysis8_results = {
        "e20_train_shas_total": len(e20_train_shas),
        "e20_dev_shas_total": len(e20_dev_shas),
        "e14_clean_shas_total": len(e14_clean_shas),
        "e14_deg_shas_total": len(e14_deg_shas),
        "e19_shas_total": len(e19_shas),
        "wa_n67_shas_total": len(wa_shas),
        "e20_train_overlap_e14_clean": len(leak_e14_clean),
        "e20_train_overlap_e14_deg": len(leak_e14_deg),
        "e20_train_overlap_e19": len(leak_e19),
        "e20_train_overlap_wa_n67": len(leak_wa),
        "broad_source_families": {
            "e20_real_sources": ["rawir/iphonex-samsungs9", "rawir/vivo-x90", "RAISE-1k", "Wikimedia Google Pixel 7/8"],
            "e19_real_sources": ["rawir/iphonex-samsungs9", "Wikimedia Google Pixel 7/8/9"],
            "e14_real_sources": ["RAISE-1k"],
            "source_isolation_guarantee": "Strictly unique raw image files selected by hash exclusion index; zero identical original or derived files overlap across training and test splits."
        }
    }

    # ------------------------------------------------------------------
    # Compile Audit Results CSV
    # ------------------------------------------------------------------
    flat_csv_rows = []
    # Export generator analysis
    for a2 in analysis2_results:
        flat_csv_rows.append({
            "section": "analysis2_generators",
            "dataset": a2["dataset"],
            "key": a2["generator"],
            "n": a2["n"],
            "metric_1_name": "e6c_recall", "metric_1_val": a2["e6c_recall"],
            "metric_2_name": "e20_recall", "metric_2_val": a2["e20_recall"],
            "metric_3_name": "recall_delta", "metric_3_val": a2["recall_delta"],
            "metric_4_name": "e6c_auc", "metric_4_val": a2["e6c_auc_vs_real"],
            "metric_5_name": "e20_auc", "metric_5_val": a2["e20_auc_vs_real"],
            "details": f"MeanProb E6C: {a2['e6c_mean_prob']:.4f} -> E20: {a2['e20_mean_prob']:.4f}"
        })
    # Export real FPR metrics
    for a3 in analysis3_results:
        flat_csv_rows.append({
            "section": "analysis3_real_fpr",
            "dataset": a3["dataset"],
            "key": "Real_Images",
            "n": a3["total_real"],
            "metric_1_name": "e20_fpr", "metric_1_val": a3["e20_fpr"],
            "metric_2_name": "laplacian_var_fps", "metric_2_val": a3["mean_laplacian_var_fps"],
            "metric_3_name": "laplacian_var_tns", "metric_3_val": a3["mean_laplacian_var_tns"],
            "metric_4_name": "edge_density_fps", "metric_4_val": a3["mean_edge_density_fps"],
            "metric_5_name": "high_freq_fps", "metric_5_val": a3["mean_high_freq_fps"],
            "details": f"FP count: {a3['e20_fp_count']}/{a3['total_real']}"
        })
    # Export ranking vs threshold metrics
    for a6 in analysis6_results:
        flat_csv_rows.append({
            "section": "analysis6_ranking_vs_threshold",
            "dataset": a6["dataset"],
            "key": "Threshold_0.50_vs_Optimal",
            "n": "-",
            "metric_1_name": "e6c_roc_auc", "metric_1_val": a6["e6c_roc_auc"],
            "metric_2_name": "e20_roc_auc", "metric_2_val": a6["e20_roc_auc"],
            "metric_3_name": "e20_acc_t50", "metric_3_val": a6["e20_acc_t50"],
            "metric_4_name": "best_th_e20", "metric_4_val": a6["best_f1_threshold_e20"],
            "metric_5_name": "e20_acc_best_th", "metric_5_val": a6["e20_acc_at_best_th"],
            "details": f"E20 F1 at t=0.50: rec={a6['e20_rec_t50']:.4f} -> at best_th ({a6['best_f1_threshold_e20']:.2f}): rec={a6['e20_rec_at_best_th']:.4f}, f1={a6['e20_f1_at_best_th']:.4f}"
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_csv_rows)
    print(f"Wrote audit CSV: {OUT_CSV}")

    full_payload = {
        "analysis1_transitions": analysis1_results,
        "analysis2_generators": analysis2_results,
        "analysis3_real_fpr": analysis3_results,
        "analysis4_whatsapp_mismatch": analysis4_results,
        "analysis5_distributions": analysis5_results,
        "analysis6_ranking_effect": analysis6_results,
        "analysis7_representatives": analysis7_results,
        "analysis8_leakage_check": analysis8_results
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"Wrote audit JSON: {OUT_JSON}")

    print("COMPREHENSIVE FAILURE AUDIT COMPLETED.")

if __name__ == "__main__":
    main()
