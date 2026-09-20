"""
Experiment 23 (E23): Confidence-Aware E20/E21 Model Routing.

Objective:
Test whether a small routing model/rule can exploit the complementary failure modes
of the frozen E20 and E21 detectors better than the fixed E22 0.1/0.9 ensemble.

Protocol & Constraints:
- NO training of detectors. NO fine-tuning.
- NO checkpoint modifications.
- NO dataset modifications.
- NO production modifications.
- NO threshold tuning (Fixed threshold = 0.50).
- Router design & selection strictly conducted on E21 DEV split ONLY (N=812).
- Sealed external benchmarks remain untouched until router is completely frozen.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model
from experiments.e22_ensemble_study.scripts.run_e22_study import EvalDataset, run_model_inference, compute_metrics

# Paths
E6C_CHECKPOINT = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
E20_CHECKPOINT = PROJECT_ROOT / "experiments/e20_training/checkpoints/e20_best_model.pt"
E21_CHECKPOINT = PROJECT_ROOT / "experiments/e21_targeted_data/checkpoints/e21_best_model.pt"

E21_DEV_MANIFEST = PROJECT_ROOT / "data/e21_targeted_data/manifests/e21_dev_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data/e21_targeted_data"

E22_PRED_DIR = PROJECT_ROOT / "experiments/e22_ensemble_study/predictions"

OUT_DIR = PROJECT_ROOT / "experiments/e23_model_routing"
SCRIPTS_DIR = OUT_DIR / "scripts"
PRED_DIR = OUT_DIR / "predictions"
SUBGROUP_DIR = OUT_DIR / "subgroups"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_features(p20: np.ndarray, p21: np.ndarray) -> np.ndarray:
    """
    Construct the 9 interpretable routing features:
    1. p_e20
    2. p_e21
    3. |p_e20 - p_e21|
    4. (p_e20 + p_e21) / 2
    5. min(p_e20, p_e21)
    6. max(p_e20, p_e21)
    7. |p_e20 - 0.5|
    8. |p_e21 - 0.5|
    9. agreement: +1 (both >= 0.5), -1 (both < 0.5), 0 (disagreement)
    """
    f1 = p20
    f2 = p21
    f3 = np.abs(p20 - p21)
    f4 = (p20 + p21) / 2.0
    f5 = np.minimum(p20, p21)
    f6 = np.maximum(p20, p21)
    f7 = np.abs(p20 - 0.5)
    f8 = np.abs(p21 - 0.5)
    f9 = np.where((p20 >= 0.5) & (p21 >= 0.5), 1.0,
         np.where((p20 < 0.5) & (p21 < 0.5), -1.0, 0.0))
    return np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9])

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SUBGROUP_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STARTING EXPERIMENT 23: MODEL ROUTING STUDY")
    print("==================================================")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Dev Split Evaluation & Feature Construction
    dev_cache = OUT_DIR / "e21_dev_predictions.csv"
    if dev_cache.exists():
        print(f"Loading cached Dev predictions from {dev_cache}...")
        dev_rows = []
        with open(dev_cache, "r", encoding="utf-8") as f:
            dev_rows = list(csv.DictReader(f))
        p_e20_dev = np.array([float(r["prob_e20"]) for r in dev_rows])
        p_e21_dev = np.array([float(r["prob_e21"]) for r in dev_rows])
        y_dev = np.array([1 if r["label"] == "AI" else 0 for r in dev_rows])
    else:
        print("Computing E20 and E21 predictions on E21 Dev (N=812)...")
        with open(E21_DEV_MANIFEST, "r", encoding="utf-8") as f:
            dev_records = list(csv.DictReader(f))
        assert len(dev_records) == 812, f"Expected 812 records in E21 Dev, got {len(dev_records)}"

        dev_dataset = EvalDataset(dev_records, root=DATA_ROOT)
        dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

        m20 = MultiViewE5Model()
        m20.load_state_dict(torch.load(E20_CHECKPOINT, map_location="cpu")["model_state_dict"], strict=True)
        m20.to(DEVICE).eval()
        p_e20_dev = run_model_inference(m20, dev_loader)
        del m20
        torch.cuda.empty_cache()

        m21 = MultiViewE5Model()
        m21.load_state_dict(torch.load(E21_CHECKPOINT, map_location="cpu")["model_state_dict"], strict=True)
        m21.to(DEVICE).eval()
        p_e21_dev = run_model_inference(m21, dev_loader)
        del m21
        torch.cuda.empty_cache()

        y_dev = np.array([1 if r["label"] == "AI" else 0 for r in dev_records])

        # Save dev cache
        with open(dev_cache, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "label", "prob_e20", "prob_e21"])
            writer.writeheader()
            for rec, p20, p21 in zip(dev_records, p_e20_dev, p_e21_dev):
                writer.writerow({
                    "image_id": rec["image_id"],
                    "label": rec["label"],
                    "prob_e20": f"{p20:.6f}",
                    "prob_e21": f"{p21:.6f}"
                })
        print(f"Dev predictions saved to {dev_cache}")

    X_dev = build_features(p_e20_dev, p_e21_dev)

    # 2. Candidate Definition & Execution on E21 Dev
    print("\n[Phase A] Evaluating Router Candidates strictly on E21 Dev split (N=812)...")

    # Define targets for learned router (choose E20=1 vs choose E21=0)
    # Target 1: Model closer to true binary label
    target_closer_dev = np.where(np.abs(p_e20_dev - y_dev) < np.abs(p_e21_dev - y_dev), 1, 0)
    
    # Target 2: Correctness-aware target (where one is right and the other is wrong)
    corr20_dev = ((p_e20_dev >= 0.5) == y_dev)
    corr21_dev = ((p_e21_dev >= 0.5) == y_dev)
    target_correct_dev = np.where(corr20_dev & ~corr21_dev, 1,
                         np.where(~corr20_dev & corr21_dev, 0,
                         np.where(np.abs(p_e20_dev - y_dev) <= np.abs(p_e21_dev - y_dev), 1, 0)))

    # Fit small interpretable learned routers on E21 Dev features
    lr_router = LogisticRegression(max_iter=1000, random_state=42)
    lr_router.fit(X_dev, target_correct_dev)

    dt2_router = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt2_router.fit(X_dev, target_correct_dev)

    dt3_router = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt3_router.fit(X_dev, target_correct_dev)

    # Helper function to execute a router candidate given p20, p21, X
    # Returns (p_out, route_decisions) where route_decisions is 'E20' or 'E21' (or 'Blend')
    def run_candidate(cand_id: str, p20: np.ndarray, p21: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        c20 = X[:, 6] # |p20 - 0.5|
        c21 = X[:, 7] # |p21 - 0.5|
        agr = X[:, 8] # agreement

        if cand_id == "A_E20_only":
            return p20, ["E20"] * len(p20)
        elif cand_id == "B_E21_only":
            return p21, ["E21"] * len(p21)
        elif cand_id == "C_E22_blend":
            return 0.1 * p20 + 0.9 * p21, ["Blend"] * len(p20)
        elif cand_id == "D1_conf_max":
            route = np.where(c20 >= c21, "E20", "E21")
            p = np.where(c20 >= c21, p20, p21)
            return p, route.tolist()
        elif cand_id == "D2_conf_min":
            route = np.where(c20 < c21, "E20", "E21")
            p = np.where(c20 < c21, p20, p21)
            return p, route.tolist()
        elif cand_id == "D3_e21_override_p80":
            # Default to E20, but route to E21 if E21 >= 0.80
            route = np.where(p21 >= 0.80, "E21", "E20")
            p = np.where(p21 >= 0.80, p21, p20)
            return p, route.tolist()
        elif cand_id == "D4_e21_override_p70":
            # Default to E20, but route to E21 if E21 >= 0.70
            route = np.where(p21 >= 0.70, "E21", "E20")
            p = np.where(p21 >= 0.70, p21, p20)
            return p, route.tolist()
        elif cand_id == "D5_e20_override_p10":
            # Default to E21, but route to E20 if E20 <= 0.10
            route = np.where(p20 <= 0.10, "E20", "E21")
            p = np.where(p20 <= 0.10, p20, p21)
            return p, route.tolist()
        elif cand_id == "D6_e20_override_p20":
            # Default to E21, but route to E20 if E20 <= 0.20
            route = np.where(p20 <= 0.20, "E20", "E21")
            p = np.where(p20 <= 0.20, p20, p21)
            return p, route.tolist()
        elif cand_id == "D7_disagree_max_conf":
            # If agree, use E21; if disagree, use whichever has max distance from 0.5
            route = np.where(agr != 0, "E21", np.where(c20 >= c21, "E20", "E21"))
            p = np.where(agr != 0, p21, np.where(c20 >= c21, p20, p21))
            return p, route.tolist()
        elif cand_id == "D8_disagree_e20":
            # If agree, use E21; if disagree, use E20
            route = np.where(agr != 0, "E21", "E20")
            p = np.where(agr != 0, p21, p20)
            return p, route.tolist()
        elif cand_id == "D9_disagree_e21":
            # If agree, use E20; if disagree, use E21
            route = np.where(agr != 0, "E20", "E21")
            p = np.where(agr != 0, p20, p21)
            return p, route.tolist()
        elif cand_id == "E_logistic_regression":
            dec = lr_router.predict(X) # 1: E20, 0: E21
            route = np.where(dec == 1, "E20", "E21")
            p = np.where(dec == 1, p20, p21)
            return p, route.tolist()
        elif cand_id == "F1_decision_tree_d2":
            dec = dt2_router.predict(X)
            route = np.where(dec == 1, "E20", "E21")
            p = np.where(dec == 1, p20, p21)
            return p, route.tolist()
        elif cand_id == "F2_decision_tree_d3":
            dec = dt3_router.predict(X)
            route = np.where(dec == 1, "E20", "E21")
            p = np.where(dec == 1, p20, p21)
            return p, route.tolist()
        else:
            raise ValueError(f"Unknown candidate {cand_id}")

    candidate_ids = [
        "A_E20_only",
        "B_E21_only",
        "C_E22_blend",
        "D1_conf_max",
        "D2_conf_min",
        "D3_e21_override_p80",
        "D4_e21_override_p70",
        "D5_e20_override_p10",
        "D6_e20_override_p20",
        "D7_disagree_max_conf",
        "D8_disagree_e20",
        "D9_disagree_e21",
        "E_logistic_regression",
        "F1_decision_tree_d2",
        "F2_decision_tree_d3",
    ]

    candidate_descriptions = {
        "A_E20_only": "E20-only prediction",
        "B_E21_only": "E21-only prediction",
        "C_E22_blend": "Fixed blend: 0.1*E20 + 0.9*E21",
        "D1_conf_max": "Confidence Max: if |p20-0.5| >= |p21-0.5| use E20 else E21",
        "D2_conf_min": "Confidence Min: if |p20-0.5| < |p21-0.5| use E20 else E21",
        "D3_e21_override_p80": "E20 default, override to E21 if p21 >= 0.80",
        "D4_e21_override_p70": "E20 default, override to E21 if p21 >= 0.70",
        "D5_e20_override_p10": "E21 default, override to E20 if p20 <= 0.10",
        "D6_e20_override_p20": "E21 default, override to E20 if p20 <= 0.20",
        "D7_disagree_max_conf": "If agree use E21, if disagree use max confidence",
        "D8_disagree_e20": "If agree use E21, if disagree use conservative E20",
        "D9_disagree_e21": "If agree use E20, if disagree use sensitive E21",
        "E_logistic_regression": "Logistic Regression Router on 9 features",
        "F1_decision_tree_d2": "Decision Tree Router (max_depth=2)",
        "F2_decision_tree_d3": "Decision Tree Router (max_depth=3)",
    }

    dev_results = []
    for cid in candidate_ids:
        p_cand, routes = run_candidate(cid, p_e20_dev, p_e21_dev, X_dev)
        m = compute_metrics(y_dev, p_cand, threshold=0.50)
        pct_e20 = sum(1 for r in routes if r == "E20") / len(routes) * 100.0
        pct_e21 = sum(1 for r in routes if r == "E21") / len(routes) * 100.0
        pct_blend = sum(1 for r in routes if r == "Blend") / len(routes) * 100.0

        dev_results.append({
            "candidate_id": cid,
            "description": candidate_descriptions[cid],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "precision": m["precision"],
            "ai_recall": m["recall"],
            "real_fpr": m["fpr"],
            "ai_fnr": m["fnr"],
            "tp": m["tp"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "pct_routed_e20": pct_e20,
            "pct_routed_e21": pct_e21,
            "pct_routed_blend": pct_blend,
        })

    # Sort strictly according to selection protocol:
    # 1. Highest F1
    # 2. Lowest Real FPR
    # 3. Highest ROC-AUC
    # 4. Highest AI Recall
    dev_results.sort(
        key=lambda x: (
            -x["f1"],
            x["real_fpr"],
            -x["roc_auc"],
            -x["ai_recall"]
        )
    )

    for rank, res in enumerate(dev_results, start=1):
        res["rank"] = rank

    # Write selection_on_e21_dev.csv
    dev_csv_path = OUT_DIR / "e23_dev_selection.csv"
    fieldnames = [
        "rank", "candidate_id", "description", "accuracy", "f1", "roc_auc", "pr_auc",
        "precision", "ai_recall", "real_fpr", "ai_fnr", "tp", "tn", "fp", "fn",
        "pct_routed_e20", "pct_routed_e21", "pct_routed_blend"
    ]
    with open(dev_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in dev_results:
            writer.writerow(res)

    print(f"\nSaved E21 Dev Selection Results to {dev_csv_path}")
    print("\n--- E21 DEV SELECTION RANKINGS ---")
    print(f"{'Rank':<4} | {'Candidate ID':<22} | {'Acc':<7} | {'F1':<7} | {'ROC-AUC':<8} | {'Recall':<7} | {'FPR':<7} | {'%E20':<5} | {'%E21':<5}")
    print("-" * 90)
    for res in dev_results:
        print(f"{res['rank']:<4} | {res['candidate_id']:<22} | {res['accuracy']*100:6.2f}% | {res['f1']:7.4f} | {res['roc_auc']:8.4f} | {res['ai_recall']*100:6.2f}% | {res['real_fpr']*100:6.2f}% | {res['pct_routed_e20']:5.1f} | {res['pct_routed_e21']:5.1f}")

    winner = dev_results[0]
    WINNING_CANDIDATE_ID = winner["candidate_id"]
    print(f"\n>>> WINNING ROUTER SELECTED & FROZEN: {WINNING_CANDIDATE_ID} (Rank 1, F1 = {winner['f1']:.4f}, Real FPR = {winner['real_fpr']*100:.2f}%) <<<")

    # Also log the top pure-routing candidate if Rank 1 is the E22 blend
    top_pure_router = next(r for r in dev_results if not r["candidate_id"].startswith("C_") and not r["candidate_id"].startswith("A_") and not r["candidate_id"].startswith("B_"))
    print(f">>> TOP PURE ROUTER: {top_pure_router['candidate_id']} (Rank {top_pure_router['rank']}, F1 = {top_pure_router['f1']:.4f}, Real FPR = {top_pure_router['real_fpr']*100:.2f}%) <<<")

    # 3. Phase B: Sealed External Benchmark Evaluation
    print("\n==================================================")
    print("PHASE B: EVALUATING SEALED EXTERNAL BENCHMARKS")
    print("==================================================")

    benchmark_configs = [
        {
            "name": "E14 Clean External",
            "file": E22_PRED_DIR / "e14_clean_e22_predictions.csv",
            "benchmark_key": "e14_clean",
            "n_expected": 200
        },
        {
            "name": "E14 Degraded External",
            "file": E22_PRED_DIR / "e14_degraded_e22_predictions.csv",
            "benchmark_key": "e14_degraded",
            "n_expected": 600
        },
        {
            "name": "E19 Smartphone Benchmark",
            "file": E22_PRED_DIR / "e19_smartphone_benchmark_e22_predictions.csv",
            "benchmark_key": "e19_smartphone",
            "n_expected": 200
        },
        {
            "name": "E19 WhatsApp Subset",
            "file": E22_PRED_DIR / "e19_whatsapp_subset_e22_predictions.csv",
            "benchmark_key": "e19_whatsapp",
            "n_expected": 60
        },
        {
            "name": "WhatsApp N=67 Benchmark",
            "file": E22_PRED_DIR / "whatsapp_n67_benchmark_e22_predictions.csv",
            "benchmark_key": "whatsapp_n67",
            "n_expected": 67
        },
    ]

    all_external_metrics = []
    all_decisions = []
    all_transitions = []
    all_subgroups = []

    for bcfg in benchmark_configs:
        bname = bcfg["name"]
        bkey = bcfg["benchmark_key"]
        print(f"\nProcessing {bname}...")

        with open(bcfg["file"], "r", encoding="utf-8") as f:
            b_rows = list(csv.DictReader(f))
        assert len(b_rows) == bcfg["n_expected"], f"Expected {bcfg['n_expected']} rows, got {len(b_rows)}"

        y_true = np.array([1 if r["label"] == "AI" else 0 for r in b_rows])
        p_e6c = np.array([float(r["prob_e6c"]) for r in b_rows])
        p_e20 = np.array([float(r["prob_e20"]) for r in b_rows])
        p_e21 = np.array([float(r["prob_e21"]) for r in b_rows])
        p_e22 = np.array([float(r["prob_e22"]) for r in b_rows])

        X_ext = build_features(p_e20, p_e21)

        # Run E23 Router (both winning candidate and top pure router for deep analysis)
        p_e23_win, routes_win = run_candidate(WINNING_CANDIDATE_ID, p_e20, p_e21, X_ext)
        p_e23_pure, routes_pure = run_candidate(top_pure_router["candidate_id"], p_e20, p_e21, X_ext)

        # Models to evaluate
        models = [
            ("E6-C Baseline", p_e6c, ["N/A"] * len(p_e6c)),
            ("E20 Baseline", p_e20, ["E20"] * len(p_e20)),
            ("E21 Baseline", p_e21, ["E21"] * len(p_e21)),
            ("E22 Ensemble", p_e22, ["Blend"] * len(p_e22)),
            (f"E23 Router ({WINNING_CANDIDATE_ID})", p_e23_win, routes_win),
            (f"E23 Pure Router ({top_pure_router['candidate_id']})", p_e23_pure, routes_pure),
        ]

        for mname, p_model, r_decisions in models:
            m = compute_metrics(y_true, p_model, threshold=0.50)
            pct_20 = sum(1 for r in r_decisions if r == "E20") / len(r_decisions) * 100.0 if "E20" in r_decisions else 0.0
            pct_21 = sum(1 for r in r_decisions if r == "E21") / len(r_decisions) * 100.0 if "E21" in r_decisions else 0.0

            all_external_metrics.append({
                "benchmark": bname,
                "n_samples": len(y_true),
                "model": mname,
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "precision": m["precision"],
                "ai_recall": m["recall"],
                "real_fpr": m["fpr"],
                "ai_fnr": m["fnr"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
                "pct_routed_e20": pct_20,
                "pct_routed_e21": pct_21
            })

        # Process per-image decisions and transitions for winning E23 and top pure router
        for idx, row in enumerate(b_rows):
            y_i = y_true[idx]
            p20_i = p_e20[idx]
            p21_i = p_e21[idx]
            p22_i = p_e22[idx]
            p23_i = p_e23_win[idx]
            p23_pure_i = p_e23_pure[idx]

            pred_20 = 1 if p20_i >= 0.5 else 0
            pred_21 = 1 if p21_i >= 0.5 else 0
            pred_22 = 1 if p22_i >= 0.5 else 0
            pred_23 = 1 if p23_i >= 0.5 else 0
            pred_23_pure = 1 if p23_pure_i >= 0.5 else 0

            corr_20 = (pred_20 == y_i)
            corr_21 = (pred_21 == y_i)
            corr_22 = (pred_22 == y_i)
            corr_23 = (pred_23 == y_i)
            corr_23_pure = (pred_23_pure == y_i)

            # Complementarity category between E20 and E21
            if corr_20 and corr_21:
                comp_type = "both_correct"
            elif corr_20 and not corr_21:
                comp_type = "E20_correct__E21_wrong"
            elif not corr_20 and corr_21:
                comp_type = "E20_wrong__E21_correct"
            else:
                comp_type = "both_wrong"

            # Transition for E23 vs E20
            if corr_20 and corr_23:
                trans_20_23 = "E20_correct__E23_correct"
            elif corr_20 and not corr_23:
                trans_20_23 = "E20_correct__E23_wrong"
            elif not corr_20 and corr_23:
                trans_20_23 = "E20_wrong__E23_correct"
            else:
                trans_20_23 = "E20_wrong__E23_wrong"

            # Transition for E23 vs E21
            if corr_21 and corr_23:
                trans_21_23 = "E21_correct__E23_correct"
            elif corr_21 and not corr_23:
                trans_21_23 = "E21_correct__E23_wrong"
            elif not corr_21 and corr_23:
                trans_21_23 = "E21_wrong__E23_correct"
            else:
                trans_21_23 = "E21_wrong__E23_wrong"

            all_decisions.append({
                "benchmark": bname,
                "image_id": row["image_id"],
                "label": row["label"],
                "prob_e6c": row["prob_e6c"],
                "prob_e20": f"{p20_i:.6f}",
                "pred_e20": pred_20,
                "prob_e21": f"{p21_i:.6f}",
                "pred_e21": pred_21,
                "prob_e22": f"{p22_i:.6f}",
                "pred_e22": pred_22,
                "prob_e23": f"{p23_i:.6f}",
                "pred_e23": pred_23,
                "route_decision_win": routes_win[idx],
                "prob_e23_pure": f"{p23_pure_i:.6f}",
                "pred_e23_pure": pred_23_pure,
                "route_decision_pure": routes_pure[idx],
                "complementarity_type": comp_type,
                "e20_vs_e23_transition": trans_20_23,
                "e21_vs_e23_transition": trans_21_23,
                "generator": row.get("generator", "unknown"),
                "device_family": row.get("device_family", "unknown"),
                "filepath": row.get("filepath", "")
            })

            all_transitions.append({
                "benchmark": bname,
                "image_id": row["image_id"],
                "label": row["label"],
                "complementarity_type": comp_type,
                "route_decision": routes_win[idx],
                "route_decision_pure": routes_pure[idx],
                "e20_vs_e23_transition": trans_20_23,
                "e21_vs_e23_transition": trans_21_23,
                "generator": row.get("generator", "unknown"),
                "device_family": row.get("device_family", "unknown")
            })

    # Write external results CSV
    ext_csv_path = OUT_DIR / "e23_external_results.csv"
    with open(ext_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_external_metrics[0].keys()))
        writer.writeheader()
        for row in all_external_metrics:
            writer.writerow(row)
    print(f"Saved external results to {ext_csv_path}")

    # Write routing decisions CSV
    dec_csv_path = OUT_DIR / "e23_routing_decisions.csv"
    with open(dec_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_decisions[0].keys()))
        writer.writeheader()
        for row in all_decisions:
            writer.writerow(row)
    print(f"Saved routing decisions to {dec_csv_path}")

    # Write error transitions CSV
    trans_csv_path = OUT_DIR / "e23_error_transitions.csv"
    with open(trans_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_transitions[0].keys()))
        writer.writeheader()
        for row in all_transitions:
            writer.writerow(row)
    print(f"Saved error transitions to {trans_csv_path}")

    # Generator Breakdown for E14 Clean
    e14_clean_dec = [d for d in all_decisions if d["benchmark"] == "E14 Clean External" and d["label"] == "AI"]
    gen_names = sorted(list(set(d["generator"] for d in e14_clean_dec)))
    gen_stats = []
    for g in gen_names:
        sub = [d for d in e14_clean_dec if d["generator"] == g]
        n_g = len(sub)
        rec_e6c = sum(1 for d in sub if float(d["prob_e6c"]) >= 0.5) / n_g * 100.0
        rec_e20 = sum(1 for d in sub if d["pred_e20"] == 1) / n_g * 100.0
        rec_e21 = sum(1 for d in sub if d["pred_e21"] == 1) / n_g * 100.0
        rec_e22 = sum(1 for d in sub if d["pred_e22"] == 1) / n_g * 100.0
        rec_e23 = sum(1 for d in sub if d["pred_e23"] == 1) / n_g * 100.0
        rec_e23_pure = sum(1 for d in sub if d["pred_e23_pure"] == 1) / n_g * 100.0
        pct_e20_routed = sum(1 for d in sub if d["route_decision_pure"] == "E20") / n_g * 100.0
        pct_e21_routed = sum(1 for d in sub if d["route_decision_pure"] == "E21") / n_g * 100.0

        gen_stats.append({
            "generator": g,
            "n_samples": n_g,
            "recall_e6c": rec_e6c,
            "recall_e20": rec_e20,
            "recall_e21": rec_e21,
            "recall_e22": rec_e22,
            "recall_e23_win": rec_e23,
            "recall_e23_pure": rec_e23_pure,
            "pct_routed_e20": pct_e20_routed,
            "pct_routed_e21": pct_e21_routed
        })

    gen_csv_path = SUBGROUP_DIR / "e14_generator_breakdown.csv"
    with open(gen_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(gen_stats[0].keys()))
        writer.writeheader()
        for row in gen_stats:
            writer.writerow(row)
    print(f"Saved generator breakdown to {gen_csv_path}")

    # Device Family Breakdown for E19
    e19_reals = [d for d in all_decisions if d["benchmark"] == "E19 Smartphone Benchmark" and d["label"] == "Real"]
    dev_names = sorted(list(set(d["device_family"] for d in e19_reals)))
    dev_stats = []
    for dev in dev_names:
        sub = [d for d in e19_reals if d["device_family"] == dev]
        n_d = len(sub)
        fpr_e6c = sum(1 for d in sub if float(d["prob_e6c"]) >= 0.5) / n_d * 100.0
        fpr_e20 = sum(1 for d in sub if d["pred_e20"] == 1) / n_d * 100.0
        fpr_e21 = sum(1 for d in sub if d["pred_e21"] == 1) / n_d * 100.0
        fpr_e22 = sum(1 for d in sub if d["pred_e22"] == 1) / n_d * 100.0
        fpr_e23 = sum(1 for d in sub if d["pred_e23"] == 1) / n_d * 100.0
        fpr_e23_pure = sum(1 for d in sub if d["pred_e23_pure"] == 1) / n_d * 100.0
        pct_e20_routed = sum(1 for d in sub if d["route_decision_pure"] == "E20") / n_d * 100.0
        pct_e21_routed = sum(1 for d in sub if d["route_decision_pure"] == "E21") / n_d * 100.0

        dev_stats.append({
            "device_family": dev,
            "n_samples": n_d,
            "fpr_e6c": fpr_e6c,
            "fpr_e20": fpr_e20,
            "fpr_e21": fpr_e21,
            "fpr_e22": fpr_e22,
            "fpr_e23_win": fpr_e23,
            "fpr_e23_pure": fpr_e23_pure,
            "pct_routed_e20": pct_e20_routed,
            "pct_routed_e21": pct_e21_routed
        })

    dev_csv_path = SUBGROUP_DIR / "e19_device_breakdown.csv"
    with open(dev_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dev_stats[0].keys()))
        writer.writeheader()
        for row in dev_stats:
            writer.writerow(row)
    print(f"Saved device breakdown to {dev_csv_path}")

    # WhatsApp Specific Breakdown
    wa_items = [d for d in all_decisions if "WhatsApp" in d["benchmark"]]
    wa_stats = []
    for b in ["E19 WhatsApp Subset", "WhatsApp N=67 Benchmark"]:
        sub = [d for d in wa_items if d["benchmark"] == b]
        sub_real = [d for d in sub if d["label"] == "Real"]
        sub_ai = [d for d in sub if d["label"] == "AI"]
        n_tot = len(sub)
        wa_stats.append({
            "benchmark": b,
            "n_total": n_tot,
            "n_real": len(sub_real),
            "n_ai": len(sub_ai),
            "ai_recall_e20": sum(1 for d in sub_ai if d["pred_e20"] == 1) / len(sub_ai) * 100.0,
            "ai_recall_e21": sum(1 for d in sub_ai if d["pred_e21"] == 1) / len(sub_ai) * 100.0,
            "ai_recall_e22": sum(1 for d in sub_ai if d["pred_e22"] == 1) / len(sub_ai) * 100.0,
            "ai_recall_e23_win": sum(1 for d in sub_ai if d["pred_e23"] == 1) / len(sub_ai) * 100.0,
            "ai_recall_e23_pure": sum(1 for d in sub_ai if d["pred_e23_pure"] == 1) / len(sub_ai) * 100.0,
            "real_fpr_e20": sum(1 for d in sub_real if d["pred_e20"] == 1) / len(sub_real) * 100.0,
            "real_fpr_e21": sum(1 for d in sub_real if d["pred_e21"] == 1) / len(sub_real) * 100.0,
            "real_fpr_e22": sum(1 for d in sub_real if d["pred_e22"] == 1) / len(sub_real) * 100.0,
            "real_fpr_e23_win": sum(1 for d in sub_real if d["pred_e23"] == 1) / len(sub_real) * 100.0,
            "real_fpr_e23_pure": sum(1 for d in sub_real if d["pred_e23_pure"] == 1) / len(sub_real) * 100.0,
        })
    wa_csv_path = SUBGROUP_DIR / "whatsapp_breakdown.csv"
    with open(wa_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wa_stats[0].keys()))
        writer.writeheader()
        for row in wa_stats:
            writer.writerow(row)
    print(f"Saved WhatsApp breakdown to {wa_csv_path}")

    print("\n==================================================")
    print("EXPERIMENT 23 EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
