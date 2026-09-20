"""
Final Frozen External Evaluation of E20 against E6-C Baseline.

Models:
1. E6-C Baseline: experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt
2. E20 Model: experiments/e20_training/checkpoints/e20_best_model.pt

Datasets Evaluated:
A. E14 Clean External Test (N=200)
B. E14 Degraded External Test (N=600)
C. E19 Smartphone Benchmark (N=200)
D. E19 Controlled WhatsApp-style Subset (N=60)
E. WhatsApp Robustness Benchmark (N=67)

Fixed Threshold = 0.50 for BOTH models.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model

# Paths
E6C_CHECKPOINT = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
E20_CHECKPOINT = PROJECT_ROOT / "experiments/e20_training/checkpoints/e20_best_model.pt"

E14_CLEAN_CSV = PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_manifest.csv"
E14_DEG_CSV = PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_degraded_manifest.csv"
E19_CSV = PROJECT_ROOT / "data/e19_smartphone_benchmark/manifests/e19_test_manifest.csv"
WHATSAPP_DIR = PROJECT_ROOT / "data/whatsapp_robustness_test"

OUT_DIR = PROJECT_ROOT / "experiments/e20_training/external_evaluation"
PRED_DIR = OUT_DIR / "predictions"
SUBGROUP_DIR = OUT_DIR / "subgroups"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_5_crops(img: Image.Image) -> list:
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    return [
        img.resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR),
    ]

class EvalDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        fp = Path(item["filepath"])
        if not fp.is_absolute():
            fp = PROJECT_ROOT / fp
        label_str = item.get("label", "Real")
        label = 1.0 if label_str == "AI" else 0.0
        try:
            with Image.open(fp) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            stacked = torch.stack([self.to_tensor(c) for c in crops], dim=0)
            return stacked, torch.tensor(label, dtype=torch.float32), idx
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            dummy = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
            return dummy, torch.tensor(label, dtype=torch.float32), idx

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_total = len(y_true)
    n_pos = tp + fn
    n_neg = tn + fp

    acc = (tp + tn) / n_total if n_total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    fnr = fn / n_pos if n_pos > 0 else 0.0

    # ROC-AUC
    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
    else:
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

    # PR-AUC
    if n_pos == 0:
        pr_auc = 0.0
    else:
        sorted_indices = np.argsort(-y_prob)
        sorted_labels = y_true[sorted_indices]
        tps = np.cumsum(sorted_labels == 1)
        fps = np.cumsum(sorted_labels == 0)
        precisions = tps / (tps + fps)
        recalls = tps / n_pos
        if hasattr(np, 'trapezoid'):
            pr_auc = float(np.trapezoid(precisions, recalls))
        elif hasattr(np, 'trapz'):
            pr_auc = float(np.trapz(precisions, recalls))
        else:
            pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * (precisions[1:] + precisions[:-1]) / 2.0))

    # Probability summary for Real and AI
    real_probs = y_prob[y_true == 0]
    ai_probs = y_prob[y_true == 1]

    mean_prob_real = float(np.mean(real_probs)) if len(real_probs) > 0 else 0.0
    median_prob_real = float(np.median(real_probs)) if len(real_probs) > 0 else 0.0
    mean_prob_ai = float(np.mean(ai_probs)) if len(ai_probs) > 0 else 0.0
    median_prob_ai = float(np.median(ai_probs)) if len(ai_probs) > 0 else 0.0

    return {
        "n": n_total,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_prob_real": mean_prob_real,
        "median_prob_real": median_prob_real,
        "mean_prob_ai": mean_prob_ai,
        "median_prob_ai": median_prob_ai
    }

def run_model_inference(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    probs_dict = {}
    with torch.no_grad():
        for x_views, _, indices in loader:
            x_views = x_views.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(x_views).squeeze(-1)
                p = torch.sigmoid(logits)
            for idx_val, prob_val in zip(indices.tolist(), p.cpu().tolist()):
                probs_dict[idx_val] = prob_val
    return np.array([probs_dict[i] for i in range(len(probs_dict))])

def classify_transition(lbl: int, p_e6c: float, p_e20: float, thresh: float = 0.50) -> str:
    c_e6c = (p_e6c >= thresh) == (lbl == 1)
    c_e20 = (p_e20 >= thresh) == (lbl == 1)
    if c_e6c and c_e20:
        return "E6C_correct__E20_correct"
    elif c_e6c and not c_e20:
        return "E6C_correct__E20_wrong"
    elif not c_e6c and c_e20:
        return "E6C_wrong__E20_correct"
    else:
        return "E6C_wrong__E20_wrong"

def evaluate_subgroups(items: List[Dict], y_true: np.ndarray, y_prob_e6c: np.ndarray, y_prob_e20: np.ndarray, group_key: str) -> List[Dict]:
    unique_groups = sorted(list(set(str(it.get(group_key, "unknown")) for it in items)))
    rows = []
    for g in unique_groups:
        idxs = [i for i, it in enumerate(items) if str(it.get(group_key, "unknown")) == g]
        if len(idxs) == 0:
            continue
        sub_y = y_true[idxs]
        sub_e6c = y_prob_e6c[idxs]
        sub_e20 = y_prob_e20[idxs]

        m_e6c = compute_metrics(sub_y, sub_e6c)
        m_e20 = compute_metrics(sub_y, sub_e20)

        rows.append({
            "group_field": group_key,
            "group_value": g,
            "n": len(idxs),
            "n_real": int(np.sum(sub_y == 0)),
            "n_ai": int(np.sum(sub_y == 1)),
            "e6c_acc": m_e6c["accuracy"],
            "e20_acc": m_e20["accuracy"],
            "acc_delta": m_e20["accuracy"] - m_e6c["accuracy"],
            "e6c_f1": m_e6c["f1"],
            "e20_f1": m_e20["f1"],
            "f1_delta": m_e20["f1"] - m_e6c["f1"],
            "e6c_recall": m_e6c["recall"] if np.sum(sub_y == 1) > 0 else None,
            "e20_recall": m_e20["recall"] if np.sum(sub_y == 1) > 0 else None,
            "e6c_fpr": m_e6c["fpr"] if np.sum(sub_y == 0) > 0 else None,
            "e20_fpr": m_e20["fpr"] if np.sum(sub_y == 0) > 0 else None,
            "e6c_auc": m_e6c["roc_auc"] if (np.sum(sub_y == 0) > 0 and np.sum(sub_y == 1) > 0) else None,
            "e20_auc": m_e20["roc_auc"] if (np.sum(sub_y == 0) > 0 and np.sum(sub_y == 1) > 0) else None,
        })
    return rows

def load_whatsapp_n67() -> List[Dict]:
    ai_dir = WHATSAPP_DIR / "ai"
    real_dir = WHATSAPP_DIR / "real"
    items = []
    for fp in sorted(list(real_dir.glob("*.*"))):
        items.append({
            "image_id": fp.stem,
            "filepath": str(fp),
            "filename": fp.name,
            "label": "Real",
            "generator": "none",
            "device_family": "smartphone_unknown",
            "degradation_type": "whatsapp_transcode"
        })
    for fp in sorted(list(ai_dir.glob("*.*"))):
        items.append({
            "image_id": fp.stem,
            "filepath": str(fp),
            "filename": fp.name,
            "label": "AI",
            "generator": "generator_unknown",
            "device_family": "none",
            "degradation_type": "whatsapp_transcode"
        })
    return items

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SUBGROUP_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STARTING FROZEN EXTERNAL EVALUATION: E20 vs E6-C")
    print("==================================================")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load Checkpoints
    print("\n[1/3] Loading models strictly...")
    model_e6c = MultiViewE5Model()
    ckpt_e6c = torch.load(E6C_CHECKPOINT, map_location="cpu")
    model_e6c.load_state_dict(ckpt_e6c["model_state_dict"], strict=True)
    model_e6c.to(DEVICE)
    model_e6c.eval()
    print("  [*] E6-C Checkpoint loaded successfully from:", E6C_CHECKPOINT)

    model_e20 = MultiViewE5Model()
    ckpt_e20 = torch.load(E20_CHECKPOINT, map_location="cpu")
    model_e20.load_state_dict(ckpt_e20["model_state_dict"], strict=True)
    model_e20.to(DEVICE)
    model_e20.eval()
    print("  [*] E20 Checkpoint loaded successfully from:", E20_CHECKPOINT)

    # 2. Prepare Datasets
    print("\n[2/3] Preparing test datasets...")
    # A. E14 Clean
    with open(E14_CLEAN_CSV, "r", encoding="utf-8") as f:
        e14_clean_items = list(csv.DictReader(f))
    for it in e14_clean_items:
        it.setdefault("degradation_type", "clean")
        it.setdefault("device_family", it.get("source_dataset", "unknown"))

    # B. E14 Degraded
    with open(E14_DEG_CSV, "r", encoding="utf-8") as f:
        e14_deg_items = list(csv.DictReader(f))
    for it in e14_deg_items:
        it.setdefault("image_id", it.get("variant_id"))
        it.setdefault("device_family", "unknown")

    # C. E19 Smartphone Benchmark (All N=200)
    with open(E19_CSV, "r", encoding="utf-8") as f:
        e19_items = list(csv.DictReader(f))
    for it in e19_items:
        it.setdefault("degradation_type", "whatsapp_simulation" if it.get("whatsapp_processed") == "True" else "clean")

    # D. E19 Controlled WhatsApp-style Subset (N=60)
    e19_wa_items = [it for it in e19_items if it.get("whatsapp_processed") == "True"]

    # E. WhatsApp N=67 Benchmark
    wa67_items = load_whatsapp_n67()

    datasets_to_eval = [
        ("E14_Clean", e14_clean_items, 200),
        ("E14_Degraded", e14_deg_items, 600),
        ("E19_Smartphone_Benchmark", e19_items, 200),
        ("E19_WhatsApp_Subset", e19_wa_items, 60),
        ("WhatsApp_N67_Benchmark", wa67_items, 67)
    ]

    for name, items, exp_n in datasets_to_eval:
        print(f"  - {name}: {len(items)} samples (Expected: {exp_n})")
        assert len(items) == exp_n, f"Dataset size mismatch for {name}: {len(items)} != {exp_n}"

    # 3. Run Inference and Evaluation
    print("\n[3/3] Running inference and metrics computation...")
    overall_results = []
    all_subgroups = []
    error_transitions_summary = []
    representative_samples = []

    for name, items, exp_n in datasets_to_eval:
        print(f"\n--- Evaluating {name} (N={len(items)}) ---")
        dataset = EvalDataset(items)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

        t0 = time.time()
        p_e6c = run_model_inference(model_e6c, loader)
        p_e20 = run_model_inference(model_e20, loader)
        dt = time.time() - t0
        print(f"  Inference completed in {dt:.2f}s ({len(items)*2 / dt:.1f} inferences/sec)")

        y_true = np.array([1 if it.get("label") == "AI" else 0 for it in items])

        m_e6c = compute_metrics(y_true, p_e6c)
        m_e20 = compute_metrics(y_true, p_e20)

        # Print quick side-by-side
        print(f"  E6-C: AUC={m_e6c['roc_auc']:.4f}, Acc={m_e6c['accuracy']*100:.2f}%, F1={m_e6c['f1']:.4f}, FPR={m_e6c['fpr']*100:.2f}%, Recall={m_e6c['recall']*100:.2f}%")
        print(f"  E20:  AUC={m_e20['roc_auc']:.4f}, Acc={m_e20['accuracy']*100:.2f}%, F1={m_e20['f1']:.4f}, FPR={m_e20['fpr']*100:.2f}%, Recall={m_e20['recall']*100:.2f}%")
        print(f"  Delta: Acc={ (m_e20['accuracy']-m_e6c['accuracy'])*100:+.2f}%, F1={m_e20['f1']-m_e6c['f1']:+.4f}, FPR={(m_e20['fpr']-m_e6c['fpr'])*100:+.2f}%")

        overall_results.append({
            "dataset": name, "model": "E6-C", **m_e6c
        })
        overall_results.append({
            "dataset": name, "model": "E20", **m_e20
        })

        # Transitions
        trans_counts = {
            "E6C_correct__E20_correct": 0,
            "E6C_correct__E20_wrong": 0,
            "E6C_wrong__E20_correct": 0,
            "E6C_wrong__E20_wrong": 0
        }
        pred_records = []
        for idx in range(len(items)):
            it = items[idx]
            lbl = y_true[idx]
            prob1 = float(p_e6c[idx])
            prob2 = float(p_e20[idx])
            trans = classify_transition(lbl, prob1, prob2, thresh=0.50)
            trans_counts[trans] += 1

            rec = {
                "dataset": name,
                "image_id": it.get("image_id", Path(it["filepath"]).stem),
                "filepath": it.get("filepath"),
                "label": it.get("label"),
                "generator": it.get("generator", "none"),
                "device_family": it.get("device_family", "none"),
                "degradation_type": it.get("degradation_type", "clean"),
                "prob_e6c": prob1,
                "pred_e6c": int(prob1 >= 0.50),
                "correct_e6c": int((prob1 >= 0.50) == (lbl == 1)),
                "prob_e20": prob2,
                "pred_e20": int(prob2 >= 0.50),
                "correct_e20": int((prob2 >= 0.50) == (lbl == 1)),
                "transition": trans
            }
            pred_records.append(rec)

            # Sample representative errors for report
            if trans in ("E6C_wrong__E20_correct", "E6C_correct__E20_wrong", "E6C_wrong__E20_wrong"):
                representative_samples.append(rec)

        # Save per-dataset prediction CSV
        pred_csv = PRED_DIR / f"predictions_{name.lower()}.csv"
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pred_records[0].keys()))
            writer.writeheader()
            writer.writerows(pred_records)

        error_transitions_summary.append({
            "dataset": name,
            "total": len(items),
            **trans_counts
        })

        # Subgroup evaluations
        for grp_key in ["generator", "device_family", "degradation_type"]:
            sub_res = evaluate_subgroups(items, y_true, p_e6c, p_e20, grp_key)
            for sr in sub_res:
                sr["dataset"] = name
                all_subgroups.append(sr)

    # 4. Save Aggregated Results
    results_csv = OUT_DIR / "e20_vs_e6c_external_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall_results[0].keys()))
        writer.writeheader()
        writer.writerows(overall_results)
    print(f"\nWrote results summary: {results_csv}")

    subgroups_csv = SUBGROUP_DIR / "e20_vs_e6c_subgroups.csv"
    with open(subgroups_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_subgroups[0].keys()))
        writer.writeheader()
        writer.writerows(all_subgroups)
    print(f"Wrote subgroups summary: {subgroups_csv}")

    trans_csv = OUT_DIR / "e20_vs_e6c_error_transitions.csv"
    with open(trans_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_transitions_summary[0].keys()))
        writer.writeheader()
        writer.writerows(error_transitions_summary)
    print(f"Wrote transition summary: {trans_csv}")

    rep_csv = OUT_DIR / "e20_vs_e6c_representative_errors.csv"
    with open(rep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(representative_samples[0].keys()))
        writer.writeheader()
        writer.writerows(representative_samples)
    print(f"Wrote representative errors: {rep_csv}")

    print("\nEVALUATION COMPLETE. Ready to compile report.")

if __name__ == "__main__":
    main()
