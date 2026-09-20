"""
Experiment 22 (E22): Inference-Only Complementary Ensemble Study.

Protocol & Constraints:
- NO training. NO fine-tuning.
- NO checkpoint modifications.
- NO dataset modifications.
- NO production modifications.
- NO threshold tuning (Fixed threshold = 0.50).
- Rule selection strictly conducted on E21 DEV split ONLY.
- Sealed external benchmarks remain untouched until rule is frozen.
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
E21_CHECKPOINT = PROJECT_ROOT / "experiments/e21_targeted_data/checkpoints/e21_best_model.pt"

E21_DEV_MANIFEST = PROJECT_ROOT / "data/e21_targeted_data/manifests/e21_dev_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data/e21_targeted_data"

E14_CLEAN_CSV = PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_manifest.csv"
E14_DEG_CSV = PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_degraded_manifest.csv"
E19_CSV = PROJECT_ROOT / "data/e19_smartphone_benchmark/manifests/e19_test_manifest.csv"
WHATSAPP_DIR = PROJECT_ROOT / "data/whatsapp_robustness_test"

OUT_DIR = PROJECT_ROOT / "experiments/e22_ensemble_study"
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
    def __init__(self, items: List[Dict], root: Path = None):
        self.items = items
        self.root = root
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        fp = Path(item["filepath"])
        if not fp.is_absolute():
            if self.root:
                fp = self.root / fp
            else:
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

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50, y_pred_override: np.ndarray = None) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    if y_pred_override is not None:
        y_pred = np.asarray(y_pred_override, dtype=int)
    else:
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
    roc_auc = 0.5
    pr_auc = 0.0
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if n_pos > 0 and n_neg > 0:
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
        if n_pos > 0:
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
        "fn": fn
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
    print("STARTING EXPERIMENT 22: ENSEMBLE COMPLEMENTARITY STUDY")
    print("==================================================")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load Checkpoints
    print("\n[1/5] Loading frozen checkpoints strictly...")
    model_e20 = MultiViewE5Model()
    ckpt_e20 = torch.load(E20_CHECKPOINT, map_location="cpu")
    model_e20.load_state_dict(ckpt_e20["model_state_dict"], strict=True)
    model_e20.to(DEVICE)
    model_e20.eval()
    print("  [*] E20 Checkpoint loaded strictly from:", E20_CHECKPOINT)

    model_e21 = MultiViewE5Model()
    ckpt_e21 = torch.load(E21_CHECKPOINT, map_location="cpu")
    model_e21.load_state_dict(ckpt_e21["model_state_dict"], strict=True)
    model_e21.to(DEVICE)
    model_e21.eval()
    print("  [*] E21 Checkpoint loaded strictly from:", E21_CHECKPOINT)

    model_e6c = MultiViewE5Model()
    ckpt_e6c = torch.load(E6C_CHECKPOINT, map_location="cpu")
    model_e6c.load_state_dict(ckpt_e6c["model_state_dict"], strict=True)
    model_e6c.to(DEVICE)
    model_e6c.eval()
    print("  [*] E6-C Baseline loaded strictly from:", E6C_CHECKPOINT)

    # 2. Phase A: Rule Selection on E21 Dev ONLY
    print("\n[2/5] Running Phase A: Rule Selection strictly on E21 Dev Split (N=812)...")
    with open(E21_DEV_MANIFEST, "r", encoding="utf-8") as f:
        dev_records = list(csv.DictReader(f))
    assert len(dev_records) == 812, f"Expected 812 records in E21 Dev, got {len(dev_records)}"

    dev_dataset = EvalDataset(dev_records, root=DATA_ROOT)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    print("  Computing E20 probabilities on E21 Dev...")
    p_e20_dev = run_model_inference(model_e20, dev_loader)

    print("  Computing E21 probabilities on E21 Dev...")
    p_e21_dev = run_model_inference(model_e21, dev_loader)

    y_dev_true = np.array([1 if r["label"] == "AI" else 0 for r in dev_records])

    # Candidate Ensembles:
    candidates = []

    # 1-11: Linear combinations
    weights = [
        ("1.0_E20 + 0.0_E21 (E20 only)", 1.0, 0.0),
        ("0.0_E20 + 1.0_E21 (E21 only)", 0.0, 1.0),
        ("0.9_E20 + 0.1_E21", 0.9, 0.1),
        ("0.8_E20 + 0.2_E21", 0.8, 0.2),
        ("0.7_E20 + 0.3_E21", 0.7, 0.3),
        ("0.6_E20 + 0.4_E21", 0.6, 0.4),
        ("0.5_E20 + 0.5_E21", 0.5, 0.5),
        ("0.4_E20 + 0.6_E21", 0.4, 0.6),
        ("0.3_E20 + 0.7_E21", 0.3, 0.7),
        ("0.2_E20 + 0.8_E21", 0.2, 0.8),
        ("0.1_E20 + 0.9_E21", 0.1, 0.9),
    ]

    for name, w20, w21 in weights:
        p_ens = w20 * p_e20_dev + w21 * p_e21_dev
        m = compute_metrics(y_dev_true, p_ens, threshold=0.50)
        candidates.append({
            "candidate_type": "linear_combination",
            "name": name,
            "w_e20": w20,
            "w_e21": w21,
            "rule_description": f"p_ens = {w20}*p_e20 + {w21}*p_e21; predict AI if p_ens >= 0.50",
            **m
        })

    # Optional Simple Gated Rules:
    # A: If E20 >= 0.50 OR E21 >= 0.70 -> AI, else Real
    pred_gate_a = ((p_e20_dev >= 0.50) | (p_e21_dev >= 0.70)).astype(int)
    p_gate_a = np.maximum(p_e20_dev, p_e21_dev * (0.50 / 0.70))
    m_gate_a = compute_metrics(y_dev_true, p_gate_a, threshold=0.50, y_pred_override=pred_gate_a)
    candidates.append({
        "candidate_type": "gated_rule",
        "name": "Gate_A",
        "w_e20": None, "w_e21": None,
        "rule_description": "If E20 >= 0.50 OR E21 >= 0.70 -> AI, else Real",
        **m_gate_a
    })

    # B: If E20 >= 0.50 AND E21 >= 0.30 -> AI, else Real
    pred_gate_b = ((p_e20_dev >= 0.50) & (p_e21_dev >= 0.30)).astype(int)
    p_gate_b = np.minimum(p_e20_dev, p_e21_dev * (0.50 / 0.30))
    m_gate_b = compute_metrics(y_dev_true, p_gate_b, threshold=0.50, y_pred_override=pred_gate_b)
    candidates.append({
        "candidate_type": "gated_rule",
        "name": "Gate_B",
        "w_e20": None, "w_e21": None,
        "rule_description": "If E20 >= 0.50 AND E21 >= 0.30 -> AI, else Real",
        **m_gate_b
    })

    # C: If E20 >= 0.50 OR (E21 >= 0.70 AND E20 >= 0.20) -> AI, else Real
    pred_gate_c = ((p_e20_dev >= 0.50) | ((p_e21_dev >= 0.70) & (p_e20_dev >= 0.20))).astype(int)
    p_gate_c = np.where((p_e21_dev >= 0.70) & (p_e20_dev >= 0.20), np.maximum(p_e20_dev, 0.50), p_e20_dev)
    m_gate_c = compute_metrics(y_dev_true, p_gate_c, threshold=0.50, y_pred_override=pred_gate_c)
    candidates.append({
        "candidate_type": "gated_rule",
        "name": "Gate_C",
        "w_e20": None, "w_e21": None,
        "rule_description": "If E20 >= 0.50 OR (E21 >= 0.70 AND E20 >= 0.20) -> AI, else Real",
        **m_gate_c
    })

    # D: If E21 >= 0.50 AND E20 >= 0.20 -> AI, else Real
    pred_gate_d = ((p_e21_dev >= 0.50) & (p_e20_dev >= 0.20)).astype(int)
    p_gate_d = np.where((p_e21_dev >= 0.50) & (p_e20_dev >= 0.20), np.maximum(p_e21_dev, 0.50), np.minimum(p_e21_dev, 0.49))
    m_gate_d = compute_metrics(y_dev_true, p_gate_d, threshold=0.50, y_pred_override=pred_gate_d)
    candidates.append({
        "candidate_type": "gated_rule",
        "name": "Gate_D",
        "w_e20": None, "w_e21": None,
        "rule_description": "If E21 >= 0.50 AND E20 >= 0.20 -> AI, else Real",
        **m_gate_d
    })

    # Save selection table
    dev_sel_csv = OUT_DIR / "selection_on_e21_dev.csv"
    with open(dev_sel_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"  [*] Saved Dev selection metrics to: {dev_sel_csv}")

    # Print selection table:
    print("\n--- E21 DEV CANDIDATE PERFORMANCE TABLE ---")
    print(f"{'Candidate Name':<35} | {'Acc':<7} | {'F1':<7} | {'ROC-AUC':<8} | {'Recall':<7} | {'FPR':<7}")
    print("-" * 80)
    for c in candidates:
        print(f"{c['name']:<35} | {c['accuracy']*100:6.2f}% | {c['f1']:7.4f} | {c['roc_auc']:8.4f} | {c['recall']*100:6.2f}% | {c['fpr']*100:6.2f}%")

    # Selection sorting:
    # PRIMARY: highest F1
    # TIE BREAK: lowest Real FPR
    # SECOND: highest ROC-AUC
    # THIRD: highest AI Recall
    def sort_key(c):
        # We negate values that we want to maximize
        return (-round(c["f1"], 4), round(c["fpr"], 4), -round(c["roc_auc"], 4), -round(c["recall"], 4))

    sorted_candidates = sorted(candidates, key=sort_key)
    winner = sorted_candidates[0]

    print("\n==================================================")
    print("WINNING ENSEMBLE RULE SELECTED STRICTLY ON E21 DEV:")
    print(f"  Selected Rule:  {winner['name']}")
    print(f"  Description:    {winner['rule_description']}")
    print(f"  Dev F1:         {winner['f1']:.4f}")
    print(f"  Dev ROC-AUC:    {winner['roc_auc']:.4f}")
    print(f"  Dev Accuracy:   {winner['accuracy']*100:.2f}%")
    print(f"  Dev AI Recall:  {winner['recall']*100:.2f}%")
    print(f"  Dev Real FPR:   {winner['fpr']*100:.2f}%")
    print("==================================================")

    # 3. Phase B: Apply Winning Ensemble Function
    def apply_ensemble(p_20: np.ndarray, p_21: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if winner["candidate_type"] == "linear_combination":
            w20 = winner["w_e20"]
            w21 = winner["w_e21"]
            p_ens = w20 * p_20 + w21 * p_21
            pred_ens = (p_ens >= 0.50).astype(int)
            return p_ens, pred_ens
        elif winner["name"] == "Gate_A":
            pred_ens = ((p_20 >= 0.50) | (p_21 >= 0.70)).astype(int)
            p_ens = np.maximum(p_20, p_21 * (0.50 / 0.70))
            return p_ens, pred_ens
        elif winner["name"] == "Gate_B":
            pred_ens = ((p_20 >= 0.50) & (p_21 >= 0.30)).astype(int)
            p_ens = np.minimum(p_20, p_21 * (0.50 / 0.30))
            return p_ens, pred_ens
        elif winner["name"] == "Gate_C":
            pred_ens = ((p_20 >= 0.50) | ((p_21 >= 0.70) & (p_20 >= 0.20))).astype(int)
            p_ens = np.where((p_21 >= 0.70) & (p_20 >= 0.20), np.maximum(p_20, 0.50), p_20)
            return p_ens, pred_ens
        elif winner["name"] == "Gate_D":
            pred_ens = ((p_21 >= 0.50) & (p_20 >= 0.20)).astype(int)
            p_ens = np.where((p_21 >= 0.50) & (p_20 >= 0.20), np.maximum(p_21, 0.50), np.minimum(p_21, 0.49))
            return p_ens, pred_ens
        else:
            raise ValueError(f"Unknown winning rule: {winner['name']}")

    # 4. Phase C: Sealed External Evaluation across all 5 benchmarks
    print("\n[3/5] Evaluating Frozen Winning Ensemble across all 5 External Benchmarks...")
    with open(E14_CLEAN_CSV, "r", encoding="utf-8") as f:
        e14_clean_items = list(csv.DictReader(f))
    for it in e14_clean_items:
        it.setdefault("degradation_type", "clean")
        it.setdefault("device_family", it.get("source_dataset", "unknown"))

    with open(E14_DEG_CSV, "r", encoding="utf-8") as f:
        e14_deg_items = list(csv.DictReader(f))
    for it in e14_deg_items:
        it.setdefault("image_id", it.get("variant_id"))
        it.setdefault("device_family", "unknown")

    with open(E19_CSV, "r", encoding="utf-8") as f:
        e19_items = list(csv.DictReader(f))
    for it in e19_items:
        it.setdefault("degradation_type", "whatsapp_simulation" if it.get("whatsapp_processed") == "True" else "clean")

    e19_wa_items = [it for it in e19_items if it.get("whatsapp_processed") == "True"]
    wa67_items = load_whatsapp_n67()

    datasets_to_eval = [
        ("E14_Clean", e14_clean_items, 200),
        ("E14_Degraded", e14_deg_items, 600),
        ("E19_Smartphone_Benchmark", e19_items, 200),
        ("E19_WhatsApp_Subset", e19_wa_items, 60),
        ("WhatsApp_N67_Benchmark", wa67_items, 67)
    ]

    all_external_results = []
    all_transitions = []
    complementarity_samples = []

    for name, items, expected_n in datasets_to_eval:
        assert len(items) == expected_n, f"Expected {expected_n} items for {name}, got {len(items)}"
        print(f"\n--- External Benchmark: {name} (N={len(items)}) ---")

        dataset = EvalDataset(items)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

        y_true = np.array([1 if it.get("label") == "AI" else 0 for it in items])

        # Inference for all 3 base models
        p_e6c = run_model_inference(model_e6c, loader)
        p_e20 = run_model_inference(model_e20, loader)
        p_e21 = run_model_inference(model_e21, loader)

        # Apply Frozen Winning Ensemble
        p_e22, pred_e22 = apply_ensemble(p_e20, p_e21)

        # Base predictions
        pred_e6c = (p_e6c >= 0.50).astype(int)
        pred_e20 = (p_e20 >= 0.50).astype(int)
        pred_e21 = (p_e21 >= 0.50).astype(int)

        # Compute metrics
        m_e6c = compute_metrics(y_true, p_e6c, threshold=0.50)
        m_e20 = compute_metrics(y_true, p_e20, threshold=0.50)
        m_e21 = compute_metrics(y_true, p_e21, threshold=0.50)
        m_e22 = compute_metrics(y_true, p_e22, threshold=0.50, y_pred_override=pred_e22)

        print(f"  [E6-C]         Acc: {m_e6c['accuracy']*100:6.2f}% | AUC: {m_e6c['roc_auc']:.4f} | F1: {m_e6c['f1']:.4f} | Rec: {m_e6c['recall']*100:6.2f}% | FPR: {m_e6c['fpr']*100:6.2f}%")
        print(f"  [E20 ]         Acc: {m_e20['accuracy']*100:6.2f}% | AUC: {m_e20['roc_auc']:.4f} | F1: {m_e20['f1']:.4f} | Rec: {m_e20['recall']*100:6.2f}% | FPR: {m_e20['fpr']*100:6.2f}%")
        print(f"  [E21 ]         Acc: {m_e21['accuracy']*100:6.2f}% | AUC: {m_e21['roc_auc']:.4f} | F1: {m_e21['f1']:.4f} | Rec: {m_e21['recall']*100:6.2f}% | FPR: {m_e21['fpr']*100:6.2f}%")
        print(f"  [E22 Ensemble] Acc: {m_e22['accuracy']*100:6.2f}% | AUC: {m_e22['roc_auc']:.4f} | F1: {m_e22['f1']:.4f} | Rec: {m_e22['recall']*100:6.2f}% | FPR: {m_e22['fpr']*100:6.2f}%")

        for m_name, m_dict in [("E6-C", m_e6c), ("E20", m_e20), ("E21", m_e21), ("E22_Ensemble", m_e22)]:
            all_external_results.append({
                "dataset": name,
                "model": m_name,
                "n": m_dict["n"],
                "accuracy": m_dict["accuracy"],
                "roc_auc": m_dict["roc_auc"],
                "pr_auc": m_dict["pr_auc"],
                "precision": m_dict["precision"],
                "recall": m_dict["recall"],
                "f1": m_dict["f1"],
                "fpr": m_dict["fpr"],
                "fnr": m_dict["fnr"],
                "tp": m_dict["tp"],
                "tn": m_dict["tn"],
                "fp": m_dict["fp"],
                "fn": m_dict["fn"]
            })

        # Transitions & Per-image predictions
        pred_csv = PRED_DIR / f"{name.lower()}_e22_predictions.csv"
        pred_rows = []
        for idx, it in enumerate(items):
            lbl = int(y_true[idx])
            p6 = float(p_e6c[idx]); c6 = int(pred_e6c[idx] == lbl)
            p20 = float(p_e20[idx]); c20 = int(pred_e20[idx] == lbl)
            p21 = float(p_e21[idx]); c21 = int(pred_e21[idx] == lbl)
            p22 = float(p_e22[idx]); c22 = int(pred_e22[idx] == lbl)

            # Transitions
            # E20 -> E22
            tr_e20_e22 = "E20_correct__E22_correct" if (c20 and c22) else \
                         ("E20_correct__E22_wrong" if (c20 and not c22) else \
                         ("E20_wrong__E22_correct" if (not c20 and c22) else "E20_wrong__E22_wrong"))

            # E21 -> E22
            tr_e21_e22 = "E21_correct__E22_correct" if (c21 and c22) else \
                         ("E21_correct__E22_wrong" if (c21 and not c22) else \
                         ("E21_wrong__E22_correct" if (not c21 and c22) else "E21_wrong__E22_wrong"))

            # Complementarity: E20 vs E21 discordance
            comp_type = "both_correct" if (c20 and c21) else \
                        ("both_wrong" if (not c20 and not c21) else \
                        ("E20_wrong__E21_correct" if (not c20 and c21) else "E20_correct__E21_wrong"))

            img_id = it.get("image_id", Path(it["filepath"]).stem)
            gen = it.get("generator", "none")
            dev_fam = it.get("device_family", "none")

            all_transitions.append({
                "dataset": name,
                "image_id": img_id,
                "label": it.get("label"),
                "generator": gen,
                "device_family": dev_fam,
                "prob_e20": p20,
                "prob_e21": p21,
                "prob_e22": p22,
                "correct_e20": c20,
                "correct_e21": c21,
                "correct_e22": c22,
                "e20_vs_e22_transition": tr_e20_e22,
                "e21_vs_e22_transition": tr_e21_e22,
                "complementarity_type": comp_type
            })

            pred_rows.append({
                "image_id": img_id,
                "label": it.get("label"),
                "prob_e6c": f"{p6:.6f}", "pred_e6c": pred_e6c[idx],
                "prob_e20": f"{p20:.6f}", "pred_e20": pred_e20[idx],
                "prob_e21": f"{p21:.6f}", "pred_e21": pred_e21[idx],
                "prob_e22": f"{p22:.6f}", "pred_e22": pred_e22[idx],
                "e20_vs_e22_transition": tr_e20_e22,
                "e21_vs_e22_transition": tr_e21_e22,
                "complementarity_type": comp_type,
                "generator": gen,
                "device_family": dev_fam,
                "filepath": it.get("filepath")
            })

        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pred_rows)

        # Subgroups for this dataset:
        for grp_key in ["generator", "device_family", "degradation_type"]:
            unique_grps = sorted(list(set(str(it.get(grp_key, "unknown")) for it in items)))
            sub_rows = []
            for g in unique_grps:
                idxs = [i for i, it in enumerate(items) if str(it.get(grp_key, "unknown")) == g]
                if len(idxs) == 0:
                    continue
                sub_y = y_true[idxs]
                sub_p_e6c = p_e6c[idxs]
                sub_p_e20 = p_e20[idxs]
                sub_p_e21 = p_e21[idxs]
                sub_p_e22 = p_e22[idxs]
                sub_pred_e22 = pred_e22[idxs]

                sm_e6c = compute_metrics(sub_y, sub_p_e6c)
                sm_e20 = compute_metrics(sub_y, sub_p_e20)
                sm_e21 = compute_metrics(sub_y, sub_p_e21)
                sm_e22 = compute_metrics(sub_y, sub_p_e22, y_pred_override=sub_pred_e22)

                sub_rows.append({
                    "group_field": grp_key,
                    "group_value": g,
                    "n": len(idxs),
                    "n_real": int(np.sum(sub_y == 0)),
                    "n_ai": int(np.sum(sub_y == 1)),
                    "e6c_acc": sm_e6c["accuracy"],
                    "e20_acc": sm_e20["accuracy"],
                    "e21_acc": sm_e21["accuracy"],
                    "e22_acc": sm_e22["accuracy"],
                    "e22_f1": sm_e22["f1"],
                    "e6c_recall": sm_e6c["recall"] if np.sum(sub_y == 1) > 0 else None,
                    "e20_recall": sm_e20["recall"] if np.sum(sub_y == 1) > 0 else None,
                    "e21_recall": sm_e21["recall"] if np.sum(sub_y == 1) > 0 else None,
                    "e22_recall": sm_e22["recall"] if np.sum(sub_y == 1) > 0 else None,
                    "e6c_fpr": sm_e6c["fpr"] if np.sum(sub_y == 0) > 0 else None,
                    "e20_fpr": sm_e20["fpr"] if np.sum(sub_y == 0) > 0 else None,
                    "e21_fpr": sm_e21["fpr"] if np.sum(sub_y == 0) > 0 else None,
                    "e22_fpr": sm_e22["fpr"] if np.sum(sub_y == 0) > 0 else None,
                })
            if sub_rows:
                sub_csv = SUBGROUP_DIR / f"{name.lower()}_subgroup_{grp_key}.csv"
                with open(sub_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(sub_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(sub_rows)

    # 5. Save Overall External Results and Transitions
    print("\n[4/5] Saving external results, transitions, and complementarity logs...")
    ext_res_csv = OUT_DIR / "e22_external_results.csv"
    with open(ext_res_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_external_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_external_results)
    print("  [*] Saved:", ext_res_csv)

    trans_csv = OUT_DIR / "e22_error_transitions.csv"
    with open(trans_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_transitions[0].keys()))
        writer.writeheader()
        writer.writerows(all_transitions)
    print("  [*] Saved:", trans_csv)

    # Transition Summary Print
    print("\n==================================================")
    print("EXTERNAL TRANSITION & COMPLEMENTARITY SUMMARY:")
    for ds_name in ["E14_Clean", "E14_Degraded", "E19_Smartphone_Benchmark", "E19_WhatsApp_Subset", "WhatsApp_N67_Benchmark"]:
        ds_tr = [t for t in all_transitions if t["dataset"] == ds_name]
        from collections import Counter
        c_e20 = Counter(t["e20_vs_e22_transition"] for t in ds_tr)
        c_e21 = Counter(t["e21_vs_e22_transition"] for t in ds_tr)
        c_comp = Counter(t["complementarity_type"] for t in ds_tr)

        print(f"\n[{ds_name}] (N={len(ds_tr)}):")
        print(f"  Complementarity Discordance: E20 wrong & E21 correct = {c_comp.get('E20_wrong__E21_correct', 0)} | E20 correct & E21 wrong = {c_comp.get('E20_correct__E21_wrong', 0)}")
        print(f"  E20 vs E22: E20 Wrong -> E22 Correct (Recovered): {c_e20.get('E20_wrong__E22_correct', 0)}")
        print(f"              E20 Correct -> E22 Wrong (Regressed): {c_e20.get('E20_correct__E22_wrong', 0)}")
        print(f"  E21 vs E22: E21 Wrong -> E22 Correct (Recovered): {c_e21.get('E21_wrong__E22_correct', 0)}")
        print(f"              E21 Correct -> E22 Wrong (Regressed): {c_e21.get('E21_correct__E22_wrong', 0)}")
    print("==================================================")

if __name__ == "__main__":
    main()
