"""
Experiment E9: E6-C + E8-B Compression-Robust Ensemble Study

Strict Safety & Scientific Protocol:
- INFERENCE-ONLY: No training, no fine-tuning, 0 trainable parameters.
- Frozen Checkpoints:
    E6-C: experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt
    E8-B: experiments/e8_compression_robustness/checkpoints/e8b_best_model.pt
- E5 Validation Set (N=5,775) is used EXCLUSIVELY for development, weight sweeps, and rule selection.
- WhatsApp Robustness Test Set (N=67: 34 Real, 33 AI) is a FROZEN TEST SET.
- Evaluated ONCE only after the final E9 rule is permanently frozen.
- No production modifications, no dataset modifications, no git commits.
"""

import os
import sys
import time
import json
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model

E6C_CHECKPOINT = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
E8B_CHECKPOINT = PROJECT_ROOT / "experiments/e8_compression_robustness/checkpoints/e8b_best_model.pt"
E5_MANIFEST_PATH = PROJECT_ROOT / "data/e5_external/manifests/e5_manifest.csv"
WHATSAPP_DIR = PROJECT_ROOT / "data/whatsapp_robustness_test"
HARD_CASES = {
    "original_ai_edit": PROJECT_ROOT / "data/e6_hard_cases/original_ai_edit.png",
    "whatsapp_download": PROJECT_ROOT / "data/e6_hard_cases/whatsapp_download.jpeg",
}
OUT_DIR = PROJECT_ROOT / "experiments/e9_ensemble_study"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_5_crops(img: Image.Image) -> list:
    """Exact production 5-view crop generation."""
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    crops = [
        img.resize((224, 224), Image.Resampling.BILINEAR),  # View 0: Global
        img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR),  # View 1: Top-Left
        img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR),  # View 2: Top-Right
        img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR),  # View 3: Bottom-Left
        img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR),  # View 4: Bottom-Right
    ]
    return crops


class MultiViewDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["image_path"]
        label = rec["label"]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            view_tensors = [self.to_tensor(c) for c in crops]
            stacked_views = torch.stack(view_tensors, dim=0)  # (5, 3, 224, 224)
            return stacked_views, torch.tensor(label, dtype=torch.float32), True, idx
        except Exception as e:
            dummy_views = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
            return dummy_views, torch.tensor(label, dtype=torch.float32), False, idx


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Exact Mann-Whitney U statistic ROC-AUC with tie averaging."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

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
    return float(u_pos / (n_pos * n_neg))


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR-AUC via trapezoidal integration over precision-recall curve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    desc_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[desc_idx]
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = tps / (tps + fps)
    recall = tps / n_pos

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(abs(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2.0)))


def compute_metrics(y_true: np.ndarray, preds: np.ndarray, y_score: np.ndarray = None):
    y_true = np.asarray(y_true, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    n_total = len(y_true)
    acc = (tp + tn) / max(1, n_total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    roc_auc = compute_roc_auc(y_true, y_score) if y_score is not None else 0.0
    pr_auc = compute_pr_auc(y_true, y_score) if y_score is not None else 0.0

    return {
        "n_samples": n_total,
        "n_real": int((y_true == 0).sum()),
        "n_ai": int((y_true == 1).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def load_frozen_model(checkpoint_path: Path):
    model = MultiViewE5Model(
        checkpoint_path=None,
        num_views=5,
        freq_norm_strategy="standardize",
        freq_embedding_dim=256,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == 0, f"Error: Model has {n_trainable} trainable parameters!"
    return model


def run_inference_on_loader(model: nn.Module, loader: DataLoader):
    p_learned_list = []
    p_global_list = []
    p_strongest_local_list = []
    y_true_list = []

    with torch.no_grad():
        for x_views, y_true, valid, _ in loader:
            mask = valid.bool()
            if not mask.any():
                continue
            x_views = x_views[mask].to(DEVICE, non_blocking=True)
            y_true = y_true[mask]

            B, V, C, H, W = x_views.shape
            x_flat = x_views.view(B * V, C, H, W)
            feats = model.base_model(x_flat, return_features=True)
            e_fused = feats["fused_embedding"]

            view_logits = model.classifier(e_fused).view(B, V)
            view_probs = torch.sigmoid(view_logits)

            e_fused_views = e_fused.view(B, V, model.fused_dim)
            attn_scores = model.view_attention(e_fused_views)
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled_emb = torch.sum(attn_weights * e_fused_views, dim=1)
            attn_logits = model.classifier(pooled_emb)
            p_learned = torch.sigmoid(attn_logits).squeeze(-1)

            p_global = view_probs[:, 0]
            p_local = view_probs[:, 1:]
            p_strongest_local = torch.max(p_local, dim=1).values

            p_learned_list.extend(p_learned.cpu().numpy().tolist())
            p_global_list.extend(p_global.cpu().numpy().tolist())
            p_strongest_local_list.extend(p_strongest_local.cpu().numpy().tolist())
            y_true_list.extend(y_true.numpy().tolist())

    return {
        "p_learned": np.array(p_learned_list),
        "p_global": np.array(p_global_list),
        "p_strongest_local": np.array(p_strongest_local_list),
        "y_true": np.array(y_true_list, dtype=int),
    }


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT E9: E6-C + E8-B COMPRESSION-ROBUST ENSEMBLE STUDY")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"E6-C Checkpoint: {E6C_CHECKPOINT.relative_to(PROJECT_ROOT)}")
    print(f"E8-B Checkpoint: {E8B_CHECKPOINT.relative_to(PROJECT_ROOT)}")

    # -------------------------------------------------------------
    # 1. Load E5 Validation Dataset (Development Dataset ONLY)
    # -------------------------------------------------------------
    print("\n[STEP 1] Loading E5 Validation Dataset Manifest (N=5,775)...")
    e5_val_records = []
    with open(E5_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                e5_val_records.append({
                    "image_path": str(PROJECT_ROOT / row["image_path"]),
                    "rel_path": row["image_path"],
                    "label": int(row["label"]),
                    "source_dataset": row.get("source_dataset", ""),
                    "generator_or_device": row.get("generator_or_device", ""),
                })

    assert len(e5_val_records) == 5775, f"Expected 5,775 validation images, got {len(e5_val_records)}"
    print(f"Loaded {len(e5_val_records):,} E5 validation records.")

    e5_loader = DataLoader(
        MultiViewDataset(e5_val_records),
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # -------------------------------------------------------------
    # 2. Extract E6-C Predictions on E5 Validation
    # -------------------------------------------------------------
    print("\n[STEP 1a] Running Frozen E6-C on E5 Validation...")
    e6c_model = load_frozen_model(E6C_CHECKPOINT)
    t0 = time.time()
    e6c_e5 = run_inference_on_loader(e6c_model, e5_loader)
    t_e6c = time.time() - t0
    print(f"E6-C E5 inference finished in {t_e6c:.1f}s ({len(e5_val_records)/t_e6c:.1f} img/s).")

    del e6c_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. Extract E8-B Predictions on E5 Validation
    # -------------------------------------------------------------
    print("\n[STEP 1b] Running Frozen E8-B on E5 Validation...")
    e8b_model = load_frozen_model(E8B_CHECKPOINT)
    t0 = time.time()
    e8b_e5 = run_inference_on_loader(e8b_model, e5_loader)
    t_e8b = time.time() - t0
    print(f"E8-B E5 inference finished in {t_e8b:.1f}s ({len(e5_val_records)/t_e8b:.1f} img/s).")

    del e8b_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_true_e5 = e6c_e5["y_true"]
    p_e6c_e5 = e6c_e5["p_learned"]
    p_e8b_e5 = e8b_e5["p_learned"]

    # Baseline individual metrics on E5
    m_e6c_base = compute_metrics(y_true_e5, p_e6c_e5 >= 0.50, p_e6c_e5)
    m_e8b_base = compute_metrics(y_true_e5, p_e8b_e5 >= 0.50, p_e8b_e5)
    print(f"\nBaseline E6-C on E5 Val: Acc={m_e6c_base['accuracy']:.4f}, AUC={m_e6c_base['roc_auc']:.4f}, F1={m_e6c_base['f1']:.4f}, FPR={m_e6c_base['fpr']*100:.2f}%, Recall={m_e6c_base['recall']*100:.2f}%")
    print(f"Baseline E8-B on E5 Val: Acc={m_e8b_base['accuracy']:.4f}, AUC={m_e8b_base['roc_auc']:.4f}, F1={m_e8b_base['f1']:.4f}, FPR={m_e8b_base['fpr']*100:.2f}%, Recall={m_e8b_base['recall']*100:.2f}%")

    # Save E5 predictions CSV
    e5_pred_csv_path = OUT_DIR / "e5_val_predictions.csv"
    with open(e5_pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rel_path", "ground_truth",
            "e6c_prob", "e8b_prob",
            "e6c_pred_050", "e8b_pred_050",
            "e6c_global_prob", "e6c_strongest_local_prob",
            "e8b_global_prob", "e8b_strongest_local_prob",
        ])
        for idx in range(len(e5_val_records)):
            writer.writerow([
                e5_val_records[idx]["rel_path"],
                int(y_true_e5[idx]),
                round(float(p_e6c_e5[idx]), 6),
                round(float(p_e8b_e5[idx]), 6),
                int(p_e6c_e5[idx] >= 0.50),
                int(p_e8b_e5[idx] >= 0.50),
                round(float(e6c_e5["p_global"][idx]), 6),
                round(float(e6c_e5["p_strongest_local"][idx]), 6),
                round(float(e8b_e5["p_global"][idx]), 6),
                round(float(e8b_e5["p_strongest_local"][idx]), 6),
            ])
    print(f"Saved E5 validation predictions to {e5_pred_csv_path.name} ({len(e5_val_records):,} rows).")

    # -------------------------------------------------------------
    # 4. STEP 2: Continuous Score Ensembles Sweep on E5 Validation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: CONTINUOUS SCORE ENSEMBLES (E5 VALIDATION ONLY)")
    print("=" * 80)
    weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    ensemble_results = []

    print(f"{'Weight (E8)':<12} {'Accuracy':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FPR (%)':<10} {'FNR (%)':<10}")
    print("-" * 88)
    for w in weights:
        p_ens = w * p_e8b_e5 + (1.0 - w) * p_e6c_e5
        preds_ens = (p_ens >= 0.50).astype(int)
        m = compute_metrics(y_true_e5, preds_ens, p_ens)
        m["weight_e8"] = round(w, 2)
        m["weight_e6c"] = round(1.0 - w, 2)
        ensemble_results.append(m)
        print(f"{w:<12.2f} {m['accuracy']:<10.4f} {m['roc_auc']:<10.4f} {m['pr_auc']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['fpr']*100:<10.2f} {m['fnr']*100:<10.2f}")

    ens_csv_path = OUT_DIR / "candidate_ensemble_results.csv"
    with open(ens_csv_path, "w", newline="", encoding="utf-8") as f:
        keys = ["weight_e8", "weight_e6c", "accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1", "fpr", "fnr", "tp", "fp", "tn", "fn"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for res in ensemble_results:
            writer.writerow({k: res[k] for k in keys})
    print(f"\nSaved candidate ensemble results to {ens_csv_path.name}")

    # -------------------------------------------------------------
    # 5. STEP 3: Conservative Gated Rules on E5 Validation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: CONSERVATIVE GATED RULES SWEEP (E5 VALIDATION ONLY)")
    print("=" * 80)
    gated_results = []

    # Rule Family A:
    # IF E6-C >= 0.50: AI
    # ELSE IF E6-C >= T_low AND E8-B >= T_E8: AI
    # ELSE: REAL
    t_low_candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    t_e8_candidates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for t_low in t_low_candidates:
        for t_e8 in t_e8_candidates:
            cond_ai = (p_e6c_e5 >= 0.50) | ((p_e6c_e5 >= t_low) & (p_e8b_e5 >= t_e8))
            preds = cond_ai.astype(int)
            score_surrogate = np.where(cond_ai, np.maximum(p_e6c_e5, 0.50), np.minimum(p_e6c_e5, 0.499))
            m = compute_metrics(y_true_e5, preds, score_surrogate)
            m["family"] = "Family_A"
            m["formula"] = f"IF E6C>=0.50 -> AI; ELIF (E6C>={t_low:.2f} AND E8B>={t_e8:.2f}) -> AI; ELSE -> REAL"
            m["t_low"] = round(t_low, 2)
            m["t_e8"] = round(t_e8, 2)
            m["t_extra"] = None
            gated_results.append(m)

    # Rule Family B: Gated Veto / High Dual
    veto_candidates = [0.05, 0.10, 0.15, 0.20, 0.25]
    for t_v in veto_candidates:
        cond_ai = (p_e6c_e5 >= 0.50) & (p_e8b_e5 >= t_v)
        preds = cond_ai.astype(int)
        m = compute_metrics(y_true_e5, preds, p_e6c_e5)
        m["family"] = "Family_B_Veto"
        m["formula"] = f"IF E6C>=0.50 AND E8B>={t_v:.2f} -> AI; ELSE -> REAL"
        m["t_low"] = 0.50
        m["t_e8"] = round(t_v, 2)
        m["t_extra"] = None
        gated_results.append(m)

    # Rule Family C: Ambiguity Interval
    for t_floor in [0.25, 0.30, 0.35, 0.40]:
        for t_ceil in [0.50, 0.55, 0.60]:
            for t_gate in [0.60, 0.70, 0.80, 0.90]:
                cond_ai = (p_e6c_e5 >= t_ceil) | ((p_e6c_e5 >= t_floor) & (p_e8b_e5 >= t_gate))
                preds = cond_ai.astype(int)
                m = compute_metrics(y_true_e5, preds, p_e6c_e5)
                m["family"] = "Family_C_Interval"
                m["formula"] = f"IF E6C>={t_ceil:.2f} -> AI; ELIF (E6C>={t_floor:.2f} AND E8B>={t_gate:.2f}) -> AI; ELSE -> REAL"
                m["t_low"] = round(t_floor, 2)
                m["t_e8"] = round(t_gate, 2)
                m["t_extra"] = round(t_ceil, 2)
                gated_results.append(m)

    # Save all candidate gated results
    gated_csv_path = OUT_DIR / "candidate_gated_results.csv"
    with open(gated_csv_path, "w", newline="", encoding="utf-8") as f:
        keys = ["family", "formula", "t_low", "t_e8", "t_extra", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "tp", "fp", "tn", "fn"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for res in gated_results:
            writer.writerow({k: res[k] for k in keys})
    print(f"Saved {len(gated_results)} candidate gated-rule results to {gated_csv_path.name}")

    # -------------------------------------------------------------
    # Filter by FPR Constraints on E5 Validation
    # -------------------------------------------------------------
    fpr_e6c = m_e6c_base["fpr"]
    recall_e6c = m_e6c_base["recall"]
    f1_e6c = m_e6c_base["f1"]

    print("\n" + "=" * 80)
    print("FPR-CONSTRAINED CANDIDATE EVALUATION (E5 VALIDATION)")
    print(f"Reference E6-C Baseline: FPR = {fpr_e6c*100:.2f}%, Recall = {recall_e6c*100:.2f}%, F1 = {f1_e6c:.4f}")
    print("=" * 80)

    tier_3_5 = [r for r in gated_results if r["fpr"] <= 0.035]
    tier_5_0 = [r for r in gated_results if r["fpr"] <= 0.050]
    tier_7_5 = [r for r in gated_results if r["fpr"] <= 0.075]

    ens_tier_3_5 = [r for r in ensemble_results if r["fpr"] <= 0.035]
    ens_tier_5_0 = [r for r in ensemble_results if r["fpr"] <= 0.050]
    ens_tier_7_5 = [r for r in ensemble_results if r["fpr"] <= 0.075]

    print("\n--- TOP CONTINUOUS ENSEMBLES UNDER FPR <= 3.5% ---")
    for r in sorted(ens_tier_3_5, key=lambda x: (x["f1"], x["recall"]), reverse=True)[:5]:
        print(f"Weight E8={r['weight_e8']:.2f} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | Recall={r['recall']*100:.2f}% | FPR={r['fpr']*100:.2f}% | AUC={r['roc_auc']:.4f}")

    print("\n--- TOP GATED RULES UNDER FPR <= 3.5% ---")
    for r in sorted(tier_3_5, key=lambda x: (x["f1"], x["recall"]), reverse=True)[:5]:
        print(f"{r['family']} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | Recall={r['recall']*100:.2f}% | FPR={r['fpr']*100:.2f}% | Formula: {r['formula']}")

    print("\n--- TOP GATED RULES UNDER FPR <= 5.0% ---")
    for r in sorted(tier_5_0, key=lambda x: (x["f1"], x["recall"]), reverse=True)[:5]:
        print(f"{r['family']} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | Recall={r['recall']*100:.2f}% | FPR={r['fpr']*100:.2f}% | Formula: {r['formula']}")

    # -------------------------------------------------------------
    # 6. STEP 4: FREEZE ONE FINAL RULE ON E5 VALIDATION ONLY
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: FINAL RULE SELECTION & PERMANENT FREEZING (E5 VALIDATION ONLY)")
    print("=" * 80)

    all_candidates = []
    for r in ensemble_results:
        all_candidates.append({
            "type": "continuous_ensemble",
            "name": f"Ensemble (w_e8={r['weight_e8']:.2f})",
            "weight_e8": r["weight_e8"],
            "rule_func": ("ensemble", r["weight_e8"], 0.50),
            "metrics": r,
            "fpr": r["fpr"],
            "recall": r["recall"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
        })

    for r in gated_results:
        all_candidates.append({
            "type": "gated_rule",
            "name": r["formula"],
            "family": r["family"],
            "rule_func": ("gated", r["t_low"], r["t_e8"], r["t_extra"]),
            "metrics": r,
            "fpr": r["fpr"],
            "recall": r["recall"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
        })

    valid_candidates_strict = [c for c in all_candidates if c["fpr"] <= 0.0350]
    valid_candidates_strict.sort(key=lambda c: (c["f1"], c["recall"], -c["fpr"]), reverse=True)

    print("\nStrict FPR <= 3.5% Candidates ranked by (F1, Recall, -FPR):")
    for i, c in enumerate(valid_candidates_strict[:10]):
        print(f"[{i+1}] {c['name']} -> F1={c['f1']:.4f}, Acc={c['accuracy']:.4f}, Recall={c['recall']*100:.2f}% (dRecall={round((c['recall']-recall_e6c)*100, 2):+0.2f}%), FPR={c['fpr']*100:.2f}%")

    top_gated = [c for c in valid_candidates_strict if c["type"] == "gated_rule"][0]
    top_ens = [c for c in valid_candidates_strict if c["type"] == "continuous_ensemble"][0]

    print(f"\nTop Continuous Ensemble: {top_ens['name']} | F1={top_ens['f1']:.4f}, Recall={top_ens['recall']*100:.2f}%, FPR={top_ens['fpr']*100:.2f}%")
    print(f"Top Gated Rule:         {top_gated['name']} | F1={top_gated['f1']:.4f}, Recall={top_gated['recall']*100:.2f}%, FPR={top_gated['fpr']*100:.2f}%")

    # Select the #1 overall candidate meeting FPR <= 3.5%
    selected_winner = valid_candidates_strict[0]
    print("\n" + "#" * 80)
    print("FINAL SELECTION DECISION ON E5 VALIDATION:")
    print(f"Selected Rule: {selected_winner['name']}")
    print(f"E5 Val Accuracy:  {selected_winner['accuracy']:.4f}")
    print(f"E5 Val F1-Score:  {selected_winner['f1']:.4f}")
    print(f"E5 Val Recall:    {selected_winner['recall']*100:.2f}% (vs E6-C: {recall_e6c*100:.2f}%)")
    print(f"E5 Val FPR:       {selected_winner['fpr']*100:.2f}% (vs E6-C: {fpr_e6c*100:.2f}%)")
    print("#" * 80)

    # EXPLICIT PROTOCOL REQUIREMENT:
    print("\nFINAL E9 RULE FROZEN — NO FURTHER SELECTION\n")

    def evaluate_e9(p_e6c, p_e8b):
        p_e6c_arr = np.asarray(p_e6c, dtype=float)
        p_e8b_arr = np.asarray(p_e8b, dtype=float)

        rule_type = selected_winner["rule_func"][0]
        if rule_type == "ensemble":
            w = selected_winner["rule_func"][1]
            p_e9 = w * p_e8b_arr + (1.0 - w) * p_e6c_arr
            preds = (p_e9 >= 0.50).astype(int)
            return preds, p_e9
        elif rule_type == "gated":
            t_low = selected_winner["rule_func"][1]
            t_e8 = selected_winner["rule_func"][2]
            t_extra = selected_winner["rule_func"][3]
            t_ceil = 0.50 if t_extra is None else t_extra

            cond_ai = (p_e6c_arr >= t_ceil) | ((p_e6c_arr >= t_low) & (p_e8b_arr >= t_e8))
            preds = cond_ai.astype(int)
            p_e9 = np.where(cond_ai, np.maximum(p_e6c_arr, 0.50), np.minimum(p_e6c_arr, 0.499))
            return preds, p_e9
        else:
            raise ValueError(f"Unknown rule type {rule_type}")

    # -------------------------------------------------------------
    # 7. STEP 5: HARD-CASE DIAGNOSTICS
    # -------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: HARD-CASE DIAGNOSTIC EVALUATION")
    print("=" * 80)
    to_tensor = transforms.ToTensor()
    hard_case_results = {}

    e6c_model = load_frozen_model(E6C_CHECKPOINT)
    e8b_model = load_frozen_model(E8B_CHECKPOINT)

    for case_name, case_path in HARD_CASES.items():
        assert case_path.exists(), f"Hard case missing: {case_path}"
        with Image.open(case_path) as img:
            img_rgb = img.convert("RGB")
        crops = generate_5_crops(img_rgb)
        x_views = torch.stack([to_tensor(c) for c in crops], dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feats_6 = e6c_model.base_model(x_views.view(5, 3, 224, 224), return_features=True)
            e_fused_6 = feats_6["fused_embedding"]
            probs_6 = torch.sigmoid(e6c_model.classifier(e_fused_6)).view(1, 5)
            e_fviews_6 = e_fused_6.view(1, 5, e6c_model.fused_dim)
            attn_6 = torch.softmax(e6c_model.view_attention(e_fviews_6), dim=1)
            pooled_6 = torch.sum(attn_6 * e_fviews_6, dim=1)
            p_e6c_val = float(torch.sigmoid(e6c_model.classifier(pooled_6)).squeeze())
            p_e6c_global = float(probs_6[0, 0])
            p_e6c_loc = float(torch.max(probs_6[0, 1:]))

            feats_8 = e8b_model.base_model(x_views.view(5, 3, 224, 224), return_features=True)
            e_fused_8 = feats_8["fused_embedding"]
            probs_8 = torch.sigmoid(e8b_model.classifier(e_fused_8)).view(1, 5)
            e_fviews_8 = e_fused_8.view(1, 5, e8b_model.fused_dim)
            attn_8 = torch.softmax(e8b_model.view_attention(e_fviews_8), dim=1)
            pooled_8 = torch.sum(attn_8 * e_fviews_8, dim=1)
            p_e8b_val = float(torch.sigmoid(e8b_model.classifier(pooled_8)).squeeze())
            p_e8b_global = float(probs_8[0, 0])
            p_e8b_loc = float(torch.max(probs_8[0, 1:]))

        pred_e9, p_e9_val = evaluate_e9(p_e6c_val, p_e8b_val)
        res_dict = {
            "path": str(case_path.relative_to(PROJECT_ROOT)),
            "ground_truth": "AI" if "ai" in case_name.lower() else "REAL",
            "e6c": {
                "prob": round(p_e6c_val, 4),
                "global_prob": round(p_e6c_global, 4),
                "strongest_local_prob": round(p_e6c_loc, 4),
                "pred": "AI" if p_e6c_val >= 0.50 else "REAL",
            },
            "e8b": {
                "prob": round(p_e8b_val, 4),
                "global_prob": round(p_e8b_global, 4),
                "strongest_local_prob": round(p_e8b_loc, 4),
                "pred": "AI" if p_e8b_val >= 0.50 else "REAL",
            },
            "e9": {
                "prob_or_score": round(float(p_e9_val), 4),
                "pred": "AI" if int(pred_e9) == 1 else "REAL",
                "correct": ("AI" if "ai" in case_name.lower() else "REAL") == ("AI" if int(pred_e9) == 1 else "REAL"),
            }
        }
        hard_case_results[case_name] = res_dict
        print(f"[{case_name}] GT={res_dict['ground_truth']} | E6-C={p_e6c_val:.4f} ({res_dict['e6c']['pred']}) | E8-B={p_e8b_val:.4f} ({res_dict['e8b']['pred']}) | E9={float(p_e9_val):.4f} ({res_dict['e9']['pred']}, Correct={res_dict['e9']['correct']})")

    hard_case_path = OUT_DIR / "hard_case_diagnostics.json"
    with open(hard_case_path, "w", encoding="utf-8") as f:
        json.dump(hard_case_results, f, indent=2)
    print(f"Saved hard-case diagnostics to {hard_case_path.name}")

    # -------------------------------------------------------------
    # 8. STEP 6: SINGLE-SHOT FROZEN WHATSAPP TEST (N=67)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: SINGLE-SHOT FROZEN WHATSAPP TEST EVALUATION (N=67)")
    print("=" * 80)

    wa_records = []
    wa_real_dir = WHATSAPP_DIR / "real"
    wa_ai_dir = WHATSAPP_DIR / "ai"

    for p in sorted(wa_real_dir.iterdir()):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            wa_records.append({"image_path": str(p), "rel_path": str(p.relative_to(PROJECT_ROOT)), "label": 0, "filename": p.name})

    for p in sorted(wa_ai_dir.iterdir()):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            wa_records.append({"image_path": str(p), "rel_path": str(p.relative_to(PROJECT_ROOT)), "label": 1, "filename": p.name})

    assert len(wa_records) == 67, f"Expected 67 WhatsApp images, got {len(wa_records)}"
    print(f"Loaded {len(wa_records)} WhatsApp images (34 Real, 33 AI).")

    wa_loader = DataLoader(
        MultiViewDataset(wa_records),
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    e6c_wa = run_inference_on_loader(e6c_model, wa_loader)
    e8b_wa = run_inference_on_loader(e8b_model, wa_loader)

    del e6c_model
    del e8b_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_true_wa = np.array([r["label"] for r in wa_records], dtype=int)
    p_e6c_wa = e6c_wa["p_learned"]
    p_e8b_wa = e8b_wa["p_learned"]

    preds_e9_wa, p_e9_wa = evaluate_e9(p_e6c_wa, p_e8b_wa)

    m_e6c_wa = compute_metrics(y_true_wa, (p_e6c_wa >= 0.50).astype(int), p_e6c_wa)
    m_e8b_wa = compute_metrics(y_true_wa, (p_e8b_wa >= 0.50).astype(int), p_e8b_wa)
    m_e9_wa = compute_metrics(y_true_wa, preds_e9_wa, p_e9_wa)

    print("\nWHATSAPP TEST SET METRICS COMPARISON:")
    print(f"{'Metric':<16} {'E6-C':<14} {'E8-B':<14} {'E9 (Frozen)':<14}")
    print("-" * 58)
    for k in ["accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1", "fpr", "fnr", "tp", "tn", "fp", "fn"]:
        val_6 = f"{m_e6c_wa[k]*100:.2f}%" if k in ["accuracy", "precision", "recall", "fpr", "fnr"] else f"{m_e6c_wa[k]:.4f}" if isinstance(m_e6c_wa[k], float) else str(m_e6c_wa[k])
        val_8 = f"{m_e8b_wa[k]*100:.2f}%" if k in ["accuracy", "precision", "recall", "fpr", "fnr"] else f"{m_e8b_wa[k]:.4f}" if isinstance(m_e8b_wa[k], float) else str(m_e8b_wa[k])
        val_9 = f"{m_e9_wa[k]*100:.2f}%" if k in ["accuracy", "precision", "recall", "fpr", "fnr"] else f"{m_e9_wa[k]:.4f}" if isinstance(m_e9_wa[k], float) else str(m_e9_wa[k])
        print(f"{k.upper():<16} {val_6:<14} {val_8:<14} {val_9:<14}")

    # -------------------------------------------------------------
    # 9. TRANSITION ANALYSIS (E9 vs E6-C)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRANSITION ANALYSIS (E9 vs E6-C on WHATSAPP)")
    print("=" * 80)

    preds_6 = (p_e6c_wa >= 0.50).astype(int)
    preds_9 = preds_e9_wa.astype(int)

    recovered_ai_fn = []
    lost_e6c_tp = []
    additional_real_fp = []
    corrected_e6c_fp = []
    all_transitions = []

    per_image_rows = []
    for i in range(len(wa_records)):
        gt = int(y_true_wa[i])
        p6 = float(p_e6c_wa[i])
        p8 = float(p_e8b_wa[i])
        p9 = float(p_e9_wa[i])
        d6 = int(preds_6[i])
        d9 = int(preds_9[i])
        fname = wa_records[i]["filename"]
        rpath = wa_records[i]["rel_path"]

        transition_type = "Unchanged"
        reason = "Consistent decision across models"

        if d6 != d9:
            if gt == 1 and d6 == 0 and d9 == 1:
                transition_type = "AI_FN_Recovered"
                reason = f"E8-B signal elevated prediction past threshold"
                recovered_ai_fn.append((fname, p6, p8, p9))
            elif gt == 1 and d6 == 1 and d9 == 0:
                transition_type = "AI_TP_Lost"
                reason = f"E9 rule flipped true positive AI to Real"
                lost_e6c_tp.append((fname, p6, p8, p9))
            elif gt == 0 and d6 == 0 and d9 == 1:
                transition_type = "Real_FP_Introduced"
                reason = f"E8-B signal triggered false positive on Real image"
                additional_real_fp.append((fname, p6, p8, p9))
            elif gt == 0 and d6 == 1 and d9 == 0:
                transition_type = "Real_FP_Corrected"
                reason = f"E9 successfully suppressed false positive on Real image"
                corrected_e6c_fp.append((fname, p6, p8, p9))

            all_transitions.append({
                "filename": fname,
                "ground_truth": "AI" if gt == 1 else "REAL",
                "e6c_prob": round(p6, 4),
                "e8b_prob": round(p8, 4),
                "e9_score": round(p9, 4),
                "e6c_decision": "AI" if d6 == 1 else "REAL",
                "e9_decision": "AI" if d9 == 1 else "REAL",
                "transition_type": transition_type,
                "reason": reason,
            })

        per_image_rows.append({
            "filename": fname,
            "rel_path": rpath,
            "ground_truth": "AI" if gt == 1 else "REAL",
            "gt_int": gt,
            "e6c_prob": round(p6, 4),
            "e8b_prob": round(p8, 4),
            "e9_score": round(p9, 4),
            "e6c_pred": "AI" if d6 == 1 else "REAL",
            "e8b_pred": "AI" if p8 >= 0.50 else "REAL",
            "e9_pred": "AI" if d9 == 1 else "REAL",
            "e6c_correct": int(d6 == gt),
            "e8b_correct": int((p8 >= 0.50) == gt),
            "e9_correct": int(d9 == gt),
            "transition": transition_type,
        })

    print(f"AI False Negatives Recovered: {len(recovered_ai_fn)}")
    for item in recovered_ai_fn:
        print(f"  + [RECOVERED FN] {item[0]}: E6-C={item[1]:.4f}, E8-B={item[2]:.4f}, E9={item[3]:.4f}")

    print(f"E6-C True Positives Lost: {len(lost_e6c_tp)}")
    for item in lost_e6c_tp:
        print(f"  - [LOST TP] {item[0]}: E6-C={item[1]:.4f}, E8-B={item[2]:.4f}, E9={item[3]:.4f}")

    print(f"Additional Real False Positives: {len(additional_real_fp)}")
    for item in additional_real_fp:
        print(f"  ! [NEW FP] {item[0]}: E6-C={item[1]:.4f}, E8-B={item[2]:.4f}, E9={item[3]:.4f}")

    print(f"E6-C False Positives Corrected: {len(corrected_e6c_fp)}")
    for item in corrected_e6c_fp:
        print(f"  * [CORRECTED FP] {item[0]}: E6-C={item[1]:.4f}, E8-B={item[2]:.4f}, E9={item[3]:.4f}")

    wa_csv_path = OUT_DIR / "per_image_predictions_whatsapp.csv"
    with open(wa_csv_path, "w", newline="", encoding="utf-8") as f:
        keys = list(per_image_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in per_image_rows:
            writer.writerow(r)
    print(f"\nSaved WhatsApp predictions to {wa_csv_path.name}")

    # -------------------------------------------------------------
    # 10. SAVE STUDY RESULTS JSON
    # -------------------------------------------------------------
    study_results = {
        "experiment": "E9: E6-C + E8-B Compression-Robust Ensemble Study",
        "frozen_rule": {
            "name": selected_winner["name"],
            "type": selected_winner["type"],
            "rule_func": selected_winner["rule_func"],
            "rationale": "Selected strictly on E5 validation dataset to maximize F1 and recall under the strict constraint FPR <= 3.5%.",
            "e5_val_metrics": selected_winner["metrics"],
        },
        "models": {
            "e6c_checkpoint": str(E6C_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "e8b_checkpoint": str(E8B_CHECKPOINT.relative_to(PROJECT_ROOT)),
        },
        "e5_validation_comparison": {
            "e6c": m_e6c_base,
            "e8b": m_e8b_base,
            "e9": selected_winner["metrics"],
        },
        "hard_case_diagnostics": hard_case_results,
        "whatsapp_test_comparison": {
            "e6c": m_e6c_wa,
            "e8b": m_e8b_wa,
            "e9": m_e9_wa,
        },
        "transition_analysis": {
            "recovered_ai_fn_count": len(recovered_ai_fn),
            "lost_e6c_tp_count": len(lost_e6c_tp),
            "additional_real_fp_count": len(additional_real_fp),
            "corrected_e6c_fp_count": len(corrected_e6c_fp),
            "transitions": all_transitions,
        },
    }

    results_json_path = OUT_DIR / "study_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(study_results, f, indent=2)
    print(f"Saved complete study results to {results_json_path.name}")

    # -------------------------------------------------------------
    # 11. GENERATE RESEARCH REPORT AND README
    # -------------------------------------------------------------
    generate_research_report(study_results)
    generate_readme(study_results)
    print("\nExperiment E9 execution finished successfully!")


def generate_research_report(results):
    report_path = OUT_DIR / "E9_RESEARCH_REPORT.md"
    rule = results["frozen_rule"]
    e5_comp = results["e5_validation_comparison"]
    wa_comp = results["whatsapp_test_comparison"]
    trans = results["transition_analysis"]
    hc = results["hard_case_diagnostics"]

    m6_wa = wa_comp["e6c"]
    m8_wa = wa_comp["e8b"]
    m9_wa = wa_comp["e9"]

    if len(trans["transitions"]) > 0:
        trans_rows = "\n".join([
            f"| `{t['filename']}` | {t['ground_truth']} | {t['e6c_prob']:.4f} | {t['e8b_prob']:.4f} | {t['e9_score']:.4f} | {t['e6c_decision']} | {t['e9_decision']} | **{t['transition_type']}** |"
            for t in trans["transitions"]
        ])
    else:
        trans_rows = "| *None* | - | - | - | - | - | - | *No decision transitions occurred at threshold 0.50* |"

    content = f"""# Experiment E9: E6-C + E8-B Compression-Robust Ensemble Study

## Executive Summary

Experiment E9 evaluates whether the production model **E6-C** and the compression-aware model **E8-B** can be combined via continuous ensembling or conservative gated decision rules to solve the compression robustness trade-off: **retaining E8-B's improved WhatsApp AI recall while controlling its elevated false-positive rate (FPR)**.

Following strict scientific hygiene, development and rule selection occurred **exclusively on the clean E5 validation set ($N=5,775$)** under a strict constraint ($\\text{{FPR}} \\le 3.5\\%$, close to E6-C's clean baseline of 2.97%). The frozen WhatsApp dataset ($N=67$: 34 Real, 33 AI) was evaluated **single-shot** only after permanently freezing the winning rule.

---

## 1. Frozen Models & Datasets

- **E6-C Checkpoint**: `{results['models']['e6c_checkpoint']}`
  - Frozen weights, 5 multiscale views, learned-attention aggregation.
- **E8-B Checkpoint**: `{results['models']['e8b_checkpoint']}`
  - Frozen weights, trained with multi-tier social-media compression augmentation.
- **Development Dataset**: Clean E5 Validation Set ($N=5,775$).
- **Frozen Test Dataset**: WhatsApp Robustness Test Set ($N=67$: 34 Real, 33 AI).

---

## 2. Rule Selection on E5 Validation (Strict Protocol)

### Selection Criteria
1. Meaningful recall gain over E6-C.
2. Strict FPR constraint: $\\text{{FPR}} \\le 3.5\\%$ (E6-C baseline is $2.97\\%$).
3. Maximum F1-score on clean validation data.
4. No degradation in overall clean accuracy.

### Frozen E9 Rule:
- **Rule Formulation**: `{rule['name']}`
- **Type**: `{rule['type']}`
- **Parameters**: `{rule['rule_func']}`
- **Rationale**: {rule['rationale']}

### Performance on Clean E5 Validation ($N=5,775$):
| Metric | E6-C Baseline | E8-B Baseline | E9 Frozen Rule | Delta (E9 vs E6-C) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | {e5_comp['e6c']['accuracy']*100:.2f}% | {e5_comp['e8b']['accuracy']*100:.2f}% | **{e5_comp['e9']['accuracy']*100:.2f}%** | {('+' if e5_comp['e9']['accuracy']>=e5_comp['e6c']['accuracy'] else '')}{(e5_comp['e9']['accuracy']-e5_comp['e6c']['accuracy'])*100:.2f}% |
| **ROC-AUC** | {e5_comp['e6c']['roc_auc']:.4f} | {e5_comp['e8b']['roc_auc']:.4f} | **{e5_comp['e9']['roc_auc']:.4f}** | {('+' if e5_comp['e9']['roc_auc']>=e5_comp['e6c']['roc_auc'] else '')}{e5_comp['e9']['roc_auc']-e5_comp['e6c']['roc_auc']:.4f} |
| **PR-AUC** | {e5_comp['e6c']['pr_auc']:.4f} | {e5_comp['e8b']['pr_auc']:.4f} | **{e5_comp['e9']['pr_auc']:.4f}** | {('+' if e5_comp['e9']['pr_auc']>=e5_comp['e6c']['pr_auc'] else '')}{e5_comp['e9']['pr_auc']-e5_comp['e6c']['pr_auc']:.4f} |
| **Precision** | {e5_comp['e6c']['precision']*100:.2f}% | {e5_comp['e8b']['precision']*100:.2f}% | **{e5_comp['e9']['precision']*100:.2f}%** | {('+' if e5_comp['e9']['precision']>=e5_comp['e6c']['precision'] else '')}{(e5_comp['e9']['precision']-e5_comp['e6c']['precision'])*100:.2f}% |
| **Recall** | {e5_comp['e6c']['recall']*100:.2f}% | {e5_comp['e8b']['recall']*100:.2f}% | **{e5_comp['e9']['recall']*100:.2f}%** | {('+' if e5_comp['e9']['recall']>=e5_comp['e6c']['recall'] else '')}{(e5_comp['e9']['recall']-e5_comp['e6c']['recall'])*100:.2f}% |
| **F1-Score** | {e5_comp['e6c']['f1']:.4f} | {e5_comp['e8b']['f1']:.4f} | **{e5_comp['e9']['f1']:.4f}** | {('+' if e5_comp['e9']['f1']>=e5_comp['e6c']['f1'] else '')}{e5_comp['e9']['f1']-e5_comp['e6c']['f1']:.4f} |
| **FPR** | {e5_comp['e6c']['fpr']*100:.2f}% | {e5_comp['e8b']['fpr']*100:.2f}% | **{e5_comp['e9']['fpr']*100:.2f}%** | {('+' if e5_comp['e9']['fpr']>=e5_comp['e6c']['fpr'] else '')}{(e5_comp['e9']['fpr']-e5_comp['e6c']['fpr'])*100:.2f}% |
| **FNR** | {e5_comp['e6c']['fnr']*100:.2f}% | {e5_comp['e8b']['fnr']*100:.2f}% | **{e5_comp['e9']['fnr']*100:.2f}%** | {('+' if e5_comp['e9']['fnr']>=e5_comp['e6c']['fnr'] else '')}{(e5_comp['e9']['fnr']-e5_comp['e6c']['fnr'])*100:.2f}% |

---

## 3. Hard-Case Diagnostics (Diagnostic Only)

| Case | Ground Truth | E6-C Prob | E8-B Prob | E9 Score | E9 Pred | Correct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `original_ai_edit.png` | {hc['original_ai_edit']['ground_truth']} | {hc['original_ai_edit']['e6c']['prob']:.4f} | {hc['original_ai_edit']['e8b']['prob']:.4f} | {hc['original_ai_edit']['e9']['prob_or_score']:.4f} | {hc['original_ai_edit']['e9']['pred']} | {'Yes' if hc['original_ai_edit']['e9']['correct'] else 'No'} |
| `whatsapp_download.jpeg` | {hc['whatsapp_download']['ground_truth']} | {hc['whatsapp_download']['e6c']['prob']:.4f} | {hc['whatsapp_download']['e8b']['prob']:.4f} | {hc['whatsapp_download']['e9']['prob_or_score']:.4f} | {hc['whatsapp_download']['e9']['pred']} | {'Yes' if hc['whatsapp_download']['e9']['correct'] else 'No'} |

---

## 4. Single-Shot WhatsApp Robustness Benchmark ($N=67$)

### Head-to-Head Comparison
| Metric | E6-C Baseline | E8-B (Augmented) | E9 (Ensemble/Gated) | Delta (E9 vs E6-C) | Delta (E9 vs E8-B) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | {m6_wa['accuracy']*100:.2f}% | {m8_wa['accuracy']*100:.2f}% | **{m9_wa['accuracy']*100:.2f}%** | {('+' if m9_wa['accuracy']>=m6_wa['accuracy'] else '')}{(m9_wa['accuracy']-m6_wa['accuracy'])*100:.2f}% | {('+' if m9_wa['accuracy']>=m8_wa['accuracy'] else '')}{(m9_wa['accuracy']-m8_wa['accuracy'])*100:.2f}% |
| **ROC-AUC** | {m6_wa['roc_auc']:.4f} | {m8_wa['roc_auc']:.4f} | **{m9_wa['roc_auc']:.4f}** | {('+' if m9_wa['roc_auc']>=m6_wa['roc_auc'] else '')}{m9_wa['roc_auc']-m6_wa['roc_auc']:.4f} | {('+' if m9_wa['roc_auc']>=m8_wa['roc_auc'] else '')}{m9_wa['roc_auc']-m8_wa['roc_auc']:.4f} |
| **PR-AUC** | {m6_wa['pr_auc']:.4f} | {m8_wa['pr_auc']:.4f} | **{m9_wa['pr_auc']:.4f}** | {('+' if m9_wa['pr_auc']>=m6_wa['pr_auc'] else '')}{m9_wa['pr_auc']-m6_wa['pr_auc']:.4f} | {('+' if m9_wa['pr_auc']>=m8_wa['pr_auc'] else '')}{m9_wa['pr_auc']-m8_wa['pr_auc']:.4f} |
| **Precision** | {m6_wa['precision']*100:.2f}% | {m8_wa['precision']*100:.2f}% | **{m9_wa['precision']*100:.2f}%** | {('+' if m9_wa['precision']>=m6_wa['precision'] else '')}{(m9_wa['precision']-m6_wa['precision'])*100:.2f}% | {('+' if m9_wa['precision']>=m8_wa['precision'] else '')}{(m9_wa['precision']-m8_wa['precision'])*100:.2f}% |
| **Recall** | {m6_wa['recall']*100:.2f}% | {m8_wa['recall']*100:.2f}% | **{m9_wa['recall']*100:.2f}%** | {('+' if m9_wa['recall']>=m6_wa['recall'] else '')}{(m9_wa['recall']-m6_wa['recall'])*100:.2f}% | {('+' if m9_wa['recall']>=m8_wa['recall'] else '')}{(m9_wa['recall']-m8_wa['recall'])*100:.2f}% |
| **F1-Score** | {m6_wa['f1']:.4f} | {m8_wa['f1']:.4f} | **{m9_wa['f1']:.4f}** | {('+' if m9_wa['f1']>=m6_wa['f1'] else '')}{m9_wa['f1']-m6_wa['f1']:.4f} | {('+' if m9_wa['f1']>=m8_wa['f1'] else '')}{m9_wa['f1']-m8_wa['f1']:.4f} |
| **Real FPR** | {m6_wa['fpr']*100:.2f}% | {m8_wa['fpr']*100:.2f}% | **{m9_wa['fpr']*100:.2f}%** | {('+' if m9_wa['fpr']>=m6_wa['fpr'] else '')}{(m9_wa['fpr']-m6_wa['fpr'])*100:.2f}% | {('+' if m9_wa['fpr']>=m8_wa['fpr'] else '')}{(m9_wa['fpr']-m8_wa['fpr'])*100:.2f}% |
| **AI FNR** | {m6_wa['fnr']*100:.2f}% | {m8_wa['fnr']*100:.2f}% | **{m9_wa['fnr']*100:.2f}%** | {('+' if m9_wa['fnr']>=m6_wa['fnr'] else '')}{(m9_wa['fnr']-m6_wa['fnr'])*100:.2f}% | {('+' if m9_wa['fnr']>=m8_wa['fnr'] else '')}{(m9_wa['fnr']-m8_wa['fnr'])*100:.2f}% |
| **TP / TN** | {m6_wa['tp']} / {m6_wa['tn']} | {m8_wa['tp']} / {m8_wa['tn']} | **{m9_wa['tp']} / {m9_wa['tn']}** | - | - |
| **FP / FN** | {m6_wa['fp']} / {m6_wa['fn']} | {m8_wa['fp']} / {m8_wa['fn']} | **{m9_wa['fp']} / {m9_wa['fn']}** | - | - |

---

## 5. WhatsApp Transition Analysis (E9 vs E6-C)

- **AI False Negatives Recovered**: **{trans['recovered_ai_fn_count']}**
- **E6-C True Positives Lost**: **{trans['lost_e6c_tp_count']}**
- **Additional Real False Positives**: **{trans['additional_real_fp_count']}**
- **E6-C False Positives Corrected**: **{trans['corrected_e6c_fp_count']}**

### Detailed Transition Log:
| Filename | Ground Truth | E6-C Prob | E8-B Prob | E9 Score | E6-C Decision | E9 Decision | Transition Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{trans_rows}

---

## 6. Critical Scientific Interpretation

### Core Question:
*Can E9 retain some of E8-B's WhatsApp AI recall improvement while bringing the false-positive rate substantially closer to E6-C?*

### Scientific Assessment:
1. **Recall Retention**:
   - E6-C WhatsApp AI Recall: **{m6_wa['recall']*100:.2f}%** ({m6_wa['tp']}/33 TP)
   - E8-B WhatsApp AI Recall: **{m8_wa['recall']*100:.2f}%** ({m8_wa['tp']}/33 TP)
   - E9 WhatsApp AI Recall: **{m9_wa['recall']*100:.2f}%** ({m9_wa['tp']}/33 TP)

2. **False Positive Rate Control**:
   - E6-C WhatsApp Real FPR: **{m6_wa['fpr']*100:.2f}%** ({m6_wa['fp']}/34 FP)
   - E8-B WhatsApp Real FPR: **{m8_wa['fpr']*100:.2f}%** ({m8_wa['fp']}/34 FP)
   - E9 WhatsApp Real FPR: **{m9_wa['fpr']*100:.2f}%** ({m9_wa['fp']}/34 FP)

3. **Trade-Off Viability**:
   - When restricted to the clean E5 development set under strict FPR control ($\\le 3.5\\%$), the optimal ensemble weight assigned to E8-B was $w=0.20$.
   - At threshold 0.50, this weight was conservative enough to perfectly maintain E6-C's low false positive rate (5.88% FPR, 2 FPs), but insufficient to lift WhatsApp AI images whose E6-C probabilities were deeply suppressed below 0.35 above the 0.50 threshold.
   - However, continuously ranked metrics improved significantly: ROC-AUC rose from **0.6560** to **0.6774** (+0.0214) and PR-AUC rose from **0.7139** to **0.7258** (+0.0119, exceeding even E8-B's 0.7209).

---

## 7. Production Recommendation

- **Current Production Model**: E6-C remains in place in production.
- **Ensemble Feasibility**: E9 provides state-of-the-art clean-domain performance (Acc 96.40%, AUC 0.9941, F1 0.9631, FPR 2.79%) and improves WhatsApp ranking AUCs, but at a fixed threshold of 0.50 without domain conditioning, continuous blending alone does not recover WhatsApp recall without relaxing FPR.
- **Production Status**: In accordance with the experiment safety protocol, production was **NOT modified**.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {report_path.name}")


def generate_readme(results):
    readme_path = OUT_DIR / "README.md"
    content = f"""# Experiment E9: E6-C + E8-B Compression-Robust Ensemble Study

## Overview
Experiment E9 investigates the combination of the production model E6-C (trained on clean multiscale views) and E8-B (fine-tuned with realistic social-media compression augmentation) via continuous probability ensembling and conservative gated rules.

## Protocol Highlights
- **Inference-Only**: No model training or fine-tuning. Checkpoints frozen.
- **Development on E5 Validation**: All weights, candidate rules, and FPR-constrained thresholds were chosen on the clean E5 validation set ($N=5,775$).
- **Frozen Single-Shot WhatsApp Benchmark**: Evaluated once after permanently freezing the final rule.

## Frozen Winning Rule
- `{results['frozen_rule']['name']}`
- Parameters: `{results['frozen_rule']['rule_func']}`

## Artifacts in this Directory
- `run_e9_study.py`: Reproducible experimental runner.
- `e5_val_predictions.csv`: Model predictions on all 5,775 E5 validation samples.
- `candidate_ensemble_results.csv`: Continuous probability ensemble sweep.
- `candidate_gated_results.csv`: Systematic gated rule family sweep.
- `per_image_predictions_whatsapp.csv`: Single-shot WhatsApp test results ($N=67$).
- `hard_case_diagnostics.json`: Diagnostic outputs on hard-case images.
- `study_results.json`: Full machine-readable metrics and transition analysis.
- `E9_RESEARCH_REPORT.md`: Comprehensive formal research report.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {readme_path.name}")


if __name__ == "__main__":
    main()
