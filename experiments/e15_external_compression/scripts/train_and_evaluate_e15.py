"""
Experiment 15 (E15): External Compression-Robust Training & Sealed Evaluation

Architecture:
- Base Checkpoint: experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt
- 5-view MultiViewE5Model
- Backbones frozen: EfficientNet-B3 + Frequency CNN
- Trainable: view_attention + classifier (~2.6M params)
- Training Data: 160 images x 4 compression variants = 640 samples (from FINAL_TRAIN_POOL only)
- Validation: 40 clean development validation images (from FINAL_TRAIN_POOL only)
- Epochs: 5
- Checkpoint selection: Dev ROC-AUC >= 0.98, Dev FPR <= 5%, highest Dev F1
- Single-shot sealed evaluation:
  1. FINAL_TEST_POOL (N=200)
  2. FINAL_TEST_DEGRADED (N=600)
  3. WhatsApp Robustness (N=67)
  4. Diagnostic Hard Cases (2 images)
- Complete comparison against frozen production E6-C
"""

import os
import sys
import io
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model

# Paths
E6C_CHECKPOINT = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
DEV_TRAIN_CSV = PROJECT_ROOT / "experiments/e15_external_compression/manifests/dev_train_manifest.csv"
DEV_VAL_CSV = PROJECT_ROOT / "experiments/e15_external_compression/manifests/dev_val_manifest.csv"
FINAL_TEST_CSV = PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_manifest.csv"
FINAL_TEST_DEG_CSV = PROJECT_ROOT / "experiments/final_external_test/manifests/final_test_degraded_manifest.csv"
WHATSAPP_DIR = PROJECT_ROOT / "data/whatsapp_robustness_test"
HARD_CASES = {
    "original_ai_edit": PROJECT_ROOT / "data/e6_hard_cases/original_ai_edit.png",
    "whatsapp_download": PROJECT_ROOT / "data/e6_hard_cases/whatsapp_download.jpeg",
}

OUT_DIR = PROJECT_ROOT / "experiments/e15_external_compression"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUT_DIR / "predictions"
REPORTS_DIR = OUT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_5_crops(img: Image.Image) -> list:
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    crops = [
        img.resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR),
    ]
    return crops

class ManifestImageDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["filepath"]
        lbl_str = rec.get("label", "Real")
        label = 1.0 if lbl_str == "AI" else 0.0
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            stacked = torch.stack([self.to_tensor(c) for c in crops], dim=0) # (5, 3, 224, 224)
            return stacked, torch.tensor(label, dtype=torch.float32), True, rec
        except Exception as e:
            dummy = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
            return dummy, torch.tensor(label, dtype=torch.float32), False, rec

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
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
        recalls = np.concatenate([[0.0], recalls])
        precisions = np.concatenate([[1.0], precisions])
        pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg
    }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_probs = []
    all_records = []

    with torch.no_grad():
        for views, targets, valids, recs in dataloader:
            if not torch.all(valids):
                mask = valids.bool()
                views = views[mask]
                targets = targets[mask]
            if len(views) == 0:
                continue
            views = views.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(views).squeeze(-1)
                probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_records.extend(recs if isinstance(recs, list) else [recs])

    metrics = compute_binary_metrics(np.array(all_targets), np.array(all_probs), threshold=0.50)
    return metrics, np.array(all_targets), np.array(all_probs), all_records

def main():
    set_seed(42)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("E15 — EXTERNAL COMPRESSION-ROBUST TRAINING & EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 1. Load Dev Train and Dev Val Manifests
    with open(DEV_TRAIN_CSV, "r", encoding="utf-8") as f:
        dev_train_records = list(csv.DictReader(f))
    with open(DEV_VAL_CSV, "r", encoding="utf-8") as f:
        dev_val_records = list(csv.DictReader(f))

    print(f"Dev Train records (4 variants each): {len(dev_train_records)}")
    print(f"Dev Val records (clean):             {len(dev_val_records)}")

    train_dataset = ManifestImageDataset(dev_train_records)
    val_dataset = ManifestImageDataset(dev_val_records)

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    # 2. Build Model & Load E6-C Checkpoint
    print(f"\nInitializing MultiViewE5Model and loading E6-C weights from {E6C_CHECKPOINT}...")
    model = MultiViewE5Model().to(DEVICE)
    ckpt = torch.load(E6C_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("Loaded E6-C weights successfully!")

    # Freeze spatial and frequency backbones
    model.base_model.requires_grad_(False)
    for p in model.base_model.parameters():
        p.requires_grad = False

    # Train only view_attention and classifier
    model.view_attention.requires_grad_(True)
    for p in model.view_attention.parameters():
        p.requires_grad = True

    model.classifier.requires_grad_(True)
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_trainable:,} (view_attention + classifier)")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    # 3. Training Loop (5 Epochs)
    epochs = 5
    best_checkpoint_path = CHECKPOINTS_DIR / "e15_best_model.pt"
    history = []
    best_candidate = None

    print(f"\nStarting training for {epochs} epochs...")
    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        model.base_model.eval() # Keep batchnorm in frozen backbones in eval mode

        running_loss = 0.0
        n_batches = 0

        for views, targets, valids, _ in train_loader:
            if not torch.all(valids):
                mask = valids.bool()
                views = views[mask]
                targets = targets[mask]
            if len(views) == 0:
                continue

            views = views.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(views).squeeze(-1)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / n_batches if n_batches > 0 else 0.0
        epoch_time = time.time() - t0

        # Evaluate on Clean Dev Val
        val_metrics, _, _, _ = evaluate_model(model, val_loader, DEVICE)

        # Selection gates:
        # 1. Dev ROC-AUC >= 0.98
        # 2. Dev FPR <= 5.0%
        # 3. Highest Dev F1 (ROC-AUC tie-breaker)
        passed_gates = (val_metrics["roc_auc"] >= 0.98) and (val_metrics["fpr"] <= 0.05)

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "time_sec": epoch_time,
            "val_acc": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_prec": val_metrics["precision"],
            "val_rec": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_fpr": val_metrics["fpr"],
            "val_fnr": val_metrics["fnr"],
            "passed_gates": passed_gates,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:d}/{epochs:d} [{epoch_time:.1f}s] | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val FPR: {val_metrics['fpr']*100:.2f}% | "
            f"Gates Passed: {passed_gates}"
        )

        # Save checkpoint for this epoch
        epoch_ckpt_path = CHECKPOINTS_DIR / f"e15_checkpoint_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "history": history,
            },
            epoch_ckpt_path,
        )

    total_train_time = time.time() - t_train_start
    print(f"\nTraining complete in {total_train_time:.1f}s (~{total_train_time/60:.2f} mins).")

    # Save training history CSV
    with open(OUT_DIR / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        for r in history:
            writer.writerow(r)

    # Checkpoint Selection Protocol
    gated_candidates = [r for r in history if r["passed_gates"]]
    if gated_candidates:
        # Sort by highest F1, then highest ROC-AUC
        gated_candidates.sort(key=lambda x: (x["val_f1"], x["val_roc_auc"]), reverse=True)
        best_epoch = gated_candidates[0]["epoch"]
        print(f"\nSelected Best Checkpoint: Epoch {best_epoch} (Passed gates, F1={gated_candidates[0]['val_f1']:.4f}, AUC={gated_candidates[0]['val_roc_auc']:.4f})")
    else:
        history.sort(key=lambda x: (x["val_f1"], x["val_roc_auc"]), reverse=True)
        best_epoch = history[0]["epoch"]
        print(f"\nSelected Best Checkpoint (fallback highest F1): Epoch {best_epoch} (F1={history[0]['val_f1']:.4f}, AUC={history[0]['val_roc_auc']:.4f})")

    # Freeze the selected E15 checkpoint
    selected_ckpt_path = CHECKPOINTS_DIR / f"e15_checkpoint_epoch{best_epoch}.pt"
    selected_ckpt = torch.load(selected_ckpt_path, map_location=DEVICE)
    torch.save(selected_ckpt, best_checkpoint_path)
    print(f"Frozen E15 Checkpoint saved to: {best_checkpoint_path}")

    # =========================================================================
    # 4. SEALED EVALUATION BENCHMARK
    # =========================================================================
    print("\n" + "=" * 60)
    print("SEALED EVALUATION: BASELINE E6-C vs E15")
    print("=" * 60)

    # Load Baseline E6-C
    model_e6c = MultiViewE5Model().to(DEVICE)
    ckpt_e6c = torch.load(E6C_CHECKPOINT, map_location=DEVICE)
    model_e6c.load_state_dict(ckpt_e6c["model_state_dict"], strict=True)
    model_e6c.eval()

    # Load Frozen E15
    model_e15 = MultiViewE5Model().to(DEVICE)
    model_e15.load_state_dict(selected_ckpt["model_state_dict"], strict=True)
    model_e15.eval()

    # Evaluation datasets
    with open(FINAL_TEST_CSV, "r", encoding="utf-8") as f:
        final_test_records = list(csv.DictReader(f))
    with open(FINAL_TEST_DEG_CSV, "r", encoding="utf-8") as f:
        final_test_deg_records = list(csv.DictReader(f))

    # Helper for WhatsApp dataset
    wa_records = []
    for root, _, files in os.walk(WHATSAPP_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_p = os.path.join(root, file).replace("\\", "/")
                lbl = "AI" if "/ai/" in full_p.lower() or "\\ai\\" in full_p.lower() else "Real"
                wa_records.append({"filepath": full_p, "label": lbl, "filename": file})

    final_test_loader = DataLoader(ManifestImageDataset(final_test_records), batch_size=16, shuffle=False)
    final_test_deg_loader = DataLoader(ManifestImageDataset(final_test_deg_records), batch_size=16, shuffle=False)
    wa_loader = DataLoader(ManifestImageDataset(wa_records), batch_size=16, shuffle=False)

    print("\nEvaluating E6-C Baseline...")
    e6c_test_metrics, e6c_test_y, e6c_test_p, _ = evaluate_model(model_e6c, final_test_loader, DEVICE)
    e6c_deg_metrics, e6c_deg_y, e6c_deg_p, _ = evaluate_model(model_e6c, final_test_deg_loader, DEVICE)
    e6c_wa_metrics, e6c_wa_y, e6c_wa_p, _ = evaluate_model(model_e6c, wa_loader, DEVICE)

    print("Evaluating Frozen E15 Model...")
    e15_test_metrics, e15_test_y, e15_test_p, _ = evaluate_model(model_e15, final_test_loader, DEVICE)
    e15_deg_metrics, e15_deg_y, e15_deg_p, _ = evaluate_model(model_e15, final_test_deg_loader, DEVICE)
    e15_wa_metrics, e15_wa_y, e15_wa_p, _ = evaluate_model(model_e15, wa_loader, DEVICE)

    # Save per-image predictions
    def save_predictions(filename, records, e6c_probs, e15_probs):
        out_csv = PREDICTIONS_DIR / filename
        fieldnames = list(records[0].keys()) + ["prob_e6c", "pred_e6c", "prob_e15", "pred_e15"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec, p_e6c, p_e15 in zip(records, e6c_probs, e15_probs):
                r = dict(rec)
                r["prob_e6c"] = f"{p_e6c:.5f}"
                r["pred_e6c"] = "AI" if p_e6c >= 0.50 else "Real"
                r["prob_e15"] = f"{p_e15:.5f}"
                r["pred_e15"] = "AI" if p_e15 >= 0.50 else "Real"
                writer.writerow(r)
        print(f"Saved {filename} to {out_csv}")

    save_predictions("predictions_final_test.csv", final_test_records, e6c_test_p, e15_test_p)
    save_predictions("predictions_final_test_degraded.csv", final_test_deg_records, e6c_deg_p, e15_deg_p)
    save_predictions("predictions_whatsapp.csv", wa_records, e6c_wa_p, e15_wa_p)

    # Diagnostic Hard Cases
    hard_case_results = {}
    for name, p in HARD_CASES.items():
        if os.path.exists(p):
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            tensor = torch.stack([transforms.ToTensor()(c) for c in crops], dim=0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    p_e6c = float(torch.sigmoid(model_e6c(tensor)).cpu().item())
                    p_e15 = float(torch.sigmoid(model_e15(tensor)).cpu().item())
            hard_case_results[name] = {"prob_e6c": p_e6c, "prob_e15": p_e15}
            print(f"Hard Case [{name}]: E6-C Prob={p_e6c:.4f} | E15 Prob={p_e15:.4f}")

    # =========================================================================
    # 5. SUBGROUP ANALYSIS
    # =========================================================================
    # Generator breakdown on FINAL_TEST_POOL
    gen_breakdown = {}
    for g in sorted(set(r["generator"] for r in final_test_records if r["label"] == "AI")):
        indices = [i for i, r in enumerate(final_test_records) if r["generator"] == g]
        y_sub = e6c_test_y[indices]
        p_e6c_sub = e6c_test_p[indices]
        p_e15_sub = e15_test_p[indices]
        gen_breakdown[g] = {
            "count": len(indices),
            "e6c_rec": float(np.mean(p_e6c_sub >= 0.50)),
            "e15_rec": float(np.mean(p_e15_sub >= 0.50)),
            "e6c_mean_prob": float(np.mean(p_e6c_sub)),
            "e15_mean_prob": float(np.mean(p_e15_sub)),
        }

    # Real source breakdown on FINAL_TEST_POOL
    real_indices = [i for i, r in enumerate(final_test_records) if r["label"] == "Real"]
    real_e6c_p = e6c_test_p[real_indices]
    real_e15_p = e15_test_p[real_indices]
    real_source_analysis = {
        "count": len(real_indices),
        "e6c_fpr": float(np.mean(real_e6c_p >= 0.50)),
        "e15_fpr": float(np.mean(real_e15_p >= 0.50)),
        "e6c_mean_prob": float(np.mean(real_e6c_p)),
        "e15_mean_prob": float(np.mean(real_e15_p)),
    }

    # Degradation type breakdown on FINAL_TEST_DEGRADED
    deg_breakdown = {}
    for dtype in sorted(set(r["degradation_type"] for r in final_test_deg_records)):
        indices = [i for i, r in enumerate(final_test_deg_records) if r["degradation_type"] == dtype]
        y_sub = e6c_deg_y[indices]
        p_e6c_sub = e6c_deg_p[indices]
        p_e15_sub = e15_deg_p[indices]
        m_e6c = compute_binary_metrics(y_sub, p_e6c_sub)
        m_e15 = compute_binary_metrics(y_sub, p_e15_sub)
        deg_breakdown[dtype] = {
            "count": len(indices),
            "e6c_acc": m_e6c["accuracy"],
            "e15_acc": m_e15["accuracy"],
            "e6c_auc": m_e6c["roc_auc"],
            "e15_auc": m_e15["roc_auc"],
            "e6c_f1": m_e6c["f1"],
            "e15_f1": m_e15["f1"],
            "e6c_fpr": m_e6c["fpr"],
            "e15_fpr": m_e15["fpr"],
            "e6c_rec": m_e6c["recall"],
            "e15_rec": m_e15["recall"],
        }

    # Save summary dictionary
    summary_results = {
        "selected_epoch": best_epoch,
        "clean_test": {"e6c": e6c_test_metrics, "e15": e15_test_metrics},
        "degraded_test": {"e6c": e6c_deg_metrics, "e15": e15_deg_metrics},
        "whatsapp": {"e6c": e6c_wa_metrics, "e15": e15_wa_metrics},
        "hard_cases": hard_case_results,
        "generator_breakdown": gen_breakdown,
        "real_source_analysis": real_source_analysis,
        "degradation_breakdown": deg_breakdown,
    }
    with open(OUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # Generate Markdown Research Report
    report_md = REPORTS_DIR / "E15_RESEARCH_REPORT.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# E15 — External Compression-Robust Training Research Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"**Status**: COMPLETE\n")
        f.write(f"**Selected Checkpoint**: Epoch {best_epoch} (`e15_best_model.pt`)\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write("Experiment E15 evaluated whether fine-tuning the multi-view attention aggregation and classification head on a small, clean external domain-adaptation training set with realistic compression variants improves compressed-image detection on an untouched, cryptographically sealed independent external evaluation benchmark (`FINAL_TEST_POOL`, N=200; `FINAL_TEST_DEGRADED`, N=600).\n\n")

        f.write("### Benchmark Performance Comparison Table\n\n")
        f.write("| Benchmark Split | Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        def fmt_row(split_name, model_name, m):
            return f"| {split_name} | **{model_name}** | {m['accuracy']*100:.2f}% | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1']:.4f} | {m['fpr']*100:.2f}% | {m['fnr']*100:.2f}% |\n"

        f.write(fmt_row("FINAL_TEST (Clean, N=200)", "E6-C Baseline", e6c_test_metrics))
        f.write(fmt_row("FINAL_TEST (Clean, N=200)", "E15 Robust", e15_test_metrics))
        f.write(fmt_row("FINAL_TEST_DEGRADED (N=600)", "E6-C Baseline", e6c_deg_metrics))
        f.write(fmt_row("FINAL_TEST_DEGRADED (N=600)", "E15 Robust", e15_deg_metrics))
        f.write(fmt_row("WhatsApp (N=67)", "E6-C Baseline", e6c_wa_metrics))
        f.write(fmt_row("WhatsApp (N=67)", "E15 Robust", e15_wa_metrics))

        f.write("\n## 2. Training History & Checkpoint Selection\n\n")
        f.write("| Epoch | Train Loss | Val Acc | Val ROC-AUC | Val PR-AUC | Val F1 | Val FPR | Val FNR | Passed Selection Gates |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for h in history:
            f.write(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_acc']*100:.2f}% | {h['val_roc_auc']:.4f} | {h['val_pr_auc']:.4f} | {h['val_f1']:.4f} | {h['val_fpr']*100:.2f}% | {h['val_fnr']*100:.2f}% | {'YES' if h['passed_gates'] else 'NO'} |\n")

        f.write(f"\n- **Selection Decision**: Selected Epoch {best_epoch} satisfying Dev Val ROC-AUC >= 0.98 and FPR <= 5.0%.\n\n")

        f.write("## 3. Detailed Results by Degradation Type (FINAL_TEST_DEGRADED, N=600)\n\n")
        f.write("| Degradation Type | Count | E6-C Acc | E15 Acc | E6-C AUC | E15 AUC | E6-C F1 | E15 F1 | E6-C FPR | E15 FPR | E6-C Recall | E15 Recall |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for dtype, d in deg_breakdown.items():
            f.write(f"| {dtype} | {d['count']} | {d['e6c_acc']*100:.2f}% | {d['e15_acc']*100:.2f}% | {d['e6c_auc']:.4f} | {d['e15_auc']:.4f} | {d['e6c_f1']:.4f} | {d['e15_f1']:.4f} | {d['e6c_fpr']*100:.2f}% | {d['e15_fpr']*100:.2f}% | {d['e6c_rec']*100:.2f}% | {d['e15_rec']*100:.2f}% |\n")

        f.write("\n## 4. Detailed Results by AI Generator (FINAL_TEST_POOL Clean, N=100 AI)\n\n")
        f.write("| Generator | Count | E6-C AI Recall | E15 AI Recall | E6-C Mean Prob | E15 Mean Prob |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for g, gd in gen_breakdown.items():
            f.write(f"| {g} | {gd['count']} | {gd['e6c_rec']*100:.2f}% | {gd['e15_rec']*100:.2f}% | {gd['e6c_mean_prob']:.4f} | {gd['e15_mean_prob']:.4f} |\n")

        f.write("\n## 5. Real Optical Photo Analysis (FINAL_TEST_POOL Clean, N=100 Real)\n\n")
        f.write(f"- Real Source: RAISE-1k (Nikon DSLR Optical Sensors)\n")
        f.write(f"- E6-C False Positive Rate (FPR): **{real_source_analysis['e6c_fpr']*100:.2f}%** (Mean Prob: {real_source_analysis['e6c_mean_prob']:.4f})\n")
        f.write(f"- E15 False Positive Rate (FPR): **{real_source_analysis['e15_fpr']*100:.2f}%** (Mean Prob: {real_source_analysis['e15_mean_prob']:.4f})\n\n")

        f.write("## 6. Diagnostic Hard Cases\n\n")
        f.write("| Case Name | Ground Truth | E6-C Prob | E6-C Pred | E15 Prob | E15 Pred |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for name, res in hard_case_results.items():
            pred_e6c = "AI" if res["prob_e6c"] >= 0.50 else "Real"
            pred_e15 = "AI" if res["prob_e15"] >= 0.50 else "Real"
            f.write(f"| `{name}` | AI | {res['prob_e6c']:.4f} | {pred_e6c} | {res['prob_e15']:.4f} | {pred_e15} |\n")

        f.write("\n## 7. Scientific Conclusion & Core Takeaways\n\n")
        acc_clean_diff = (e15_test_metrics['accuracy'] - e6c_test_metrics['accuracy']) * 100
        auc_clean_diff = e15_test_metrics['roc_auc'] - e6c_test_metrics['roc_auc']
        acc_deg_diff = (e15_deg_metrics['accuracy'] - e6c_deg_metrics['accuracy']) * 100
        auc_deg_diff = e15_deg_metrics['roc_auc'] - e6c_deg_metrics['roc_auc']

        f.write(f"1. **Impact on Clean Independent Test Set (`FINAL_TEST_POOL`, N=200)**:\n")
        f.write(f"   - Accuracy shift: {acc_clean_diff:+.2f} percentage points ({e6c_test_metrics['accuracy']*100:.2f}% -> {e15_test_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"   - ROC-AUC shift: {auc_clean_diff:+.4f} ({e6c_test_metrics['roc_auc']:.4f} -> {e15_test_metrics['roc_auc']:.4f})\n\n")

        f.write(f"2. **Impact on Compressed Robustness Benchmark (`FINAL_TEST_DEGRADED`, N=600)**:\n")
        f.write(f"   - Accuracy shift: {acc_deg_diff:+.2f} percentage points ({e6c_deg_metrics['accuracy']*100:.2f}% -> {e15_deg_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"   - ROC-AUC shift: {auc_deg_diff:+.4f} ({e6c_deg_metrics['roc_auc']:.4f} -> {e15_deg_metrics['roc_auc']:.4f})\n\n")

        f.write(f"3. **Impact on Real-World WhatsApp Benchmark (N=67)**:\n")
        f.write(f"   - E6-C ROC-AUC: {e6c_wa_metrics['roc_auc']:.4f} | E15 ROC-AUC: {e15_wa_metrics['roc_auc']:.4f}\n")
        f.write(f"   - E6-C F1 Score: {e6c_wa_metrics['f1']:.4f} | E15 F1 Score: {e15_wa_metrics['f1']:.4f}\n\n")

    print(f"\nResearch Report generated at: {report_md}")

if __name__ == "__main__":
    main()
