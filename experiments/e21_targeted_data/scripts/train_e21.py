"""
Experiment 21 (E21) Model Training: Targeted Data-Balancing Forensic Detector
DeepVision-Forensics Research Pipeline

Protocol & Constraints:
- Starting Checkpoint: experiments/e20_training/checkpoints/e20_best_model.pt
- Exact Architecture: 5-view MultiViewE5Model
    * Spatial branch: EfficientNet-B3 (ImageNet-normalized) -> 1536-D
    * Frequency branch: Standardized 2D FFT Log-Magnitude CNN -> 256-D
    * Fused view embedding: 1792-D
    * Multi-view attention module: 1792 -> 128 -> 1
    * Classifier head: 1792 -> 512 -> 128 -> 1
- Zero architectural changes: No LayerNorm, no gating, no consistency loss, no quality conditioning.
- Staged Training Schedule:
    * Phase 1: Freeze backbones. Train view_attention + classifier head (lr=1e-4) for 2 epochs.
    * Phase 2: Unfreeze backbones. Train backbones (lr=1e-5) + head (lr=1e-4) for up to 3 epochs.
    * Total max epochs = 5.
- Hardware & Batch Setup:
    * Batch size = 4, grad_accum = 2 (effective batch size = 8)
    * PyTorch AMP FP16 (autocast + GradScaler)
    * AdamW, weight decay = 1e-4
    * Loss: BCEWithLogitsLoss
    * Fixed threshold: 0.50
- Checkpoint Selection:
    * Strictly governed by E21 Dev split: Primary F1 -> ROC-AUC -> Real FPR -> AI Recall.
    * Best checkpoint saved to experiments/e21_targeted_data/checkpoints/e21_best_model.pt
"""

import os
import sys
import time
import json
import csv
import random
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
E20_CHECKPOINT = PROJECT_ROOT / "experiments/e20_training/checkpoints/e20_best_model.pt"
TRAIN_MANIFEST_CSV = PROJECT_ROOT / "data/e21_targeted_data/manifests/e21_train_manifest.csv"
DEV_MANIFEST_CSV = PROJECT_ROOT / "data/e21_targeted_data/manifests/e21_dev_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data/e21_targeted_data"

OUT_DIR = PROJECT_ROOT / "experiments/e21_targeted_data"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUT_DIR / "predictions"
METRICS_DIR = OUT_DIR / "metrics"

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
    return [
        img.resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR),
    ]

class E21Dataset(Dataset):
    def __init__(self, records: List[Dict], data_root: Path):
        self.records = records
        self.data_root = data_root
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_rel_path = rec["filepath"]
        full_path = self.data_root / img_rel_path
        label = 1.0 if rec["label"] == "AI" else 0.0
        try:
            with Image.open(full_path) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            stacked = torch.stack([self.to_tensor(c) for c in crops], dim=0) # (5, 3, 224, 224)
            return stacked, torch.tensor(label, dtype=torch.float32), idx
        except Exception as e:
            print(f"Error reading image {full_path}: {e}")
            dummy = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
            return dummy, torch.tensor(label, dtype=torch.float32), idx

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict:
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

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos": n_pos, "n_neg": n_neg, "n_total": n_total
    }

def evaluate_model(model: nn.Module, loader: DataLoader, records: List[Dict]) -> Tuple[Dict, Dict, List[Dict]]:
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x_views, y_target, _ in loader:
            x_views = x_views.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(x_views).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(y_target.numpy().tolist())

    y_prob = np.array(all_probs)
    y_true = np.array(all_targets)

    overall_metrics = compute_binary_metrics(y_true, y_prob)

    # Subgroup breakdowns
    subgroup_metrics = {}
    generators = sorted(list(set(r.get("generator", "none") for r in records if r.get("generator") != "none")))
    devices = sorted(list(set(r.get("device_family", "none") for r in records if r.get("device_family") != "none")))
    compressions = sorted(list(set(r.get("variant_type", "clean") for r in records)))

    for gen in generators:
        mask = np.array([r.get("generator") == gen for r in records])
        if np.sum(mask) > 0:
            m = compute_binary_metrics(y_true[mask], y_prob[mask])
            subgroup_metrics[f"ai_gen_{gen}_recall"] = m["recall"]
            subgroup_metrics[f"ai_gen_{gen}_f1"] = m["f1"]
            subgroup_metrics[f"ai_gen_{gen}_n"] = int(np.sum(mask))

    for dev in devices:
        mask = np.array([r.get("device_family") == dev for r in records])
        if np.sum(mask) > 0:
            m = compute_binary_metrics(y_true[mask], y_prob[mask])
            subgroup_metrics[f"real_dev_{dev}_fpr"] = m["fpr"]
            subgroup_metrics[f"real_dev_{dev}_acc"] = m["accuracy"]
            subgroup_metrics[f"real_dev_{dev}_n"] = int(np.sum(mask))

    for comp in compressions:
        mask = np.array([r.get("variant_type") == comp for r in records])
        if np.sum(mask) > 0:
            m = compute_binary_metrics(y_true[mask], y_prob[mask])
            subgroup_metrics[f"comp_{comp}_acc"] = m["accuracy"]
            subgroup_metrics[f"comp_{comp}_f1"] = m["f1"]
            subgroup_metrics[f"comp_{comp}_n"] = int(np.sum(mask))

    predictions = []
    for i, r in enumerate(records):
        pred_row = dict(r)
        pred_row["probability"] = float(y_prob[i])
        pred_row["prediction"] = 1 if y_prob[i] >= 0.50 else 0
        pred_row["correct"] = int(pred_row["prediction"] == int(y_true[i]))
        predictions.append(pred_row)

    return overall_metrics, subgroup_metrics, predictions

def main():
    set_seed(42)

    for d in [CHECKPOINTS_DIR, PREDICTIONS_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STARTING E21 TARGETED DATA-BALANCING TRAINING")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==================================================")

    # 1. Load Manifests
    print(f"Loading E21 Train Manifest from {TRAIN_MANIFEST_CSV}...")
    with open(TRAIN_MANIFEST_CSV, "r", encoding="utf-8") as f:
        train_records = list(csv.DictReader(f))
    print(f"Loaded {len(train_records)} training records (Real: {sum(1 for r in train_records if r['label'] == 'Real')}, AI: {sum(1 for r in train_records if r['label'] == 'AI')})")

    print(f"Loading E21 Dev Manifest from {DEV_MANIFEST_CSV}...")
    with open(DEV_MANIFEST_CSV, "r", encoding="utf-8") as f:
        dev_records = list(csv.DictReader(f))
    print(f"Loaded {len(dev_records)} dev records (Real: {sum(1 for r in dev_records if r['label'] == 'Real')}, AI: {sum(1 for r in dev_records if r['label'] == 'AI')})")

    # 2. Datasets & Dataloaders
    train_dataset = E21Dataset(train_records, DATA_ROOT)
    dev_dataset = E21Dataset(dev_records, DATA_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # 3. Model Initialization (From E20 Best Checkpoint)
    print(f"\nInitializing 5-view MultiViewE5Model from E20 Checkpoint: {E20_CHECKPOINT}...")
    model = MultiViewE5Model()
    
    if not E20_CHECKPOINT.exists():
        raise FileNotFoundError(f"E20 Checkpoint not found at {E20_CHECKPOINT}")

    ckpt = torch.load(E20_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"E20 Checkpoint loaded strictly with 0 missing and 0 unexpected keys.")

    model = model.to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()

    # Initial Zero-Shot Dev Evaluation (Before E21 Training)
    print("\n--- Zero-Shot Evaluation of Starting E20 Weights on E21 Dev ---")
    base_m, base_sub, _ = evaluate_model(model, dev_loader, dev_records)
    print(f"E20 on E21 Dev: AUC={base_m['roc_auc']:.4f} | Acc={base_m['accuracy']:.4f} | F1={base_m['f1']:.4f} | Prec={base_m['precision']:.4f} | Rec={base_m['recall']:.4f} | FPR={base_m['fpr']:.4f}")
    for k, v in base_sub.items():
        if "recall" in k or "fpr" in k:
            print(f"  {k}: {v:.4f}")

    metrics_history = []
    best_dev_f1 = base_m["f1"]
    best_dev_auc = base_m["roc_auc"]
    best_dev_fpr = base_m["fpr"]
    best_dev_rec = base_m["recall"]
    best_epoch = 0

    total_training_start = time.time()
    grad_accum_steps = 2

    # ==================================================
    # PHASE 1: Head Only Fine-Tuning (Epochs 1-2)
    # ==================================================
    print("\n==================================================")
    print("PHASE 1: Freeze Backbones, Train View Attention + Head")
    print("Learning Rate: Head = 1e-4, Weight Decay = 1e-4")
    print("==================================================")

    for p in model.base_model.spatial_branch.parameters():
        p.requires_grad = False
    for p in model.base_model.frequency_branch.parameters():
        p.requires_grad = False
    for p in model.view_attention.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt_phase1 = torch.optim.AdamW(
        [
            {"params": model.view_attention.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4}
        ],
        weight_decay=1e-4
    )

    for epoch in range(1, 3):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)
        opt_phase1.zero_grad()

        print(f"\n--- Epoch {epoch}/5 [Phase 1: Head Only] ---")
        for batch_idx, (x_views, y_target, _) in enumerate(train_loader):
            x_views = x_views.to(DEVICE, non_blocking=True)
            y_target = y_target.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(x_views).squeeze(-1)
                loss = criterion(logits, y_target)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == n_batches:
                scaler.step(opt_phase1)
                scaler.update()
                opt_phase1.zero_grad()

            running_loss += loss.item()

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - epoch_start
                throughput = ((batch_idx + 1) * 4) / max(elapsed, 1.0)
                vram_mb = torch.cuda.memory_allocated() / 1e6
                print(f"  Batch {batch_idx+1}/{n_batches} | Loss: {loss.item():.4f} (Avg: {running_loss/(batch_idx+1):.4f}) | Throughput: {throughput:.1f} img/s | VRAM: {vram_mb:.0f} MB", flush=True)

        train_loss = running_loss / n_batches

        # Validation
        print(f"Evaluating Epoch {epoch} on E21 Dev...")
        dev_m, dev_sub, dev_preds = evaluate_model(model, dev_loader, dev_records)

        print(f"Epoch {epoch} Dev: AUC={dev_m['roc_auc']:.4f} | Acc={dev_m['accuracy']:.4f} | F1={dev_m['f1']:.4f} | Prec={dev_m['precision']:.4f} | Rec={dev_m['recall']:.4f} | FPR={dev_m['fpr']:.4f}")
        print(f"  SD1.3 Rec={dev_sub.get('ai_gen_Stable Diffusion 1.3_recall', 0):.4f} | Firefly Rec={dev_sub.get('ai_gen_Adobe Firefly 1_recall', 0):.4f} | Pixel FPR={dev_sub.get('real_dev_Google Pixel_fpr', 0):.4f} | WA Acc={dev_sub.get('comp_social_media_whatsapp_acc', 0):.4f}")

        # Checkpoint saving
        ckpt_path = CHECKPOINTS_DIR / f"e21_checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "phase": "phase1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt_phase1.state_dict(),
            "metrics": dev_m,
            "subgroups": dev_sub
        }, ckpt_path)

        # Checkpoint selection (Primary: F1, Tiebreakers: ROC-AUC, FPR, Recall)
        is_best = False
        if dev_m["f1"] > best_dev_f1:
            is_best = True
        elif dev_m["f1"] == best_dev_f1:
            if dev_m["roc_auc"] > best_dev_auc:
                is_best = True
            elif dev_m["roc_auc"] == best_dev_auc and dev_m["fpr"] < best_dev_fpr:
                is_best = True
            elif dev_m["roc_auc"] == best_dev_auc and dev_m["fpr"] == best_dev_fpr and dev_m["recall"] > best_dev_rec:
                is_best = True

        if is_best:
            best_dev_f1 = dev_m["f1"]
            best_dev_auc = dev_m["roc_auc"]
            best_dev_fpr = dev_m["fpr"]
            best_dev_rec = dev_m["recall"]
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "phase": "phase1",
                "model_state_dict": model.state_dict(),
                "metrics": dev_m,
                "subgroups": dev_sub
            }, CHECKPOINTS_DIR / "e21_best_model.pt")
            with open(PREDICTIONS_DIR / "e21_best_dev_predictions.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(dev_preds[0].keys()))
                writer.writeheader()
                writer.writerows(dev_preds)
            print(f"[*] NEW BEST CHECKPOINT SAVED at Epoch {epoch} (F1: {best_dev_f1:.4f}, AUC: {best_dev_auc:.4f})")

        metrics_history.append({
            "epoch": epoch,
            "phase": "phase1",
            "train_loss": train_loss,
            "dev_roc_auc": dev_m["roc_auc"],
            "dev_pr_auc": dev_m["pr_auc"],
            "dev_accuracy": dev_m["accuracy"],
            "dev_f1": dev_m["f1"],
            "dev_precision": dev_m["precision"],
            "dev_recall": dev_m["recall"],
            "dev_fpr": dev_m["fpr"],
            "dev_fnr": dev_m["fnr"],
            "tp": dev_m["tp"],
            "tn": dev_m["tn"],
            "fp": dev_m["fp"],
            "fn": dev_m["fn"],
            "sd13_recall": dev_sub.get("ai_gen_Stable Diffusion 1.3_recall", 0.0),
            "sd14_recall": dev_sub.get("ai_gen_Stable Diffusion 1.4_recall", 0.0),
            "sd2_recall": dev_sub.get("ai_gen_Stable Diffusion 2_recall", 0.0),
            "firefly_recall": dev_sub.get("ai_gen_Adobe Firefly 1_recall", 0.0),
            "dalle2_recall": dev_sub.get("ai_gen_OpenAI DALL-E 2_recall", 0.0),
            "pixel_fpr": dev_sub.get("real_dev_Google Pixel_fpr", 0.0),
            "iphone_fpr": dev_sub.get("real_dev_Apple iPhone_fpr", 0.0),
            "whatsapp_f1": dev_sub.get("comp_social_media_whatsapp_f1", 0.0),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6)
        })

    # ==================================================
    # PHASE 2: Unfreeze Backbones (Epochs 3-5)
    # ==================================================
    print("\n==================================================")
    print("PHASE 2: Unfreeze Backbones and Full End-to-End Fine-Tuning")
    print("Learning Rates: Backbones = 1e-5, Head = 1e-4")
    print("==================================================")

    for p in model.base_model.spatial_branch.parameters():
        p.requires_grad = True
    for p in model.base_model.frequency_branch.parameters():
        p.requires_grad = True
    for p in model.view_attention.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt_phase2 = torch.optim.AdamW(
        [
            {"params": model.base_model.spatial_branch.parameters(), "lr": 1e-5},
            {"params": model.base_model.frequency_branch.parameters(), "lr": 1e-5},
            {"params": model.view_attention.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-4}
        ],
        weight_decay=1e-4
    )

    for epoch in range(3, 6):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)
        opt_phase2.zero_grad()

        print(f"\n--- Epoch {epoch}/5 [Phase 2: Full End-to-End] ---")
        for batch_idx, (x_views, y_target, _) in enumerate(train_loader):
            x_views = x_views.to(DEVICE, non_blocking=True)
            y_target = y_target.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(x_views).squeeze(-1)
                loss = criterion(logits, y_target)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == n_batches:
                scaler.step(opt_phase2)
                scaler.update()
                opt_phase2.zero_grad()

            running_loss += loss.item()

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - epoch_start
                throughput = ((batch_idx + 1) * 4) / max(elapsed, 1.0)
                vram_mb = torch.cuda.memory_allocated() / 1e6
                print(f"  Batch {batch_idx+1}/{n_batches} | Loss: {loss.item():.4f} (Avg: {running_loss/(batch_idx+1):.4f}) | Throughput: {throughput:.1f} img/s | VRAM: {vram_mb:.0f} MB", flush=True)

        train_loss = running_loss / n_batches

        # Validation
        print(f"Evaluating Epoch {epoch} on E21 Dev...")
        dev_m, dev_sub, dev_preds = evaluate_model(model, dev_loader, dev_records)

        print(f"Epoch {epoch} Dev: AUC={dev_m['roc_auc']:.4f} | Acc={dev_m['accuracy']:.4f} | F1={dev_m['f1']:.4f} | Prec={dev_m['precision']:.4f} | Rec={dev_m['recall']:.4f} | FPR={dev_m['fpr']:.4f}")
        print(f"  SD1.3 Rec={dev_sub.get('ai_gen_Stable Diffusion 1.3_recall', 0):.4f} | Firefly Rec={dev_sub.get('ai_gen_Adobe Firefly 1_recall', 0):.4f} | Pixel FPR={dev_sub.get('real_dev_Google Pixel_fpr', 0):.4f} | WA Acc={dev_sub.get('comp_social_media_whatsapp_acc', 0):.4f}")

        # Checkpoint saving
        ckpt_path = CHECKPOINTS_DIR / f"e21_checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "phase": "phase2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt_phase2.state_dict(),
            "metrics": dev_m,
            "subgroups": dev_sub
        }, ckpt_path)

        # Checkpoint selection (Primary: F1, Tiebreakers: ROC-AUC, FPR, Recall)
        is_best = False
        if dev_m["f1"] > best_dev_f1:
            is_best = True
        elif dev_m["f1"] == best_dev_f1:
            if dev_m["roc_auc"] > best_dev_auc:
                is_best = True
            elif dev_m["roc_auc"] == best_dev_auc and dev_m["fpr"] < best_dev_fpr:
                is_best = True
            elif dev_m["roc_auc"] == best_dev_auc and dev_m["fpr"] == best_dev_fpr and dev_m["recall"] > best_dev_rec:
                is_best = True

        if is_best:
            best_dev_f1 = dev_m["f1"]
            best_dev_auc = dev_m["roc_auc"]
            best_dev_fpr = dev_m["fpr"]
            best_dev_rec = dev_m["recall"]
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "phase": "phase2",
                "model_state_dict": model.state_dict(),
                "metrics": dev_m,
                "subgroups": dev_sub
            }, CHECKPOINTS_DIR / "e21_best_model.pt")
            with open(PREDICTIONS_DIR / "e21_best_dev_predictions.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(dev_preds[0].keys()))
                writer.writeheader()
                writer.writerows(dev_preds)
            print(f"[*] NEW BEST CHECKPOINT SAVED at Epoch {epoch} (F1: {best_dev_f1:.4f}, AUC: {best_dev_auc:.4f})")

        metrics_history.append({
            "epoch": epoch,
            "phase": "phase2",
            "train_loss": train_loss,
            "dev_roc_auc": dev_m["roc_auc"],
            "dev_pr_auc": dev_m["pr_auc"],
            "dev_accuracy": dev_m["accuracy"],
            "dev_f1": dev_m["f1"],
            "dev_precision": dev_m["precision"],
            "dev_recall": dev_m["recall"],
            "dev_fpr": dev_m["fpr"],
            "dev_fnr": dev_m["fnr"],
            "tp": dev_m["tp"],
            "tn": dev_m["tn"],
            "fp": dev_m["fp"],
            "fn": dev_m["fn"],
            "sd13_recall": dev_sub.get("ai_gen_Stable Diffusion 1.3_recall", 0.0),
            "sd14_recall": dev_sub.get("ai_gen_Stable Diffusion 1.4_recall", 0.0),
            "sd2_recall": dev_sub.get("ai_gen_Stable Diffusion 2_recall", 0.0),
            "firefly_recall": dev_sub.get("ai_gen_Adobe Firefly 1_recall", 0.0),
            "dalle2_recall": dev_sub.get("ai_gen_OpenAI DALL-E 2_recall", 0.0),
            "pixel_fpr": dev_sub.get("real_dev_Google Pixel_fpr", 0.0),
            "iphone_fpr": dev_sub.get("real_dev_Apple iPhone_fpr", 0.0),
            "whatsapp_f1": dev_sub.get("comp_social_media_whatsapp_f1", 0.0),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6)
        })

    # Save metrics CSV
    csv_file = OUT_DIR / "e21_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_history)
    print(f"\nTraining metrics saved to {csv_file}")

    total_time = time.time() - total_training_start
    print(f"\n==================================================")
    print(f"E21 TRAINING FINISHED IN {total_time/60:.2f} MINUTES")
    print(f"Best Model Selected at Epoch {best_epoch}: Dev F1 = {best_dev_f1:.4f}, AUC = {best_dev_auc:.4f}, FPR = {best_dev_fpr:.4f}, Recall = {best_dev_rec:.4f}")
    print(f"Best Checkpoint: {CHECKPOINTS_DIR / 'e21_best_model.pt'}")
    print(f"==================================================")

    # Generate E21_TRAINING_REPORT.md
    with open(OUT_DIR / "E21_TRAINING_REPORT.md", "w", encoding="utf-8") as f:
        f.write(f"""# E21 Training & Validation Experiment Report

**Experiment**: E21 Targeted Data-Balancing Fine-Tuning  
**Date**: {time.strftime('%B %d, %Y')}  
**Status**: Completed and Frozen  
**Starting Checkpoint**: `experiments/e20_training/checkpoints/e20_best_model.pt`  
**Best Checkpoint Selected**: `experiments/e21_targeted_data/checkpoints/e21_best_model.pt` (Epoch {best_epoch})  
**Evaluation Threshold**: Strictly 0.50  

---

## 1. Executive Summary & Training Progression

The objective of **Experiment 21 (E21)** was to fine-tune the high-performing E20 multi-scale forensic detector using a targeted data-balancing curriculum:
1. **Restoring Legacy Generative Model Sensitivity**: Direct exposure to early latent diffusion models (Stable Diffusion 1.3, 1.4, 2, Adobe Firefly 1, DALL-E 2).
2. **Defocus & Low-Edge Invariance**: Incorporating natural shallow depth-of-field, indoor casual, and low-light smartphone photography.
3. **Preserving Modern Smartphone Invariance**: Retaining high sensitivity on FLUX.1 [dev], Google Gemini, DALL-E 3, SDXL, and Pixel/iPhone/Samsung mobile captures.

### Training Progression Table

| Epoch | Phase | Train Loss | Dev Acc | Dev ROC-AUC | Dev F1 | Dev FPR | Dev AI Recall | SD 1.3 Rec | Firefly Rec | Pixel FPR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
""")
        for m in metrics_history:
            f.write(f"| {m['epoch']} | {m['phase']} | {m['train_loss']:.4f} | {m['dev_accuracy']*100:.2f}% | {m['dev_roc_auc']:.4f} | {m['dev_f1']:.4f} | {m['dev_fpr']*100:.2f}% | {m['dev_recall']*100:.2f}% | {m['sd13_recall']*100:.1f}% | {m['firefly_recall']*100:.1f}% | {m['pixel_fpr']*100:.1f}% |\n")

        f.write(f"""
---

## 2. Checkpoint Selection

Selected **Epoch {best_epoch}** as `e21_best_model.pt` strictly following pre-registered E21 Dev selection criteria:
- **Primary (Dev F1)**: **{best_dev_f1:.4f}**
- **Tie-breaker (Dev ROC-AUC)**: **{best_dev_auc:.4f}**
- **False Alarm Rate (Dev FPR)**: **{best_dev_fpr*100:.2f}%**
- **Detection Sensitivity (Dev Recall)**: **{best_dev_rec*100:.2f}%**
""")

if __name__ == "__main__":
    main()
