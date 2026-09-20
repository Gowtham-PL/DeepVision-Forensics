"""
Experiment 20 (E20) Model Training: High-Generalization Forensic Detector

Architecture:
- Base Checkpoint: experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt
- 5-view MultiViewE5Model:
    1 Global (224x224)
    4 Corner crops (60% coverage, resized to 224x224)
    Spatial branch: EfficientNet-B3 (ImageNet-normalized)
    Frequency branch: Standardized 2D FFT Log-Magnitude CNN
    Multi-view attention module: 1792 -> 128 -> 1
    Classifier head: 1792 -> 512 -> 128 -> 1
- Staged Training Strategy:
    Phase 1: Freeze backbones. Train view_attention + classifier head (lr=1e-4) for 2 epochs.
    Phase 2: Unfreeze backbones. Train backbones (lr=1e-5) + head (lr=1e-4) for up to 3 epochs.
    Total Max Epochs: 5
- Hardware & Batch Setup:
    batch_size = 4, grad_accum = 2 (effective batch_size = 8)
    num_workers = 2, persistent_workers = True, pin_memory = True
    AMP enabled (FP16 autocast + GradScaler)
    Threshold: exactly 0.50
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
E6C_CHECKPOINT = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
TRAIN_MANIFEST_CSV = PROJECT_ROOT / "data/e20_training/manifests/e20_train_manifest.csv"
DEV_MANIFEST_CSV = PROJECT_ROOT / "data/e20_training/manifests/e20_dev_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data/e20_training"

OUT_DIR = PROJECT_ROOT / "experiments/e20_training"
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

class E20Dataset(Dataset):
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": n_total,
        "positives": n_pos,
        "negatives": n_neg
    }

def evaluate_model(model: nn.Module, loader: DataLoader, dev_records: List[Dict]) -> Tuple[Dict, Dict, List[Dict]]:
    model.eval()
    all_probs = []
    all_targets = []
    all_indices = []

    with torch.no_grad():
        for x_views, y_target, batch_indices in loader:
            x_views = x_views.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(x_views).squeeze(-1)
                probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(y_target.tolist())
            all_indices.extend(batch_indices.tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    overall_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.50)

    # Subgroup metrics
    subgroup_metrics = {}

    # AI Generators
    ai_generators = ["FLUX.1 [dev]", "Stable Diffusion", "Gemini / Imagen", "DALL-E", "Midjourney"]
    for gen in ai_generators:
        idxs = [i for i in range(len(dev_records)) if dev_records[all_indices[i]]["generator"] == gen and y_true[i] == 1]
        if len(idxs) > 0:
            rec_gen = np.mean(y_prob[idxs] >= 0.50)
            subgroup_metrics[f"ai_gen_{gen}_recall"] = float(rec_gen)
            subgroup_metrics[f"ai_gen_{gen}_count"] = len(idxs)

    # Real Device Families
    device_families = ["Apple iPhone", "Samsung Galaxy", "Google Pixel", "Vivo", "Nikon DSLR"]
    for dev_fam in device_families:
        idxs = [i for i in range(len(dev_records)) if dev_records[all_indices[i]]["device_family"] == dev_fam and y_true[i] == 0]
        if len(idxs) > 0:
            fpr_dev = np.mean(y_prob[idxs] >= 0.50)
            subgroup_metrics[f"real_dev_{dev_fam}_fpr"] = float(fpr_dev)
            subgroup_metrics[f"real_dev_{dev_fam}_count"] = len(idxs)

    # Compression Types
    compression_types = ["clean", "resize_jpeg", "moderate_jpeg", "severe_jpeg", "sequential_jpeg", "resize_jpeg_resize", "social_media_whatsapp"]
    for comp in compression_types:
        idxs = [i for i in range(len(dev_records)) if dev_records[all_indices[i]]["compression_type"] == comp]
        if len(idxs) > 0:
            comp_metrics = compute_binary_metrics(y_true[idxs], y_prob[idxs], threshold=0.50)
            subgroup_metrics[f"comp_{comp}_acc"] = comp_metrics["accuracy"]
            subgroup_metrics[f"comp_{comp}_f1"] = comp_metrics["f1"]
            subgroup_metrics[f"comp_{comp}_rec"] = comp_metrics["recall"]
            subgroup_metrics[f"comp_{comp}_fpr"] = comp_metrics["fpr"]
            subgroup_metrics[f"comp_{comp}_auc"] = comp_metrics["roc_auc"]
            subgroup_metrics[f"comp_{comp}_count"] = len(idxs)

    # Prediction rows for auditing
    pred_rows = []
    for i in range(len(all_indices)):
        orig_rec = dev_records[all_indices[i]]
        pred_rows.append({
            "image_id": orig_rec["image_id"],
            "filepath": orig_rec["filepath"],
            "label": orig_rec["label"],
            "generator": orig_rec["generator"],
            "device_family": orig_rec["device_family"],
            "compression_type": orig_rec["compression_type"],
            "prob": float(y_prob[i]),
            "pred": int(y_prob[i] >= 0.50)
        })

    return overall_metrics, subgroup_metrics, pred_rows

def main():
    set_seed(42)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STARTING E20 FORENSIC DETECTOR TRAINING")
    print("==================================================")
    print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. Read Manifests
    with open(TRAIN_MANIFEST_CSV, "r", encoding="utf-8") as f:
        train_records = list(csv.DictReader(f))
    with open(DEV_MANIFEST_CSV, "r", encoding="utf-8") as f:
        dev_records = list(csv.DictReader(f))

    print(f"Train Manifest: {len(train_records)} examples")
    print(f"Dev Manifest:   {len(dev_records)} examples")

    train_dataset = E20Dataset(train_records, DATA_ROOT)
    dev_dataset = E20Dataset(dev_records, DATA_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # 2. Build Model and Load Initial E6-C Weights
    print(f"\nLoading base MultiViewE5Model and E6-C weights from {E6C_CHECKPOINT}...")
    model = MultiViewE5Model()
    ckpt = torch.load(E6C_CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    print("Model initialized and E6-C checkpoint loaded successfully.")

    # 3. Initial Baseline Evaluation on E20 Dev (Epoch 0)
    print("\n--- Evaluating E6-C Baseline on E20 Dev (Epoch 0) ---")
    e6c_metrics, e6c_subgroups, e6c_preds = evaluate_model(model, dev_loader, dev_records)
    print(f"E6-C Dev ROC-AUC:  {e6c_metrics['roc_auc']:.4f}")
    print(f"E6-C Dev Accuracy: {e6c_metrics['accuracy']:.4f}")
    print(f"E6-C Dev F1:       {e6c_metrics['f1']:.4f}")
    print(f"E6-C Dev Precision:{e6c_metrics['precision']:.4f}")
    print(f"E6-C Dev Recall:   {e6c_metrics['recall']:.4f}")
    print(f"E6-C Dev FPR:      {e6c_metrics['fpr']:.4f}")
    print(f"E6-C FLUX Recall:  {e6c_subgroups.get('ai_gen_FLUX.1 [dev]_recall', 0.0):.4f}")
    print(f"E6-C Gemini Recall:{e6c_subgroups.get('ai_gen_Gemini / Imagen_recall', 0.0):.4f}")
    print(f"E6-C Pixel FPR:    {e6c_subgroups.get('real_dev_Google Pixel_fpr', 0.0):.4f}")
    print(f"E6-C WhatsApp F1:  {e6c_subgroups.get('comp_social_media_whatsapp_f1', 0.0):.4f}")

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    # Metrics history log
    metrics_history = []
    metrics_history.append({
        "epoch": 0,
        "phase": "baseline_e6c",
        "train_loss": 0.0,
        "dev_roc_auc": e6c_metrics["roc_auc"],
        "dev_pr_auc": e6c_metrics["pr_auc"],
        "dev_accuracy": e6c_metrics["accuracy"],
        "dev_f1": e6c_metrics["f1"],
        "dev_precision": e6c_metrics["precision"],
        "dev_recall": e6c_metrics["recall"],
        "dev_fpr": e6c_metrics["fpr"],
        "dev_fnr": e6c_metrics["fnr"],
        "tp": e6c_metrics["tp"],
        "tn": e6c_metrics["tn"],
        "fp": e6c_metrics["fp"],
        "fn": e6c_metrics["fn"],
        "flux_recall": e6c_subgroups.get("ai_gen_FLUX.1 [dev]_recall", 0.0),
        "gemini_recall": e6c_subgroups.get("ai_gen_Gemini / Imagen_recall", 0.0),
        "sd_recall": e6c_subgroups.get("ai_gen_Stable Diffusion_recall", 0.0),
        "dalle_recall": e6c_subgroups.get("ai_gen_DALL-E_recall", 0.0),
        "mj_recall": e6c_subgroups.get("ai_gen_Midjourney_recall", 0.0),
        "iphone_fpr": e6c_subgroups.get("real_dev_Apple iPhone_fpr", 0.0),
        "samsung_fpr": e6c_subgroups.get("real_dev_Samsung Galaxy_fpr", 0.0),
        "pixel_fpr": e6c_subgroups.get("real_dev_Google Pixel_fpr", 0.0),
        "vivo_fpr": e6c_subgroups.get("real_dev_Vivo_fpr", 0.0),
        "raise_fpr": e6c_subgroups.get("real_dev_Nikon DSLR_fpr", 0.0),
        "clean_f1": e6c_subgroups.get("comp_clean_f1", 0.0),
        "whatsapp_f1": e6c_subgroups.get("comp_social_media_whatsapp_f1", 0.0),
        "severe_jpeg_f1": e6c_subgroups.get("comp_severe_jpeg_f1", 0.0),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6)
    })

    best_dev_f1 = -1.0
    best_dev_auc = -1.0
    best_dev_fpr = 1.0
    best_dev_rec = -1.0
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

            if (batch_idx + 1) % 400 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - epoch_start
                throughput = ((batch_idx + 1) * 4) / max(elapsed, 1.0)
                vram_mb = torch.cuda.memory_allocated() / 1e6
                print(f"  Batch {batch_idx+1}/{n_batches} | Loss: {loss.item():.4f} (Avg: {running_loss/(batch_idx+1):.4f}) | Throughput: {throughput:.1f} img/s | VRAM: {vram_mb:.0f} MB", flush=True)

        train_loss = running_loss / n_batches
        epoch_time = time.time() - epoch_start

        # Validation
        print(f"Evaluating Epoch {epoch} on E20 Dev...")
        dev_m, dev_sub, dev_preds = evaluate_model(model, dev_loader, dev_records)

        print(f"Epoch {epoch} Dev Metrics: AUC={dev_m['roc_auc']:.4f} | Acc={dev_m['accuracy']:.4f} | F1={dev_m['f1']:.4f} | Prec={dev_m['precision']:.4f} | Rec={dev_m['recall']:.4f} | FPR={dev_m['fpr']:.4f}")
        print(f"  Subgroups: FLUX Rec={dev_sub.get('ai_gen_FLUX.1 [dev]_recall', 0):.4f} | Gemini Rec={dev_sub.get('ai_gen_Gemini / Imagen_recall', 0):.4f} | Pixel FPR={dev_sub.get('real_dev_Google Pixel_fpr', 0):.4f} | WA F1={dev_sub.get('comp_social_media_whatsapp_f1', 0):.4f}")

        # Checkpoint saving
        ckpt_path = CHECKPOINTS_DIR / f"e20_checkpoint_epoch{epoch}.pt"
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
            }, CHECKPOINTS_DIR / "e20_best_model.pt")
            with open(PREDICTIONS_DIR / "e20_best_dev_predictions.csv", "w", newline="", encoding="utf-8") as f:
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
            "flux_recall": dev_sub.get("ai_gen_FLUX.1 [dev]_recall", 0.0),
            "gemini_recall": dev_sub.get("ai_gen_Gemini / Imagen_recall", 0.0),
            "sd_recall": dev_sub.get("ai_gen_Stable Diffusion_recall", 0.0),
            "dalle_recall": dev_sub.get("ai_gen_DALL-E_recall", 0.0),
            "mj_recall": dev_sub.get("ai_gen_Midjourney_recall", 0.0),
            "iphone_fpr": dev_sub.get("real_dev_Apple iPhone_fpr", 0.0),
            "samsung_fpr": dev_sub.get("real_dev_Samsung Galaxy_fpr", 0.0),
            "pixel_fpr": dev_sub.get("real_dev_Google Pixel_fpr", 0.0),
            "vivo_fpr": dev_sub.get("real_dev_Vivo_fpr", 0.0),
            "raise_fpr": dev_sub.get("real_dev_Nikon DSLR_fpr", 0.0),
            "clean_f1": dev_sub.get("comp_clean_f1", 0.0),
            "whatsapp_f1": dev_sub.get("comp_social_media_whatsapp_f1", 0.0),
            "severe_jpeg_f1": dev_sub.get("comp_severe_jpeg_f1", 0.0),
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

            if (batch_idx + 1) % 400 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - epoch_start
                throughput = ((batch_idx + 1) * 4) / max(elapsed, 1.0)
                vram_mb = torch.cuda.memory_allocated() / 1e6
                print(f"  Batch {batch_idx+1}/{n_batches} | Loss: {loss.item():.4f} (Avg: {running_loss/(batch_idx+1):.4f}) | Throughput: {throughput:.1f} img/s | VRAM: {vram_mb:.0f} MB", flush=True)

        train_loss = running_loss / n_batches
        epoch_time = time.time() - epoch_start

        # Validation
        print(f"Evaluating Epoch {epoch} on E20 Dev...")
        dev_m, dev_sub, dev_preds = evaluate_model(model, dev_loader, dev_records)

        print(f"Epoch {epoch} Dev Metrics: AUC={dev_m['roc_auc']:.4f} | Acc={dev_m['accuracy']:.4f} | F1={dev_m['f1']:.4f} | Prec={dev_m['precision']:.4f} | Rec={dev_m['recall']:.4f} | FPR={dev_m['fpr']:.4f}")
        print(f"  Subgroups: FLUX Rec={dev_sub.get('ai_gen_FLUX.1 [dev]_recall', 0):.4f} | Gemini Rec={dev_sub.get('ai_gen_Gemini / Imagen_recall', 0):.4f} | Pixel FPR={dev_sub.get('real_dev_Google Pixel_fpr', 0):.4f} | WA F1={dev_sub.get('comp_social_media_whatsapp_f1', 0):.4f}")

        # Checkpoint saving
        ckpt_path = CHECKPOINTS_DIR / f"e20_checkpoint_epoch{epoch}.pt"
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
            }, CHECKPOINTS_DIR / "e20_best_model.pt")
            with open(PREDICTIONS_DIR / "e20_best_dev_predictions.csv", "w", newline="", encoding="utf-8") as f:
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
            "flux_recall": dev_sub.get("ai_gen_FLUX.1 [dev]_recall", 0.0),
            "gemini_recall": dev_sub.get("ai_gen_Gemini / Imagen_recall", 0.0),
            "sd_recall": dev_sub.get("ai_gen_Stable Diffusion_recall", 0.0),
            "dalle_recall": dev_sub.get("ai_gen_DALL-E_recall", 0.0),
            "mj_recall": dev_sub.get("ai_gen_Midjourney_recall", 0.0),
            "iphone_fpr": dev_sub.get("real_dev_Apple iPhone_fpr", 0.0),
            "samsung_fpr": dev_sub.get("real_dev_Samsung Galaxy_fpr", 0.0),
            "pixel_fpr": dev_sub.get("real_dev_Google Pixel_fpr", 0.0),
            "vivo_fpr": dev_sub.get("real_dev_Vivo_fpr", 0.0),
            "raise_fpr": dev_sub.get("real_dev_Nikon DSLR_fpr", 0.0),
            "clean_f1": dev_sub.get("comp_clean_f1", 0.0),
            "whatsapp_f1": dev_sub.get("comp_social_media_whatsapp_f1", 0.0),
            "severe_jpeg_f1": dev_sub.get("comp_severe_jpeg_f1", 0.0),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6)
        })

    total_training_time = time.time() - total_training_start

    # Save metrics CSV
    csv_path = OUT_DIR / "e20_training_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_history)
    print(f"\nSaved metrics history to {csv_path}")

    # Save metadata JSON
    meta = {
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "best_dev_auc": best_dev_auc,
        "best_dev_fpr": best_dev_fpr,
        "best_dev_rec": best_dev_rec,
        "total_training_time_seconds": total_training_time,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1e6),
        "history": metrics_history
    }
    with open(CHECKPOINTS_DIR / "e20_best_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n==================================================")
    print("E20 TRAINING COMPLETE")
    print(f"Total Time: {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print(f"Best Epoch: {best_epoch} (F1: {best_dev_f1:.4f}, AUC: {best_dev_auc:.4f})")
    print(f"Peak VRAM:  {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    print("==================================================")

if __name__ == "__main__":
    main()
