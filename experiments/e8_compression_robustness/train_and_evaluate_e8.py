"""
Experiment E8: Compression-Robust DeepVision Forensics (E8-B Fine-Tuning Pipeline)

Scientific Protocol:
- Start checkpoint: E6-C Epoch 2 (experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt)
- Exact 5-view architecture, learned-attention aggregation, production threshold 0.50.
- On-the-fly in-memory social-media compression augmentation during training (35% clean, 65% degraded across mild, moderate, severe recompression).
- Clean, unaugmented E5 validation split (N=5,775) used for checkpoint selection.
- Single-shot final evaluation on frozen WhatsApp test set (N=67).
- Diagnostic hard cases evaluation (original_ai_edit.png, whatsapp_download.jpeg).
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
from experiments.e8_compression_robustness.augmentations import apply_compression_augmentation

CHECKPOINT_PATH = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
MANIFEST_PATH = PROJECT_ROOT / "data/e5_external/manifests/e5_manifest.csv"
WHATSAPP_DIR = PROJECT_ROOT / "data/whatsapp_robustness_test"
HARD_CASES_DIR = PROJECT_ROOT / "data/e6_hard_cases"
OUTPUT_DIR = PROJECT_ROOT / "experiments/e8_compression_robustness"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"

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
    crops = []
    crops.append(img.resize((224, 224), Image.Resampling.BILINEAR))
    crops.append(img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR))
    crops.append(img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR))
    crops.append(img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR))
    crops.append(img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR))
    return crops


class E8MultiViewDataset(Dataset):
    def __init__(self, records, is_train: bool = True):
        self.records = records
        self.is_train = is_train
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

            # Apply probabilistic compression augmentation during training only
            if self.is_train:
                img_rgb = apply_compression_augmentation(img_rgb)
                if random.random() > 0.5:
                    img_rgb = img_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            crops = generate_5_crops(img_rgb)
            view_tensors = [self.to_tensor(c) for c in crops]
            stacked_views = torch.stack(view_tensors, dim=0)  # (5, 3, 224, 224)
            return stacked_views, torch.tensor(label, dtype=torch.float32), True
        except Exception:
            dummy_views = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
            return dummy_views, torch.tensor(label, dtype=torch.float32), False


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    preds = (y_prob >= threshold).astype(int)

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

    roc_auc = compute_roc_auc(y_true, y_prob)
    pr_auc = compute_pr_auc(y_true, y_prob)

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


def get_source_category(rec):
    src = rec.get("source_dataset", "")
    gen = rec.get("generator_or_device", "")
    if "GenImage" in src:
        return "GenImage"
    elif "Flux-1-Dev" in src or "FLUX.1 [dev]" in gen:
        return "FLUX dev"
    elif "Flux-1-Schnell" in src or "FLUX.1 [schnell]" in gen:
        return "FLUX schnell"
    elif "SDXL" in gen or "Stable Diffusion XL" in gen:
        return "SDXL"
    elif "VISION" in src or "VISION" in gen:
        if "Apple" in gen or "iPhone" in gen:
            return "VISION Apple"
        elif "Samsung" in gen or "Galaxy" in gen or "OnePlus" in gen or "Android" in gen:
            return "VISION Android"
        else:
            return "VISION Other"
    return "Other"


def main():
    set_seed(42)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75, flush=True)
    print("EXPERIMENT E8: COMPRESSION-ROBUST DEEPVISION FORENSICS (E8-B)", flush=True)
    print("=" * 75, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Starting Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}", flush=True)

    # 1. Model Instantiation
    model = MultiViewE5Model(
        checkpoint_path=None,
        num_views=5,
        freq_norm_strategy="standardize",
        freq_embedding_dim=256,
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    # UNFREEZE ALL PARAMETERS FOR FINE-TUNING
    for p in model.parameters():
        p.requires_grad = True

    model.to(DEVICE)

    # Setup differential parameter groups
    backbone_params = list(model.base_model.spatial_branch.parameters()) + list(model.base_model.frequency_branch.parameters())
    multiview_head_params = list(model.view_attention.parameters()) + list(model.classifier.parameters())

    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in multiview_head_params)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[VERIFIED] Model Initialized from E6-C Epoch 2:", flush=True)
    print(f"  - Backbone Trainable Params: {n_backbone:,} (lr=1e-5)", flush=True)
    print(f"  - Head Trainable Params:     {n_head:,} (lr=1e-4)", flush=True)
    print(f"  - Total Trainable Params:    {n_total:,}", flush=True)

    # 2. Parse Manifest
    train_records, val_records = [], []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "image_path": str(PROJECT_ROOT / row["image_path"]),
                "rel_path": row["image_path"],
                "label": int(row["label"]),
                "source_dataset": row.get("source_dataset", ""),
                "generator_or_device": row.get("generator_or_device", ""),
            }
            if row["split"] == "train":
                train_records.append(rec)
            elif row["split"] == "val":
                val_records.append(rec)

    print(f"[*] Dataset: {len(train_records):,} Train images | {len(val_records):,} Val images", flush=True)

    batch_size = 8
    num_epochs = 2

    train_dataset = E8MultiViewDataset(train_records, is_train=True)
    val_dataset = E8MultiViewDataset(val_records, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-4},
        {"params": multiview_head_params, "lr": 1e-4, "weight_decay": 1e-4},
    ])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    epoch_history = []
    best_epoch = 1
    best_val_auc = 0.0
    best_val_f1 = 0.0
    best_val_metrics = None

    print("\n" + "-" * 75, flush=True)
    print(f"[*] Starting E8-B Fine-Tuning: {num_epochs} Epochs, Batch Size={batch_size}, AMP=True", flush=True)
    print("-" * 75, flush=True)

    t_train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t_ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step_idx, (x_views, y_true, valid) in enumerate(train_loader, 1):
            mask = valid.bool()
            if not mask.any():
                continue
            x_views = x_views[mask].to(DEVICE, non_blocking=True)
            y_true = y_true[mask].to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(x_views)
                loss = criterion(logits, y_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(y_true)
            preds = (torch.sigmoid(logits) >= 0.50).float()
            train_correct += (preds == y_true).sum().item()
            train_total += len(y_true)

            if step_idx % 250 == 0 or step_idx == len(train_loader):
                elapsed = time.time() - t_ep_start
                rate = train_total / max(1e-4, elapsed)
                avg_l = train_loss / max(1, train_total)
                acc = train_correct / max(1, train_total)
                print(
                    f"    [Epoch {epoch:2d}/{num_epochs:2d}] Step {step_idx:4d}/{len(train_loader):4d} | "
                    f"Loss: {avg_l:.4f} Acc: {acc*100:5.2f}% ({rate:.1f} img/s)",
                    flush=True,
                )

        avg_train_loss = train_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Clean E5 Validation Pass
        model.eval()
        val_loss = 0.0
        val_probs_list, val_labels_list = [], []

        with torch.no_grad():
            for x_views, y_true, valid in val_loader:
                mask = valid.bool()
                if not mask.any():
                    continue
                x_views = x_views[mask].to(DEVICE, non_blocking=True)
                y_true = y_true[mask].to(DEVICE, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(x_views)
                    loss = criterion(logits, y_true)

                val_loss += loss.item() * len(y_true)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs_list.extend(probs)
                val_labels_list.extend(y_true.cpu().numpy().flatten())

        avg_val_loss = val_loss / max(1, len(val_labels_list))
        val_y_true = np.array(val_labels_list)
        val_y_prob = np.array(val_probs_list)
        v_m = compute_metrics(val_y_true, val_y_prob, threshold=0.50)
        t_ep = time.time() - t_ep_start

        ep_info = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": v_m["accuracy"],
            "val_roc_auc": v_m["roc_auc"],
            "val_pr_auc": v_m["pr_auc"],
            "val_f1": v_m["f1"],
            "val_precision": v_m["precision"],
            "val_recall": v_m["recall"],
            "val_fpr": v_m["fpr"],
            "val_fnr": v_m["fnr"],
            "epoch_time_sec": round(t_ep, 2),
        }
        epoch_history.append(ep_info)

        # Save Per-Epoch Checkpoint
        epoch_ckpt_path = CHECKPOINTS_DIR / f"e8b_checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": v_m,
            "val_roc_auc": v_m["roc_auc"],
        }, epoch_ckpt_path)
        print(f"  [+] Saved epoch checkpoint: {epoch_ckpt_path.name}", flush=True)

        print(
            f"[SUMMARY] Epoch {epoch:2d}/{num_epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {v_m['accuracy']*100:5.2f}% "
            f"AUC: {v_m['roc_auc']:.4f} F1: {v_m['f1']:.4f} FPR: {v_m['fpr']*100:4.2f}% ({t_ep/60.0:.1f}m)",
            flush=True,
        )

        if v_m["roc_auc"] > best_val_auc or (v_m["roc_auc"] == best_val_auc and v_m["f1"] > best_val_f1):
            best_val_auc = v_m["roc_auc"]
            best_val_f1 = v_m["f1"]
            best_epoch = epoch
            best_val_metrics = v_m
            best_ckpt_path = CHECKPOINTS_DIR / "e8b_best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": v_m,
                "val_roc_auc": best_val_auc,
            }, best_ckpt_path)
            print(f"  [***] New best checkpoint: {best_ckpt_path.name} (Val AUC: {best_val_auc:.4f}, F1: {best_val_f1:.4f})", flush=True)

    t_train_total = time.time() - t_train_start
    print(f"\n[*] Training Complete in {t_train_total/3600.0:.2f} hours. Best Epoch: {best_epoch}", flush=True)

    # 3. Reload Best Model for Evaluation
    best_ckpt = torch.load(CHECKPOINTS_DIR / "e8b_best_model.pt", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # 4. Comprehensive Clean E5 Validation Breakdown
    print("\n" + "-" * 75, flush=True)
    print("PHASE 4: Comprehensive Clean E5 Validation Breakdown", flush=True)
    print("-" * 75, flush=True)

    val_probs_all, val_labels_all = [], []
    with torch.no_grad():
        for x_views, y_true, valid in val_loader:
            mask = valid.bool()
            if not mask.any():
                continue
            x_views = x_views[mask].to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(x_views)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            val_probs_all.extend(probs)
            val_labels_all.extend(y_true[mask].numpy().flatten())

    val_y_true = np.array(val_labels_all)
    val_y_prob = np.array(val_probs_all)
    overall_val_metrics = compute_metrics(val_y_true, val_y_prob, threshold=0.50)

    cat_indices = {}
    for i, rec in enumerate(val_records):
        cat = get_source_category(rec)
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    source_metrics = {}
    print(f"{'Source Category':<18} | {'N':<6} | {'Accuracy':<8} | {'ROC-AUC':<8} | {'Precision':<8} | {'Recall':<8} | {'F1':<8}", flush=True)
    print("-" * 75, flush=True)
    for cat, idxs in cat_indices.items():
        sub_true = val_y_true[idxs]
        sub_prob = val_y_prob[idxs]
        sm = compute_metrics(sub_true, sub_prob, threshold=0.50)
        source_metrics[cat] = sm
        print(f"{cat:<18} | {sm['n_samples']:<6} | {sm['accuracy']*100:>6.2f}% | {sm['roc_auc']:>8.4f} | {sm['precision']*100:>6.2f}% | {sm['recall']*100:>6.2f}% | {sm['f1']:>8.4f}", flush=True)

    # 5. Diagnostic Hard-Case Evaluation
    print("\n" + "-" * 75, flush=True)
    print("PHASE 5: Diagnostic Hard-Case Evaluation", flush=True)
    print("-" * 75, flush=True)

    hard_cases = [
        {"name": "original_ai_edit.png", "path": HARD_CASES_DIR / "original_ai_edit.png", "true_label": "AI"},
        {"name": "whatsapp_download.jpeg", "path": HARD_CASES_DIR / "whatsapp_download.jpeg", "true_label": "REAL"},
    ]
    hard_case_results = []
    to_tensor = transforms.ToTensor()

    for hc in hard_cases:
        p = hc["path"]
        if not p.exists():
            continue
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            crops = generate_5_crops(rgb)
        tensors = [to_tensor(c) for c in crops]
        x_views = torch.stack(tensors, dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            B, V, C, H, W = x_views.shape
            x_flat = x_views.view(B * V, C, H, W)
            feats = model.base_model(x_flat, return_features=True)
            e_fused = feats["fused_embedding"]
            view_logits = model.classifier(e_fused).view(B, V)
            view_probs = torch.sigmoid(view_logits).view(5).cpu().tolist()

            attn_scores = model.view_attention(e_fused.view(B, V, model.fused_dim))
            attn_w = torch.softmax(attn_scores, dim=1)
            pooled_emb = torch.sum(attn_w * e_fused.view(B, V, model.fused_dim), dim=1)
            e8_prob = float(torch.sigmoid(model.classifier(pooled_emb)).item())

        hc_entry = {
            "name": hc["name"],
            "true_label": hc["true_label"],
            "e8_prob": round(e8_prob, 4),
            "p_global": round(view_probs[0], 4),
            "p_strongest_local": round(float(max(view_probs[1:])), 4),
            "local_probs": [round(p, 4) for p in view_probs[1:]],
        }
        hard_case_results.append(hc_entry)
        print(f"  [{hc['name']}] True={hc['true_label']} | E8 Prob: {e8_prob:.4f} | Global: {view_probs[0]:.4f} | Strongest Local: {max(view_probs[1:]):.4f}", flush=True)

    # 6. Single-Shot Frozen WhatsApp Evaluation
    print("\n" + "-" * 75, flush=True)
    print("PHASE 6: Single-Shot Evaluation on Frozen WhatsApp Test Set (N=67)", flush=True)
    print("-" * 75, flush=True)

    real_files = sorted([f for f in (WHATSAPP_DIR / "real").iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    ai_files = sorted([f for f in (WHATSAPP_DIR / "ai").iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

    whatsapp_samples = []
    for f in real_files:
        whatsapp_samples.append({"path": f, "label": 0, "label_str": "REAL"})
    for f in ai_files:
        whatsapp_samples.append({"path": f, "label": 1, "label_str": "AI"})

    # Load baseline E6-C results for direct sample-by-sample comparison
    e6c_csv_path = PROJECT_ROOT / "experiments/whatsapp_robustness_test/per_image_predictions.csv"
    e6c_data = {}
    if e6c_csv_path.exists():
        with open(e6c_csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                e6c_data[row["filename"]] = {
                    "e6c_prob": float(row["overall_ai_prob"]),
                    "e6c_pred": row["predicted_label"],
                }

    wa_results = []
    with torch.no_grad():
        for idx, sample in enumerate(whatsapp_samples, 1):
            img_path = sample["path"]
            label = sample["label"]
            label_str = sample["label_str"]

            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                orig_w, orig_h = img_rgb.size
                crops = generate_5_crops(img_rgb)

            crop_tensors = [to_tensor(c) for c in crops]
            x_views = torch.stack(crop_tensors, dim=0).unsqueeze(0).to(DEVICE)

            B, V, C, H, W = x_views.shape
            x_flat = x_views.view(B * V, C, H, W)
            feats = model.base_model(x_flat, return_features=True)
            e_fused = feats["fused_embedding"]
            view_logits = model.classifier(e_fused).view(B, V)
            view_probs = torch.sigmoid(view_logits).view(5).cpu().tolist()

            attn_scores = model.view_attention(e_fused.view(B, V, model.fused_dim))
            attn_w = torch.softmax(attn_scores, dim=1)
            pooled_emb = torch.sum(attn_w * e_fused.view(B, V, model.fused_dim), dim=1)
            e8_prob = float(torch.sigmoid(model.classifier(pooled_emb)).item())

            e8_pred = "AI" if e8_prob >= 0.50 else "REAL"

            e6c_info = e6c_data.get(img_path.name, {"e6c_prob": 0.0, "e6c_pred": "REAL"})
            e6c_prob = e6c_info["e6c_prob"]
            e6c_pred = e6c_info["e6c_pred"]

            transition = "UNCHANGED"
            if e6c_pred != e8_pred:
                transition = f"{e6c_pred}->{e8_pred}"

            wa_results.append({
                "index": idx,
                "filename": img_path.name,
                "true_label": label_str,
                "true_numeric": label,
                "e6c_prob": round(e6c_prob, 6),
                "e6c_pred": e6c_pred,
                "is_e6c_correct": (e6c_pred == label_str),
                "e8_prob": round(e8_prob, 6),
                "e8_pred": e8_pred,
                "is_e8_correct": (e8_pred == label_str),
                "transition": transition,
                "p_global": round(view_probs[0], 6),
                "p_strongest_local": round(float(max(view_probs[1:])), 6),
                "tl_prob": round(view_probs[1], 6),
                "tr_prob": round(view_probs[2], 6),
                "bl_prob": round(view_probs[3], 6),
                "br_prob": round(view_probs[4], 6),
                "width": orig_w,
                "height": orig_h,
            })

    wa_true = np.array([r["true_numeric"] for r in wa_results], dtype=int)
    wa_e6c_probs = np.array([r["e6c_prob"] for r in wa_results], dtype=float)
    wa_e8_probs = np.array([r["e8_prob"] for r in wa_results], dtype=float)

    m_wa_e6c = compute_metrics(wa_true, wa_e6c_probs, threshold=0.50)
    m_wa_e8 = compute_metrics(wa_true, wa_e8_probs, threshold=0.50)

    recovered_ai = [r for r in wa_results if r["true_label"] == "AI" and r["e6c_pred"] == "REAL" and r["e8_pred"] == "AI"]
    additional_fp = [r for r in wa_results if r["true_label"] == "REAL" and r["e6c_pred"] == "REAL" and r["e8_pred"] == "AI"]
    degraded_ai = [r for r in wa_results if r["true_label"] == "AI" and r["e6c_pred"] == "AI" and r["e8_pred"] == "REAL"]
    recovered_real = [r for r in wa_results if r["true_label"] == "REAL" and r["e6c_pred"] == "AI" and r["e8_pred"] == "REAL"]

    print("\n" + "=" * 75, flush=True)
    print("WHATSAPP EVALUATION BENCHMARK: E6-C vs E8-B", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Metric':<25} | {'E6-C Baseline':<15} | {'E8-B Compression-Robust':<25} | {'Delta'}", flush=True)
    print("-" * 75, flush=True)
    metrics_comp = [
        ("Accuracy", f"{m_wa_e6c['accuracy']*100:.2f}%", f"{m_wa_e8['accuracy']*100:.2f}%", f"{(m_wa_e8['accuracy'] - m_wa_e6c['accuracy'])*100:+.2f}%"),
        ("ROC-AUC", f"{m_wa_e6c['roc_auc']:.4f}", f"{m_wa_e8['roc_auc']:.4f}", f"{m_wa_e8['roc_auc'] - m_wa_e6c['roc_auc']:+.4f}"),
        ("PR-AUC", f"{m_wa_e6c['pr_auc']:.4f}", f"{m_wa_e8['pr_auc']:.4f}", f"{m_wa_e8['pr_auc'] - m_wa_e6c['pr_auc']:+.4f}"),
        ("Precision", f"{m_wa_e6c['precision']*100:.2f}%", f"{m_wa_e8['precision']*100:.2f}%", f"{(m_wa_e8['precision'] - m_wa_e6c['precision'])*100:+.2f}%"),
        ("Recall", f"{m_wa_e6c['recall']*100:.2f}%", f"{m_wa_e8['recall']*100:.2f}%", f"{(m_wa_e8['recall'] - m_wa_e6c['recall'])*100:+.2f}%"),
        ("F1 Score", f"{m_wa_e6c['f1']:.4f}", f"{m_wa_e8['f1']:.4f}", f"{m_wa_e8['f1'] - m_wa_e6c['f1']:+.4f}"),
        ("FPR (Real False Alarm)", f"{m_wa_e6c['fpr']*100:.2f}%", f"{m_wa_e8['fpr']*100:.2f}%", f"{(m_wa_e8['fpr'] - m_wa_e6c['fpr'])*100:+.2f}%"),
        ("FNR (AI Miss Rate)", f"{m_wa_e6c['fnr']*100:.2f}%", f"{m_wa_e8['fnr']*100:.2f}%", f"{(m_wa_e8['fnr'] - m_wa_e6c['fnr'])*100:+.2f}%"),
        ("Confusion (TP / TN)", f"{m_wa_e6c['tp']} / {m_wa_e6c['tn']}", f"{m_wa_e8['tp']} / {m_wa_e8['tn']}", f"{m_wa_e8['tp'] - m_wa_e6c['tp']:+d} TP / {m_wa_e8['tn'] - m_wa_e6c['tn']:+d} TN"),
        ("Confusion (FP / FN)", f"{m_wa_e6c['fp']} / {m_wa_e6c['fn']}", f"{m_wa_e8['fp']} / {m_wa_e8['fn']}", f"{m_wa_e8['fp'] - m_wa_e6c['fp']:+d} FP / {m_wa_e8['fn'] - m_wa_e6c['fn']:+d} FN"),
    ]
    for row in metrics_comp:
        print(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<25} | {row[3]}", flush=True)

    print("\nTransition Summary on WhatsApp Dataset:", flush=True)
    print(f"  - AI False Negatives Recovered: {len(recovered_ai)} / 24", flush=True)
    for r in recovered_ai:
        print(f"    * RECOVERED: {r['filename']} (E6-C={r['e6c_prob']:.4f} -> E8={r['e8_prob']:.4f})", flush=True)
    print(f"  - Additional Real False Positives: {len(additional_fp)}", flush=True)
    for r in additional_fp:
        print(f"    * NEW FP: {r['filename']} (E6-C={r['e6c_prob']:.4f} -> E8={r['e8_prob']:.4f})", flush=True)
    print(f"  - AI True Positives Degraded: {len(degraded_ai)}", flush=True)
    print(f"  - Real False Positives Corrected: {len(recovered_real)}", flush=True)

    # 7. Save WhatsApp Predictions CSV
    wa_csv = RESULTS_DIR / "per_image_predictions_whatsapp.csv"
    with open(wa_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wa_results[0].keys()))
        writer.writeheader()
        writer.writerows(wa_results)
    print(f"\n[*] Saved WhatsApp predictions CSV: {wa_csv.name}", flush=True)

    # 8. Save Full Study JSON
    study_json = RESULTS_DIR / "e8_study_results.json"
    study_data = {
        "experiment": "Experiment E8: Compression-Robust DeepVision Forensics (E8-B)",
        "starting_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "best_epoch": best_epoch,
        "total_training_time_hours": round(t_train_total / 3600.0, 2),
        "parameters": {
            "total_trainable": n_total,
            "backbone": n_backbone,
            "head": n_head,
        },
        "training_history": epoch_history,
        "clean_e5_validation": {
            "overall": overall_val_metrics,
            "source_breakdown": source_metrics,
        },
        "hard_case_diagnostics": hard_case_results,
        "whatsapp_test": {
            "e6c_baseline": m_wa_e6c,
            "e8b_compression_robust": m_wa_e8,
            "ai_false_negatives_recovered_count": len(recovered_ai),
            "recovered_samples": recovered_ai,
            "additional_real_false_positives_count": len(additional_fp),
            "additional_fp_samples": additional_fp,
        }
    }
    with open(study_json, "w", encoding="utf-8") as f:
        json.dump(study_data, f, indent=2)
    print(f"[*] Saved study JSON: {study_json.name}", flush=True)

    # 9. Generate E8_RESEARCH_REPORT.md
    report_md = OUTPUT_DIR / "E8_RESEARCH_REPORT.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Experiment E8 Research Report: Compression-Robust DeepVision Forensics\n\n")
        f.write("**Status**: Complete | **Model**: E8-B Compression-Augmented Fine-Tuning\n\n")
        f.write(f"- **Starting Checkpoint**: `{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}`\n")
        f.write(f"- **Architecture**: Production 5-View Multi-Scale Spatial + Spectral Dual-Branch\n")
        f.write(f"- **Total Trainable Parameters**: **{n_total:,}**\n")
        f.write(f"- **Augmentation Strategy**: On-the-fly in-memory compression mixture (35% clean, 65% degraded)\n")
        f.write(f"- **Training Dataset**: E5 Train Split ($N=23,165$, dynamic augmentation)\n")
        f.write(f"- **Selection Dataset**: Clean E5 Validation ($N=5,775$, strictly unaugmented)\n")
        f.write(f"- **Frozen Test Set**: WhatsApp Robustness Dataset ($N=67$: 34 Real, 33 AI)\n")
        f.write(f"- **Training Duration**: **{t_train_total/3600.0:.2f} hours** (2 Epochs)\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write("Experiment E8 directly evaluates the central hypothesis: that exposure to realistic on-the-fly social-media compression and degradation during training enables DeepVision Forensics to bridge the domain shift on WhatsApp-compressed images without sacrificing clean-image performance.\n\n")

        f.write("## 2. Augmentation Pipeline Specification\n\n")
        f.write("- **Clean (35%)**: Unmodified original image, preserving clean-domain feature representations.\n")
        f.write("- **Mild Compression (25%)**: Random single-pass JPEG ($Q \\in [75, 95]$).\n")
        f.write("- **Moderate Compression (20%)**: Downscale ($0.5\\times - 0.75\\times$), resize back + JPEG ($Q \\in [55, 75]$).\n")
        f.write("- **Severe / WhatsApp Recompression (20%)**: Downscale ($0.4\\times - 0.6\\times$), optional Gaussian blur ($\\sigma \\in [0.4, 0.9]$), 1st JPEG pass ($Q \\in [40, 65]$ with 4:2:0 chroma subsampling), decode, 2nd JPEG pass ($Q \\in [45, 70]$), resize back.\n\n")

        f.write("## 3. Clean E5 Validation Benchmark ($N=5,775$)\n\n")
        f.write("| Model / Experiment | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | FPR | FNR |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| **E6-C Baseline (Production)** | 96.16% | 0.9936 | 0.9937 | 96.89% | 95.25% | 0.9606 | **2.97%** | 4.75% |\n")
        f.write("| **E7-A Learned Feature Fusion** | 96.26% | 0.9948 | 0.9948 | 94.93% | 97.61% | 0.9625 | 5.04% | 2.39% |\n")
        f.write(f"| **E8-B Compression-Robust (Ours)** | **{overall_val_metrics['accuracy']*100:.2f}%** | **{overall_val_metrics['roc_auc']:.4f}** | **{overall_val_metrics['pr_auc']:.4f}** | **{overall_val_metrics['precision']*100:.2f}%** | **{overall_val_metrics['recall']*100:.2f}%** | **{overall_val_metrics['f1']:.4f}** | **{overall_val_metrics['fpr']*100:.2f}%** | **{overall_val_metrics['fnr']*100:.2f}%** |\n\n")

        f.write("### Clean E5 Source-Level Breakdown (E8-B):\n\n")
        f.write("| Source Category | N Samples | Accuracy | ROC-AUC | Precision | Recall | F1 Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cat, sm in source_metrics.items():
            f.write(f"| **{cat}** | {sm['n_samples']} | {sm['accuracy']*100:.2f}% | {sm['roc_auc']:.4f} | {sm['precision']*100:.2f}% | {sm['recall']*100:.2f}% | {sm['f1']:.4f} |\n")
        f.write("\n")

        f.write("## 4. Diagnostic Hard Cases\n\n")
        f.write("| Hard Case | Ground Truth | E6-C Baseline Prob | E8-B Prob | Global View | Strongest Local |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for hc in hard_case_results:
            f.write(f"| `{hc['name']}` | **{hc['true_label']}** | {hc.get('e6c_prob', '-')} | **{hc['e8_prob']:.4f}** | {hc['p_global']:.4f} | {hc['p_strongest_local']:.4f} |\n")
        f.write("\n")

        f.write("## 5. Frozen WhatsApp Test Set Performance ($N=67$)\n\n")
        f.write("Single-shot evaluation on the frozen WhatsApp dataset (34 Real, 33 AI):\n\n")
        f.write("| Metric | E6-C Baseline | E8-B Compression-Robust | Delta |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for row in metrics_comp:
            f.write(f"| **{row[0]}** | {row[1]} | {row[2]} | **{row[3]}** |\n")
        f.write("\n")

        f.write("### Confusion Matrix Comparison:\n\n")
        f.write("```\n")
        f.write(f"E6-C Baseline:   TN={m_wa_e6c['tn']}, FP={m_wa_e6c['fp']} (FPR={m_wa_e6c['fpr']*100:.2f}%) | FN={m_wa_e6c['fn']}, TP={m_wa_e6c['tp']} (Recall={m_wa_e6c['recall']*100:.2f}%)\n")
        f.write(f"E8-B Robust:     TN={m_wa_e8['tn']}, FP={m_wa_e8['fp']} (FPR={m_wa_e8['fpr']*100:.2f}%) | FN={m_wa_e8['fn']}, TP={m_wa_e8['tp']} (Recall={m_wa_e8['recall']*100:.2f}%)\n")
        f.write("```\n\n")

        f.write("### Transition Audit:\n\n")
        f.write(f"- **AI False Negatives Recovered**: **{len(recovered_ai)} / 24**\n")
        for r in recovered_ai:
            f.write(f"  - `{r['filename']}`: E6-C = {r['e6c_prob']:.4f} $\\to$ E8 = {r['e8_prob']:.4f} (Strongest Local = {r['p_strongest_local']:.4f})\n")
        f.write(f"- **Additional Real False Positives**: **{len(additional_fp)}**\n")
        for r in additional_fp:
            f.write(f"  - `{r['filename']}`: E6-C = {r['e6c_prob']:.4f} $\\to$ E8 = {r['e8_prob']:.4f}\n")
        f.write(f"- **AI True Positives Degraded**: **{len(degraded_ai)}**\n")
        f.write(f"- **Real False Positives Corrected**: **{len(recovered_real)}**\n\n")

        f.write("## 6. Scientific Analysis & Production Assessment\n\n")
        f.write("1. **Hypothesis Evaluation**: Did compression-aware augmentation improve WhatsApp AI detection while controlling real false alarms?\n")
        f.write("2. **Domain Adaptation Trade-off**: Analyzing the trade-off between clean-image precision and lossy domain recall.\n")
        f.write("3. **Production Recommendation**: Clear scientific guidance on whether E8 justifies deployment or further development.\n")

    print(f"[*] Saved Research Report: {report_md.name}", flush=True)

    # 10. Generate README.md
    readme_md = OUTPUT_DIR / "README.md"
    with open(readme_md, "w", encoding="utf-8") as f:
        f.write("# Experiment E8: Compression-Robust DeepVision Forensics\n\n")
        f.write("Contains training pipeline, checkpoints, and evaluation results for compression-aware multi-view training.\n\n")
        f.write("## Directory Layout\n")
        f.write("- [`augmentations.py`](./augmentations.py): On-the-fly in-memory realistic social media compression pipeline.\n")
        f.write("- [`smoke_test_e8.py`](./smoke_test_e8.py): Pre-training verification and runtime pilot.\n")
        f.write("- [`train_and_evaluate_e8.py`](./train_and_evaluate_e8.py): Complete training and evaluation runner.\n")
        f.write("- [`checkpoints/e8b_best_model.pt`](./checkpoints/e8b_best_model.pt): Best E8-B checkpoint.\n")
        f.write("- [`results/e8_study_results.json`](./results/e8_study_results.json): Full experimental metrics.\n")
        f.write("- [`results/per_image_predictions_whatsapp.csv`](./results/per_image_predictions_whatsapp.csv): WhatsApp test predictions.\n")
        f.write("- [`E8_RESEARCH_REPORT.md`](./E8_RESEARCH_REPORT.md): Comprehensive scientific research report.\n")

    print(f"[*] Saved README: {readme_md.name}", flush=True)
    print("\n" + "=" * 75, flush=True)
    print("EXPERIMENT E8 COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
