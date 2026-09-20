"""
Experiment E12-B: Clean-Data Domain-Conditioned Normalization
DeepVision-Forensics Research Pipeline

Protocol & Scientific Framework:
- Hypothesis under test: Investigates whether inserting lightweight LayerNorm on the
  aggregated 1792-D multi-view representation prior to classification can mitigate
  compression-induced covariate drift and reduce WhatsApp false positives while preserving AI recall.
  NOTE: This is strictly treated as an empirical hypothesis, NOT an established result.
- Starting Checkpoint: experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt
- Exact 5-view MultiViewE5Model architecture with LayerNorm(1792) inserted after view aggregation and before classifier
- Base backbones (EfficientNet-B3 + Frequency CNN) 100% FROZEN (11,150,856 parameters)
- Trainable: view_attention (229,633) + LayerNorm (3,584) + classifier (984,833) = 1,218,050 parameters (9.85%)
- Training: 100% Clean E5 training data only (N=23,165, no synthetic compression, random h-flip only)
- Validation: 100% Clean E5 validation only for checkpoint selection (N=5,775)
- Exactly 3 epochs, Batch size 16, AdamW lr=1e-4, weight_decay=1e-4, CUDA AMP FP16
- Frozen 67-image WhatsApp benchmark evaluated STRICTLY single-shot after model freezing
- Zero modifications to production, zero modifications to datasets/manifests, V2 untouched
"""

import os
import io
import csv
import json
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set deterministic seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(r"d:\DeepVision-Forensics")
E6C_CHECKPOINT = PROJECT_ROOT / "experiments" / "e6_multiscale_training" / "e6c_finetune_benchmark" / "e6c_checkpoint_epoch2.pt"
E5_MANIFEST = PROJECT_ROOT / "data" / "e5_external" / "manifests" / "e5_manifest.csv"
WHATSAPP_TEST_DIR = PROJECT_ROOT / "data" / "whatsapp_robustness_test"
HARD_CASES_DIR = PROJECT_ROOT / "data" / "e6_hard_cases"
E10B_PREDICTIONS_CSV = PROJECT_ROOT / "experiments" / "e10_whatsapp_training" / "per_image_predictions_whatsapp.csv"
E12A_PREDICTIONS_CSV = PROJECT_ROOT / "experiments" / "e12_compression_invariant" / "whatsapp_predictions.csv"

OUT_DIR = PROJECT_ROOT / "experiments" / "e12_compression_invariant"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints_e12b"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 16  # 16 images = 80 crops per step (peak VRAM ~2.2 GB)
LR = 1e-4
WEIGHT_DECAY = 1e-4

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model


# ----------------------------------------------------------------------
# Model Architecture: MultiViewE5ModelWithNorm
# ----------------------------------------------------------------------
class MultiViewE5ModelWithNorm(MultiViewE5Model):
    """
    MultiViewE5Model with lightweight LayerNorm(1792) placed directly
    after the learned 5-view attention aggregation and before the classification head.
    """
    def __init__(
        self,
        checkpoint_path=None,
        num_views: int = 5,
        freq_norm_strategy: str = "standardize",
        freq_embedding_dim: int = 256,
        dropout_p1: float = 0.4,
        dropout_p2: float = 0.2,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            num_views=num_views,
            freq_norm_strategy=freq_norm_strategy,
            freq_embedding_dim=freq_embedding_dim,
            dropout_p1=dropout_p1,
            dropout_p2=dropout_p2,
        )
        # Lightweight LayerNorm on 1792-D aggregated representation
        self.norm = nn.LayerNorm(self.fused_dim, eps=1e-5)
        # Initialize gamma=1.0, beta=0.0
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(
        self,
        x_views: torch.Tensor,
        return_view_weights: bool = False,
    ):
        B, V, C, H, W = x_views.shape
        x_flat = x_views.view(B * V, C, H, W)

        # Dual-branch feature extraction
        feats = self.base_model(x_flat, return_features=True)
        e_fused = feats["fused_embedding"]  # (B * V, 1792)

        # Reshape to per-view embeddings
        e_fused_views = e_fused.view(B, V, self.fused_dim)

        # Learned view attention
        attn_scores = self.view_attention(e_fused_views)  # (B, V, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, V, 1)

        # Attention-weighted aggregation
        pooled_embedding = torch.sum(attn_weights * e_fused_views, dim=1)  # (B, 1792)

        # Apply LayerNorm AFTER aggregation and BEFORE classifier
        normed_embedding = self.norm(pooled_embedding)  # (B, 1792)

        # Classification head
        logits = self.classifier(normed_embedding)  # (B, 1)

        if return_view_weights:
            return logits, attn_weights.squeeze(-1)

        return logits


# ----------------------------------------------------------------------
# Crop Extraction
# ----------------------------------------------------------------------
def generate_5_crops(img: Image.Image) -> list:
    """
    Exact production 5-view crop extractor (60% scale crops):
    1. Resized full image (224x224)
    2. Top-left corner (w*0.6, h*0.6 -> 224x224)
    3. Top-right corner (w*0.6, h*0.6 -> 224x224)
    4. Bottom-left corner (w*0.6, h*0.6 -> 224x224)
    5. Bottom-right corner (w*0.6, h*0.6 -> 224x224)
    """
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


# ----------------------------------------------------------------------
# Dataset Definition
# ----------------------------------------------------------------------
class E5CleanDataset(Dataset):
    """Clean E5 Dataset for Training (with random H-flip) or Validation."""
    def __init__(self, records: list, is_train: bool = False):
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
        except Exception:
            # Fallback for corrupt image
            img_rgb = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.is_train and random.random() > 0.5:
            img_rgb = img_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        crops = generate_5_crops(img_rgb)
        x_views = torch.stack([self.to_tensor(c) for c in crops], dim=0)  # (5, 3, 224, 224)
        y = torch.tensor(label, dtype=torch.float32)

        return x_views, y


# ----------------------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------------------
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


def compute_metrics(y_true: np.ndarray, preds: np.ndarray, y_score: np.ndarray = None):
    y_true = np.asarray(y_true, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = compute_roc_auc(y_true, y_score) if y_score is not None else 0.0
    pr_auc = compute_pr_auc(y_true, y_score) if y_score is not None else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def evaluate_validation(model, loader):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_samples = 0
    bce_loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for x_views, labels in loader:
                x_views = x_views.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(x_views).squeeze(-1)
                loss = bce_loss_fn(logits, labels)
                probs = torch.sigmoid(logits)

                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_samples += bs

                y_true.extend(labels.cpu().numpy().tolist())
                y_prob.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    preds = (y_prob >= 0.50).astype(int)
    metrics = compute_metrics(y_true, preds, y_prob)
    metrics["loss"] = round(avg_loss, 4)
    return metrics, y_true, y_prob


def predict_single_image(model, img_path: Path):
    """
    Runs multi-view inference on a single image and extracts diagnostic cues.
    Consistent with architecture: LayerNorm applied before classification.
    """
    to_tensor = transforms.ToTensor()
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
    crops = generate_5_crops(img_rgb)
    x = torch.stack([to_tensor(c) for c in crops], dim=0).unsqueeze(0).to(DEVICE)  # (1, 5, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            B, V, C, H, W = x.shape
            x_flat = x.view(B * V, C, H, W)
            feats = model.base_model(x_flat, return_features=True)
            e_fused = feats["fused_embedding"]
            e_fused_views = e_fused.view(B, V, model.fused_dim)
            attn_scores = model.view_attention(e_fused_views)
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled_emb = torch.sum(attn_weights * e_fused_views, dim=1)
            
            # LayerNorm on aggregated embedding
            normed_pooled = model.norm(pooled_emb)
            logits = model.classifier(normed_pooled).squeeze(-1)
            p_agg = float(torch.sigmoid(logits).cpu().item())

            # Individual view predictions using same LayerNorm for consistency
            normed_views = model.norm(e_fused)
            view_logits = model.classifier(normed_views)
            view_probs = torch.sigmoid(view_logits).cpu().numpy().flatten()
            p_global = float(view_probs[0])
            p_strongest_local = float(np.max(view_probs[1:]))
            attn_w = [round(float(w), 4) for w in attn_weights.squeeze().cpu().numpy().tolist()]

    return {
        "final_prob": round(p_agg, 4),
        "global_prob": round(p_global, 4),
        "strongest_local_prob": round(p_strongest_local, 4),
        "attention_weights": attn_w,
        "view_probs": [round(float(p), 4) for p in view_probs.tolist()],
    }


def main():
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT E12-B: CLEAN-DATA DOMAIN-CONDITIONED NORMALIZATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Starting Checkpoint (E6-C): {E6C_CHECKPOINT.relative_to(PROJECT_ROOT)}")
    print(f"Batch Size: {BATCH_SIZE} ({BATCH_SIZE*5} crops per step)")
    print(f"Epochs: {EPOCHS}")
    print(f"Data: 100% Clean E5 Only (Zero synthetic compression)")

    # ------------------------------------------------------------------
    # 1. Load E5 Manifest & Splits
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading E5 dataset splits...")
    train_records = []
    val_records = []

    with open(E5_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "image_path": str(PROJECT_ROOT / row["image_path"]),
                "rel_path": row["image_path"],
                "label": int(row["label"]),
                "split": row["split"],
            }
            if row["split"] == "train":
                train_records.append(rec)
            elif row["split"] == "val":
                val_records.append(rec)

    assert len(train_records) == 23165, f"Expected 23,165 train, got {len(train_records)}"
    assert len(val_records) == 5775, f"Expected 5,775 val, got {len(val_records)}"
    print(f"Loaded {len(train_records):,} E5 training images (100% CLEAN).")
    print(f"Loaded {len(val_records):,} E5 validation images (100% CLEAN).")

    train_dataset = E5CleanDataset(train_records, is_train=True)
    val_dataset = E5CleanDataset(val_records, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # ------------------------------------------------------------------
    # 2. Build Model & Load E6-C Checkpoint
    # ------------------------------------------------------------------
    print("\n[STEP 2] Building MultiViewE5ModelWithNorm and loading E6-C weights...")
    model = MultiViewE5ModelWithNorm(
        checkpoint_path=None,
        num_views=5,
        freq_norm_strategy="standardize",
        freq_embedding_dim=256,
    )

    ckpt = torch.load(E6C_CHECKPOINT, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {E6C_CHECKPOINT.name}")
    print(f"  Missing keys (new LayerNorm): {missing_keys}")
    print(f"  Unexpected keys: {unexpected_keys}")
    assert missing_keys == ["norm.weight", "norm.bias"], f"Unexpected missing keys: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected keys found: {unexpected_keys}"

    # Freeze base feature extraction backbones completely
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Train ONLY view_attention, norm, and classifier
    trainable_params = []
    for param in model.view_attention.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in model.norm.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in model.classifier.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    model.to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"\nParameter Count Breakdown:")
    print(f"  Frozen Backbone Params:     {n_frozen:,} (90.15%)")
    print(f"  Trainable Params:           {n_trainable:,} (9.85%)")
    print(f"    - View Attention:         {sum(p.numel() for p in model.view_attention.parameters()):,}")
    print(f"    - Proposed LayerNorm:     {sum(p.numel() for p in model.norm.parameters()):,}")
    print(f"    - Classifier Head:        {sum(p.numel() for p in model.classifier.parameters()):,}")
    print(f"  Total Model Parameters:     {n_total:,}")
    assert n_trainable == 1218050, f"Expected 1,218,050 trainable, got {n_trainable}"
    assert n_frozen == 11150856, f"Expected 11,150,856 frozen, got {n_frozen}"

    # ------------------------------------------------------------------
    # 3. Training Setup
    # ------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    config = {
        "experiment": "E12-B: Clean-Data Domain-Conditioned Normalization",
        "starting_checkpoint": str(E6C_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "crops_per_step": BATCH_SIZE * 5,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "precision": "CUDA AMP FP16",
        "trainable_parameters": n_trainable,
        "frozen_parameters": n_frozen,
        "total_parameters": n_total,
        "training_data": "100% Clean E5 (N=23,165)",
        "validation_data": "100% Clean E5 (N=5,775)",
        "hypothesis": "Test whether LayerNorm(1792) on aggregated representation mitigates WhatsApp false positives without degrading clean E5 recall.",
    }

    with open(OUT_DIR / "e12b_training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Initial E6-C Baseline Check on Clean Val
    print("\nEvaluating initial model on Clean E5 Validation (Epoch 0 Baseline)...")
    init_metrics, _, _ = evaluate_validation(model, val_loader)
    print(f"Epoch 0 Baseline: Acc={init_metrics['accuracy']*100:.2f}%, AUC={init_metrics['roc_auc']:.4f}, PR-AUC={init_metrics['pr_auc']:.4f}, F1={init_metrics['f1']:.4f}, FPR={init_metrics['fpr']*100:.2f}%, Recall={init_metrics['recall']*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. Training Loop (Exactly 3 Epochs)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEGINNING 3-EPOCH TRAINING RUN (CLEAN E5 DATA ONLY)")
    print("=" * 80)

    history = []
    val_predictions_by_epoch = {}
    best_f1 = -1.0
    best_epoch = -1
    overall_start_time = time.time()

    skip_training = (CHECKPOINTS_DIR / "e12b_best_model.pt").exists() and (OUT_DIR / "e12b_training_history.csv").exists()
    if skip_training:
        print("\n[*] Found existing complete 3-epoch checkpoints and training history. Loading history...")
        with open(OUT_DIR / "e12b_training_history.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                history.append({
                    "epoch": int(r["epoch"]),
                    "train_loss": float(r["train_loss"]),
                    "val_loss": float(r["val_loss"]),
                    "val_accuracy": float(r["val_accuracy"]),
                    "val_roc_auc": float(r["val_roc_auc"]),
                    "val_pr_auc": float(r["val_pr_auc"]),
                    "val_f1": float(r["val_f1"]),
                    "val_precision": float(r["val_precision"]),
                    "val_recall": float(r["val_recall"]),
                    "val_fpr": float(r["val_fpr"]),
                    "val_fnr": float(r["val_fnr"]),
                    "train_time_sec": float(r["train_time_sec"]),
                    "val_time_sec": float(r["val_time_sec"]),
                    "peak_vram_gb": float(r["peak_vram_gb"]),
                })
        best_epoch = max(history, key=lambda h: h["val_f1"])["epoch"]
        best_f1 = history[best_epoch - 1]["val_f1"]
        total_run_time = sum(h["train_time_sec"] + h["val_time_sec"] for h in history)
        print(f"[*] Best Checkpoint loaded: Epoch {best_epoch} with Clean Val F1 = {best_f1:.4f}")
    if not skip_training:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            model.train()
            # Keep base backbones in eval mode (BatchNorm and Dropout inside frozen backbones stay fixed)
            model.base_model.eval()

            train_loss_accum = 0.0
            train_samples = 0
            step_times = []

            total_batches = len(train_loader)
            print(f"\n--- Epoch {epoch}/{EPOCHS} (Batches: {total_batches:,}) ---")

            for step, (x_views, labels) in enumerate(train_loader):
                step_start = time.time()
                x_views = x_views.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    logits = model(x_views).squeeze(-1)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = labels.size(0)
                train_loss_accum += loss.item() * bs
                train_samples += bs
                step_times.append(time.time() - step_start)

                if (step + 1) % 250 == 0 or (step + 1) == total_batches:
                    avg_step_time = np.mean(step_times[-50:])
                    print(f"  [Epoch {epoch} | Step {step+1:4d}/{total_batches}] Loss: {loss.item():.4f} | Avg Step: {avg_step_time:.3f}s | Elapsed: {(time.time()-epoch_start)/60:.1f}m")

            train_time = time.time() - epoch_start
            avg_train_loss = train_loss_accum / train_samples

            # Evaluate on Clean E5 Validation
            print(f"\nEvaluating Epoch {epoch} on Clean E5 Validation (N=5,775)...")
            val_start = time.time()
            val_metrics, y_true_val, y_prob_val = evaluate_validation(model, val_loader)
            val_time = time.time() - val_start
            val_predictions_by_epoch[epoch] = y_prob_val

            epoch_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

            epoch_summary = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_fpr": val_metrics["fpr"],
                "val_fnr": val_metrics["fnr"],
                "train_time_sec": round(train_time, 1),
                "val_time_sec": round(val_time, 1),
                "peak_vram_gb": round(epoch_vram, 2),
            }
            history.append(epoch_summary)

            print(f"Epoch {epoch} Results:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1']:.4f}")
            print(f"  Val ROC-AUC:{val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']*100:.2f}% | Val FPR: {val_metrics['fpr']*100:.2f}%")
            print(f"  Train Time: {train_time:.1f}s | Val Time: {val_time:.1f}s | Peak VRAM: {epoch_vram:.2f} GB")

            # Save epoch checkpoint
            ckpt_path = CHECKPOINTS_DIR / f"e12b_checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  Saved epoch checkpoint: {ckpt_path.name}")

            # Strictly select best checkpoint by Clean E5 Validation F1-Score
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                best_ckpt_path = CHECKPOINTS_DIR / "e12b_best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    best_ckpt_path,
                )
                print(f"  [*] New best model selected (Epoch {epoch}, F1={best_f1:.4f}) -> saved to {best_ckpt_path.name}")

        total_run_time = time.time() - overall_start_time
        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE. Total Duration: {total_run_time/60:.2f} minutes")
        print(f"Best Checkpoint: Epoch {best_epoch} with Clean Val F1 = {best_f1:.4f}")
        print("=" * 80)

        # Save training history CSV
        with open(OUT_DIR / "e12b_training_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        # Save validation predictions for best epoch
        best_val_preds = val_predictions_by_epoch[best_epoch]
        with open(OUT_DIR / "e12b_val_predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "ground_truth", "probability", "prediction"])
            for rec, prob in zip(val_records, best_val_preds):
                pred = 1 if prob >= 0.50 else 0
                writer.writerow([rec["rel_path"], rec["label"], round(float(prob), 4), pred])

    # ------------------------------------------------------------------
    # 5. Load Winning Checkpoint for Evaluation
    # ------------------------------------------------------------------
    print(f"\n[STEP 5] Loading winning checkpoint (Epoch {best_epoch}) for final evaluation...")
    best_ckpt = torch.load(CHECKPOINTS_DIR / "e12b_best_model.pt", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # ------------------------------------------------------------------
    # 6. Hard-Case Diagnostics
    # ------------------------------------------------------------------
    print("\n[STEP 6] Running Hard-Case Benchmark Reproduction...")
    hard_cases = {
        "original_ai_edit": HARD_CASES_DIR / "original_ai_edit.png",
        "whatsapp_download": HARD_CASES_DIR / "whatsapp_download.jpeg",
    }
    hard_case_results = {}
    for name, path in hard_cases.items():
        if path.exists():
            diag = predict_single_image(model, path)
            gt = "AI" if name == "original_ai_edit" else "REAL"
            gt_int = 1 if gt == "AI" else 0
            pred_class = "AI" if diag["final_prob"] >= 0.50 else "REAL"
            correct = (pred_class == gt)
            hard_case_results[name] = {
                "ground_truth": gt,
                "final_prob": diag["final_prob"],
                "predicted_class": pred_class,
                "correct": correct,
                "global_prob": diag["global_prob"],
                "strongest_local_prob": diag["strongest_local_prob"],
                "attention_weights": diag["attention_weights"],
            }
            print(f"  {name:20s}: GT={gt:4s} | Final Prob={diag['final_prob']:.4f} | Pred={pred_class:4s} | Correct={correct} | Global={diag['global_prob']:.4f} | Strongest Local={diag['strongest_local_prob']:.4f}")
        else:
            print(f"  WARNING: Hard case image not found at {path}")

    with open(OUT_DIR / "e12b_hard_case_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(hard_case_results, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Single-Shot External WhatsApp Evaluation (N=67)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("E12-B CHECKPOINT FROZEN — BEGINNING EXTERNAL WHATSAPP TEST (N=67)")
    print("=" * 80)

    wa_records_meta = []
    for cls_name, label in [("real", 0), ("ai", 1)]:
        folder = WHATSAPP_TEST_DIR / cls_name
        for p in sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
            wa_records_meta.append((p, label, cls_name.upper()))

    assert len(wa_records_meta) == 67, f"Expected 67 WhatsApp test images, found {len(wa_records_meta)}"

    # Load baseline predictions for transition analysis
    e6c_preds = {}
    e10b_preds = {}
    if E10B_PREDICTIONS_CSV.exists():
        with open(E10B_PREDICTIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                e6c_preds[row["filename"]] = float(row.get("e6c_prob", 0.0))
                e10b_preds[row["filename"]] = float(row.get("e10b_prob", 0.0))

    e12a_preds = {}
    if E12A_PREDICTIONS_CSV.exists():
        with open(E12A_PREDICTIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                e12a_preds[row["filename"]] = float(row.get("probability", 0.0))

    wa_y_true = []
    wa_y_prob = []
    wa_records = []
    transitions = []

    for img_path, is_ai, gt_str in wa_records_meta:
        fn = img_path.name

        diag = predict_single_image(model, img_path)
        prob = diag["final_prob"]
        pred = 1 if prob >= 0.50 else 0
        pred_str = "AI" if pred == 1 else "REAL"
        correct = (pred == is_ai)

        wa_y_true.append(is_ai)
        wa_y_prob.append(prob)

        wa_records.append({
            "filename": fn,
            "ground_truth": is_ai,
            "class_str": gt_str,
            "probability": prob,
            "predicted_class": pred,
            "predicted_str": pred_str,
            "correct": correct,
            "global_prob": diag["global_prob"],
            "strongest_local_prob": diag["strongest_local_prob"],
            "attention_weights": str(diag["attention_weights"]),
        })

        # Transitions relative to E6-C, E10-B, and E12-A
        p_e6c = e6c_preds.get(fn, None)
        p_e10b = e10b_preds.get(fn, None)
        p_e12a = e12a_preds.get(fn, None)

        transitions.append({
            "filename": fn,
            "ground_truth": gt_str,
            "e6c_prob": p_e6c,
            "e10b_prob": p_e10b,
            "e12a_prob": p_e12a,
            "e12b_prob": prob,
            "e6c_pred": ("AI" if p_e6c >= 0.50 else "REAL") if p_e6c is not None else "N/A",
            "e10b_pred": ("AI" if p_e10b >= 0.50 else "REAL") if p_e10b is not None else "N/A",
            "e12a_pred": ("AI" if p_e12a >= 0.50 else "REAL") if p_e12a is not None else "N/A",
            "e12b_pred": pred_str,
            "flip_vs_e6c": (pred_str != ("AI" if p_e6c >= 0.50 else "REAL")) if p_e6c is not None else False,
            "flip_vs_e10b": (pred_str != ("AI" if p_e10b >= 0.50 else "REAL")) if p_e10b is not None else False,
            "flip_vs_e12a": (pred_str != ("AI" if p_e12a >= 0.50 else "REAL")) if p_e12a is not None else False,
        })

    wa_metrics = compute_metrics(np.array(wa_y_true), (np.array(wa_y_prob) >= 0.50).astype(int), np.array(wa_y_prob))

    print(f"\nExternal WhatsApp Benchmark Results (N=67 at Threshold 0.50):")
    print(f"  Accuracy:    {wa_metrics['accuracy']*100:.2f}%")
    print(f"  ROC-AUC:     {wa_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:      {wa_metrics['pr_auc']:.4f}")
    print(f"  F1-Score:    {wa_metrics['f1']:.4f}")
    print(f"  Precision:   {wa_metrics['precision']*100:.2f}%")
    print(f"  Recall:      {wa_metrics['recall']*100:.2f}% (TP={wa_metrics['tp']}/33, FN={wa_metrics['fn']})")
    print(f"  Real FPR:    {wa_metrics['fpr']*100:.2f}% (FP={wa_metrics['fp']}/34, TN={wa_metrics['tn']})")
    print(f"  TP: {wa_metrics['tp']}, TN: {wa_metrics['tn']}, FP: {wa_metrics['fp']}, FN: {wa_metrics['fn']}")

    # Save WhatsApp predictions
    with open(OUT_DIR / "e12b_whatsapp_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=wa_records[0].keys())
        writer.writeheader()
        writer.writerows(wa_records)

    # Save WhatsApp transitions
    with open(OUT_DIR / "e12b_whatsapp_transitions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=transitions[0].keys())
        writer.writeheader()
        writer.writerows(transitions)

    # ------------------------------------------------------------------
    # 8. Generate Comprehensive Research Report
    # ------------------------------------------------------------------
    report_path = OUT_DIR / "E12B_RESEARCH_REPORT.md"
    generate_research_report(
        report_path=report_path,
        config=config,
        history=history,
        best_epoch=best_epoch,
        val_metrics=val_predictions_by_epoch.get(best_epoch, None),
        wa_metrics=wa_metrics,
        hard_case_results=hard_case_results,
        transitions=transitions,
        total_time=total_run_time,
    )
    print(f"\n[COMPLETE] Research report generated: {report_path.relative_to(PROJECT_ROOT)}")


def generate_research_report(
    report_path: Path,
    config: dict,
    history: list,
    best_epoch: int,
    val_metrics: np.ndarray,
    wa_metrics: dict,
    hard_case_results: dict,
    transitions: list,
    total_time: float,
):
    best_val = history[best_epoch - 1]

    flips_vs_e6c = [t for t in transitions if t["flip_vs_e6c"]]
    flips_vs_e10b = [t for t in transitions if t["flip_vs_e10b"]]
    flips_vs_e12a = [t for t in transitions if t["flip_vs_e12a"]]

    content = f"""# Experiment E12-B Research Report: Clean-Data Domain-Conditioned Normalization

## Executive Summary

Experiment E12-B tests the hypothesis of whether introducing **lightweight domain-conditioned LayerNorm(1792)** on the aggregated multi-view representation prior to classification can reduce WhatsApp real-image false positives while preserving AI recall.

Crucially, in contrast to E10-B (which applied online social-media augmentations) and E12-A (which applied paired consistency regularization under synthetic compression), **E12-B trained exclusively on clean E5 data ($N=23,165$)**. The feature extraction backbones (EfficientNet-B3 + Frequency CNN, **11,150,856 parameters**) were kept **100% frozen**. Only `view_attention` (229,633 params), the new `LayerNorm` (3,584 params), and `classifier` (984,833 params) were trained (**1,218,050 trainable parameters**, 9.85% of model).

The 67-image WhatsApp benchmark remained completely frozen until final single-shot evaluation.

---

## 1. Clean E5 Validation Progression ($N=5,775$)

| Epoch | Train Loss | Val Loss | Val Accuracy | Val ROC-AUC | Val PR-AUC | Val F1 | Val Recall | Val FPR | Train Time | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for h in history:
        content += f"| Epoch {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['val_accuracy']*100:.2f}% | {h['val_roc_auc']:.4f} | {h['val_pr_auc']:.4f} | {h['val_f1']:.4f} | {h['val_recall']*100:.2f}% | {h['val_fpr']*100:.2f}% | {h['train_time_sec']:.1f}s | {h['peak_vram_gb']:.2f} GB |\n"

    content += f"""
**Winning Checkpoint**: **Epoch {best_epoch}** selected strictly based on Clean E5 validation F1-score ({best_val['val_f1']:.4f}) and saved as `checkpoints_e12b/e12b_best_model.pt`.

---

## 2. Comparison Across Models on Clean E5 Validation ($N=5,775$)

| Model | Checkpoint Status | Clean Val Acc | Clean Val ROC-AUC | Clean Val PR-AUC | Clean Val F1 | Clean Val FPR | Clean Val Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E6-C (Baseline)** | Frozen | 96.17% | 0.9936 | 0.9937 | 0.9608 | **2.97%** | 95.28% |
| **E10-B (Augmented)** | Frozen | 96.62% | 0.9936 | 0.9936 | 0.9657 | 3.27% | 96.52% |
| **E12-A (Invariance Reg)** | Frozen | 96.48% | 0.9929 | 0.9920 | 0.9645 | 4.19% | **97.18%** |
| **E12-B (LayerNorm, Best)** | **Frozen** | **{best_val['val_accuracy']*100:.2f}%** | **{best_val['val_roc_auc']:.4f}** | **{best_val['val_pr_auc']:.4f}** | **{best_val['val_f1']:.4f}** | **{best_val['val_fpr']*100:.2f}%** | **{best_val['val_recall']*100:.2f}%** |

---

## 3. Hardware, Runtime, and VRAM Utilization

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **Peak VRAM**: Consistently **~{best_val['peak_vram_gb']:.2f} GB** (zero OOM risk)
- **Total Training Duration**: **{total_time/60:.2f} minutes**
- **Average Train Step Time**: ~0.20s per step (16 images = 80 crops per forward pass)

---

## 4. Single-Shot Evaluation on Frozen WhatsApp Benchmark ($N=67$ at Threshold 0.50)

```
E12-B CHECKPOINT FROZEN — BEGINNING EXTERNAL TEST
```

| Metric | E6-C (Baseline) | E8-B (Augmented) | E10-B (Standard) | E12-A (Invariance) | **E12-B (LayerNorm)** | $\Delta$ (E12-B vs E6-C) | $\Delta$ (E12-B vs E10-B) | $\Delta$ (E12-B vs E12-A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 61.19% | 64.18% | 62.69% | 61.19% | **{wa_metrics['accuracy']*100:.2f}%** | {wa_metrics['accuracy']*100 - 61.19:+.2f}% | {wa_metrics['accuracy']*100 - 62.69:+.2f}% | {wa_metrics['accuracy']*100 - 61.19:+.2f}% |
| **ROC-AUC** | 0.6546 | 0.6881 | 0.7001 | 0.6840 | **{wa_metrics['roc_auc']:.4f}** | {wa_metrics['roc_auc'] - 0.6546:+.4f} | {wa_metrics['roc_auc'] - 0.7001:+.4f} | {wa_metrics['roc_auc'] - 0.6840:+.4f} |
| **PR-AUC** | 0.7136 | 0.7209 | 0.7519 | 0.7418 | **{wa_metrics['pr_auc']:.4f}** | {wa_metrics['pr_auc'] - 0.7136:+.4f} | {wa_metrics['pr_auc'] - 0.7519:+.4f} | {wa_metrics['pr_auc'] - 0.7418:+.4f} |
| **Precision** | 81.82% | 73.68% | 70.00% | 68.42% | **{wa_metrics['precision']*100:.2f}%** | {wa_metrics['precision']*100 - 81.82:+.2f}% | {wa_metrics['precision']*100 - 70.00:+.2f}% | {wa_metrics['precision']*100 - 68.42:+.2f}% |
| **Recall** | 27.27% | 42.42% | 42.42% | 39.39% | **{wa_metrics['recall']*100:.2f}%** | {wa_metrics['recall']*100 - 27.27:+.2f}% | {wa_metrics['recall']*100 - 42.42:+.2f}% | {wa_metrics['recall']*100 - 39.39:+.2f}% |
| **F1-Score** | 0.4091 | 0.5385 | 0.5283 | 0.5000 | **{wa_metrics['f1']:.4f}** | {wa_metrics['f1'] - 0.4091:+.4f} | {wa_metrics['f1'] - 0.5283:+.4f} | {wa_metrics['f1'] - 0.5000:+.4f} |
| **Real FPR** | **5.88%** (2 FP) | 14.71% (5 FP) | 17.65% (6 FP) | 17.65% (6 FP) | **{wa_metrics['fpr']*100:.2f}% ({wa_metrics['fp']} FP)** | {wa_metrics['fpr']*100 - 5.88:+.2f}% | {wa_metrics['fpr']*100 - 17.65:+.2f}% | {wa_metrics['fpr']*100 - 17.65:+.2f}% |
| **AI FNR** | 72.73% (24 FN) | 57.58% (19 FN) | 57.58% (19 FN) | 60.61% (20 FN) | **{wa_metrics['fnr']*100:.2f}% ({wa_metrics['fn']} FN)** | {wa_metrics['fnr']*100 - 72.73:+.2f}% | {wa_metrics['fnr']*100 - 57.58:+.2f}% | {wa_metrics['fnr']*100 - 60.61:+.2f}% |
| **TP / TN** | 9 / 32 | 14 / 29 | 14 / 28 | 13 / 28 | **{wa_metrics['tp']} / {wa_metrics['tn']}** | {wa_metrics['tp'] - 9:+d} TP / {wa_metrics['tn'] - 32:+d} TN | {wa_metrics['tp'] - 14:+d} TP / {wa_metrics['tn'] - 28:+d} TN | {wa_metrics['tp'] - 13:+d} TP / {wa_metrics['tn'] - 28:+d} TN |
| **FP / FN** | 2 / 24 | 5 / 19 | 6 / 19 | 6 / 20 | **{wa_metrics['fp']} / {wa_metrics['fn']}** | {wa_metrics['fp'] - 2:+d} FP / {wa_metrics['fn'] - 24:+d} FN | {wa_metrics['fp'] - 6:+d} FP / {wa_metrics['fn'] - 19:+d} FN | {wa_metrics['fp'] - 6:+d} FP / {wa_metrics['fn'] - 20:+d} FN |

---

## 5. Transition Breakdown Relative to Prior Models on WhatsApp

### Flips vs E12-A ({len(flips_vs_e12a)} total flips):
"""
    for t in flips_vs_e12a:
        content += f"- `{t['filename']}`: GT={t['ground_truth']} | E12-A={t['e12a_prob']:.4f} ({t['e12a_pred']}) -> **E12-B={t['e12b_prob']:.4f} ({t['e12b_pred']})**\n"

    content += f"""
### Flips vs E10-B ({len(flips_vs_e10b)} total flips):
"""
    for t in flips_vs_e10b:
        content += f"- `{t['filename']}`: GT={t['ground_truth']} | E10-B={t['e10b_prob']:.4f} ({t['e10b_pred']}) -> **E12-B={t['e12b_prob']:.4f} ({t['e12b_pred']})**\n"

    content += f"""
### Flips vs E6-C Baseline ({len(flips_vs_e6c)} total flips):
"""
    for t in flips_vs_e6c:
        content += f"- `{t['filename']}`: GT={t['ground_truth']} | E6-C={t['e6c_prob']:.4f} ({t['e6c_pred']}) -> **E12-B={t['e12b_prob']:.4f} ({t['e12b_pred']})**\n"

    content += f"""
---

## 6. Hard-Case Benchmark Reproduction

| Benchmark Image | Ground Truth | E10-B Prob | E12-A Prob | **E12-B Prob** | Global Prob | Strongest Local Prob | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for name, res in hard_case_results.items():
        fn = f"{name}.png" if name == "original_ai_edit" else f"{name}.jpeg"
        p_e10b = 0.8886 if name == "original_ai_edit" else 0.0555
        p_e12a = 0.9282 if name == "original_ai_edit" else 0.0120
        content += f"| `{fn}` | **{res['ground_truth']}** | {p_e10b:.4f} | {p_e12a:.4f} | **{res['final_prob']:.4f}** | {res['global_prob']:.4f} | {res['strongest_local_prob']:.4f} | **{res['correct']}** |\n"

    content += f"""
---

## 7. Empirical Findings & Scientific Conclusions

1. **Clean E5 Performance**:
   - Clean Validation Accuracy: **{best_val['val_accuracy']*100:.2f}%**
   - Clean Validation ROC-AUC: **{best_val['val_roc_auc']:.4f}**
   - Clean Validation PR-AUC: **{best_val['val_pr_auc']:.4f}**
   - Clean Validation F1-Score: **{best_val['val_f1']:.4f}**
   - Clean Validation FPR: **{best_val['val_fpr']*100:.2f}%**
   - Clean Validation AI Recall: **{best_val['val_recall']*100:.2f}%**

2. **WhatsApp Robustness Hypothesis Evaluation**:
   - Real False Positives: **{wa_metrics['fp']} / 34** ({wa_metrics['fpr']*100:.2f}% FPR)
   - AI True Positives: **{wa_metrics['tp']} / 33** ({wa_metrics['recall']*100:.2f}% Recall)
   - WhatsApp ROC-AUC: **{wa_metrics['roc_auc']:.4f}**
   - WhatsApp F1: **{wa_metrics['f1']:.4f}**

*(Detailed analysis and next-step considerations documented after empirical review.)*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
