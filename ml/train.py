"""
Training pipeline for DeepVision-Forensics.

Supports controlled experiments:
- E1: EfficientNet-B3 spatial-only
- E2: Frequency-only FFT model
- E3: Dual-branch Spatial + Frequency Fusion

Features:
- Automatic CUDA acceleration
- Mixed precision training with torch.amp
- Differential learning rates with AdamW
- Linear warmup followed by Cosine Annealing decay
- Best checkpoint selection strictly via validation ROC-AUC
- Zero test split leakage
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ml.data import config
from ml.data.dataset import DeepVisionDataset
from models.fusion import build_model


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_raw_rgb_transform() -> transforms.Compose:
    """
    Returns transformation pipeline that produces unnormalized [0, 1] RGB tensors.
    
    Both spatial (with internal ImageNet normalization) and frequency branches
    receive this exact unnormalized tensor.
    """
    return transforms.Compose([
        transforms.Resize(config.TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def build_parameter_groups(
    model: nn.Module,
    lr_backbone: float = 1e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-4,
) -> List[Dict]:
    """
    Partitions model parameters into differential learning rate groups:
    - Pretrained EfficientNet-B3 backbone: lower learning rate (lr_backbone)
    - Newly initialized scratch layers (frequency branch, heads): higher learning rate (lr_head)
    """
    backbone_params = []
    scratch_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if parameter belongs to the spatial backbone feature extractor
        if "spatial_branch.features" in name or "spatial_branch.backbone" in name:
            backbone_params.append(param)
        else:
            scratch_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
            "name": "spatial_backbone",
        })
    if scratch_params:
        param_groups.append({
            "params": scratch_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "name": "scratch_layers",
        })

    return param_groups


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes Receiver Operating Characteristic Area Under the Curve (ROC-AUC).
    Compatible with NumPy 1.x and 2.x.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Sort scores descending
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # Trapezoidal integration compatible with NumPy 1.x and 2.x
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    elif hasattr(np, "trapz"):
        return float(np.trapz(tpr, fpr))
    return float(np.sum((tpr[1:] + tpr[:-1]) * 0.5 * np.diff(fpr)))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
) -> float:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logit"]
                loss = criterion(logits, labels)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["logit"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, float]:
    """
    Evaluates the model on validation data.
    
    Returns:
        (avg_loss, accuracy, roc_auc)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logit"]
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["logit"]
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())

    avg_loss = total_loss / max(total_samples, 1)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(np.mean(y_true == y_pred))
    roc_auc = compute_roc_auc(y_true, y_prob)

    return avg_loss, accuracy, roc_auc


def run_training(
    experiment: str = "E3",
    batch_size: int = 16,
    epochs: int = 10,
    lr_backbone: float = 1e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    norm_strategy: str = "minmax",
    freq_embedding_dim: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    save_dir: str = "experiments/e1_spatial",
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Configures, verifies, and coordinates model training.
    """
    import time
    start_time = time.time()
    set_seed(seed)

    # 1. Device Safety Verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. Training aborted as CPU training is not permitted.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    torch.cuda.reset_peak_memory_stats(0)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DEEPVISION FORENSICS — EXPERIMENT {experiment} TRAINING")
    print("=" * 60)
    print(f"Device:           CUDA ({gpu_name})")
    print(f"CUDA Version:     {cuda_version}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Mixed Precision:  {'Enabled (torch.amp float16)' if use_amp else 'Disabled'}")
    print(f"Batch Size:       {batch_size}")
    print(f"Epochs:           {epochs} (Warmup: {warmup_epochs} epochs)")
    print(f"Random Seed:      {seed}")
    print(f"Save Directory:   {save_path}")

    # 2. Data Split Safety Verification
    raw_transform = get_raw_rgb_transform()
    train_dataset = DeepVisionDataset(split="train", transform=raw_transform)
    val_dataset = DeepVisionDataset(split="val", transform=raw_transform)

    train_gens = sorted(list({r.get("generator") for r in train_dataset.records}))
    val_gens = sorted(list({r.get("generator") for r in val_dataset.records}))

    print("-" * 60)
    print("DATASET & GENERATOR VERIFICATION:")
    print(f"Train samples:    {len(train_dataset):,} records")
    print(f"Train generators: {train_gens}")
    print(f"Val samples:      {len(val_dataset):,} records")
    print(f"Val generators:   {val_gens}")

    # Explicit holdout verification
    forbidden_gens = {"BigGAN", "Midjourney"}
    if any(g in forbidden_gens for g in train_gens):
        raise RuntimeError(f"FATAL: Unseen test generator found in training split: {train_gens}")
    if any(g in forbidden_gens for g in val_gens):
        raise RuntimeError(f"FATAL: Unseen test generator found in validation split: {val_gens}")

    print("Holdout Check:    PASSED (BigGAN & Midjourney are 100% invisible)")
    print("Test DataLoader:  NOT constructed or accessed (Zero Test Leakage)")
    print("-" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model Architecture Verification
    model = build_model(
        experiment=experiment,
        pretrained=True,
        freq_norm_strategy=norm_strategy,
        freq_embedding_dim=freq_embedding_dim,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("MODEL ARCHITECTURE:")
    print(f"Model Class:      {model.__class__.__name__}")
    if hasattr(model, "spatial_branch"):
        sp_params = sum(p.numel() for p in model.spatial_branch.parameters())
        print(f"Spatial Params:   {sp_params:,}")
    if hasattr(model, "frequency_branch"):
        fq_params = sum(p.numel() for p in model.frequency_branch.parameters())
        print(f"Frequency Params: {fq_params:,}")
    if hasattr(model, "classifier"):
        cl_params = sum(p.numel() for p in model.classifier.parameters())
        print(f"Head Params:      {cl_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Total Params:     {total_params:,}")
    print(f"Initial LR:       Backbone={lr_backbone:.1e}, Head/Scratch={lr_head:.1e}")
    print(f"Weight Decay:     {weight_decay:.1e}")
    print(f"Freq Normalization: {norm_strategy}")
    print("=" * 60)

    criterion = nn.BCEWithLogitsLoss()
    param_groups = build_parameter_groups(
        model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Learning rate schedule: Linear Warmup followed by Cosine Decay
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    best_val_auc = -1.0
    best_metrics = {}
    history = []

    print(f"Beginning {epochs} training epochs on {gpu_name}...\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lrs = [group["lr"] * scheduler.get_last_lr()[0] for group in optimizer.param_groups]
        bb_lr = current_lrs[0] if len(current_lrs) > 0 else lr_backbone
        hd_lr = current_lrs[1] if len(current_lrs) > 1 else lr_head

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss, val_acc, val_auc = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        vram_peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_roc_auc": round(val_auc, 4),
            "lr_backbone": float(f"{bb_lr:.2e}"),
            "lr_head": float(f"{hd_lr:.2e}"),
            "vram_allocated_mb": round(vram_allocated_mb, 1),
            "vram_peak_mb": round(vram_peak_mb, 1),
            "epoch_duration_sec": round(epoch_time, 1),
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"LR: [bb={bb_lr:.1e}, hd={hd_lr:.1e}] | "
            f"VRAM: {vram_allocated_mb:.0f}MB (Peak: {vram_peak_mb:.0f}MB)"
        )

        # Checkpoint selection strictly by validation ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {
                "experiment": experiment,
                "best_epoch": epoch,
                "best_val_roc_auc": round(val_auc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "vram_peak_mb": round(vram_peak_mb, 1),
            }
            checkpoint_file = save_path / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "experiment": experiment,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "experiment": experiment,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr_backbone": lr_backbone,
                    "lr_head": lr_head,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "norm_strategy": norm_strategy,
                    "freq_embedding_dim": freq_embedding_dim,
                }
            }, checkpoint_file)
            print(f"  --> [*] Saved new best checkpoint to {checkpoint_file} (Val AUC: {val_auc:.4f})")

    total_duration = time.time() - start_time
    best_metrics["total_duration_sec"] = round(total_duration, 1)
    best_metrics["total_duration_min"] = round(total_duration / 60.0, 2)

    # Save training history and summary JSON
    import json
    with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(save_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    print("=" * 60)
    print(f"TRAINING COMPLETE — EXPERIMENT {experiment}")
    print(f"Total Duration:   {total_duration/60.0:.2f} minutes")
    print(f"Best Epoch:       {best_metrics.get('best_epoch')}")
    print(f"Best Val ROC-AUC: {best_metrics.get('best_val_roc_auc'):.4f}")
    print(f"Best Val Acc:     {best_metrics.get('val_accuracy'):.4f}")
    print(f"Best Val Loss:    {best_metrics.get('val_loss'):.4f}")
    print(f"Peak VRAM:        {best_metrics.get('vram_peak_mb'):.1f} MB")
    print("=" * 60)

    return best_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="DeepVision-Forensics Training Script")
    parser.add_argument("--experiment", type=str, choices=["E1", "E2", "E3"], default="E1",
                        help="Experiment configuration: E1 (Spatial), E2 (Frequency), E3 (Fusion)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (conservative default 16 for RTX 3050 4GB)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for pretrained spatial backbone")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for newly initialized layers")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--norm-strategy", type=str, default="minmax",
                        choices=["minmax", "standardize", "instance_norm", "none"],
                        help="Frequency normalization strategy")
    parser.add_argument("--freq-embedding-dim", type=int, default=256, help="Frequency embedding dimension")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=str, default="experiments/e1_spatial", help="Directory to save checkpoints")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        experiment=args.experiment,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        norm_strategy=args.norm_strategy,
        freq_embedding_dim=args.freq_embedding_dim,
        num_workers=args.num_workers,
        seed=args.seed,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
    )
