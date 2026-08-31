"""
Leave-One-Generator-Out (LOGO) Out-of-Distribution (OOD) Development Engine.

This module implements:
1. 5-Fold Leave-One-Generator-Out (LOGO) cross-validation on the 5 development
   generators (ADM, GLIDE, SDv5, VQDM, Wukong).
2. Frequency-normalization ablation across candidate strategies:
   - minmax: per-sample min-max normalization to [0, 1]
   - standardize: per-sample z-score normalization (zero-mean, unit-variance)
   - none: raw unnormalized log-magnitude spectrum
3. Strict zero-leakage assertions ensuring BigGAN and Midjourney are never touched.
4. Comprehensive reporting of fold-level and aggregate statistics.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

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
from ml.train import (
    build_parameter_groups,
    compute_roc_auc,
    evaluate_epoch,
    get_raw_rgb_transform,
    set_seed,
    train_one_epoch,
)
from ml.evaluate import compute_metrics
from models.fusion import build_model


# ── Strict Generator Constants ────────────────────────────────────────────────
DEV_GENERATORS: List[str] = ["ADM", "GLIDE", "SDv5", "VQDM", "Wukong"]
FORBIDDEN_TEST_GENERATORS: Set[str] = {"BigGAN", "Midjourney"}

# Supported normalization strategies
NORM_STRATEGIES: List[str] = ["minmax", "standardize", "none"]


def assert_zero_test_leakage(records: List[Dict], context: str = "Dataset") -> None:
    """Verifies that no test generators (BigGAN, Midjourney) exist in the records."""
    gens_present = {r.get("generator") for r in records}
    leakage = gens_present.intersection(FORBIDDEN_TEST_GENERATORS)
    if leakage:
        raise RuntimeError(
            f"FATAL LEAKAGE DETECTED in {context}: Forbidden test generator(s) {leakage} found! "
            f"BigGAN and Midjourney must NEVER be accessed during OOD development."
        )


def get_logo_folds(generators: Optional[List[str]] = None) -> List[Dict[str, Union[int, str, List[str]]]]:
    """
    Constructs the 5 canonical Leave-One-Generator-Out folds.
    
    Each fold holds out exactly one generator for validation and uses
    the remaining four for training.
    """
    gens = generators or DEV_GENERATORS
    # Enforce forbidden generator assertion
    if any(g in FORBIDDEN_TEST_GENERATORS for g in gens):
        raise ValueError(f"Forbidden test generators in fold list: {gens}")

    folds = []
    for i, held_out in enumerate(sorted(gens), start=1):
        train_gens = [g for g in sorted(gens) if g != held_out]
        folds.append({
            "fold_index": i,
            "held_out_generator": held_out,
            "train_generators": train_gens,
        })
    return folds


@torch.no_grad()
def evaluate_model_on_records(
    model: nn.Module,
    records: List[Dict],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    num_workers: int = 2,
    transform: Optional[transforms.Compose] = None,
) -> Dict[str, Union[float, int, Dict[str, int]]]:
    """
    Evaluates a model over a specific list of manifest records.
    
    Returns standard binary metrics (Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1, CM).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if transform is None:
        transform = get_raw_rgb_transform()

    # Build an in-memory dataset over records
    dataset = DeepVisionDataset(split=None, transform=transform)
    dataset.records = records

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logit"]
                probs = torch.sigmoid(logits)
        else:
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["logit"]
            probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)
    return metrics


def run_single_logo_fold(
    held_out_generator: str,
    norm_strategy: str = "minmax",
    batch_size: int = 16,
    epochs: int = 10,
    lr_backbone: float = 1e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    seed: int = 42,
    save_dir: Optional[Path] = None,
    num_workers: int = 2,
    use_amp: bool = True,
) -> Dict:
    """
    Trains and evaluates a single Leave-One-Generator-Out fold.
    
    Training: 4 remaining development generators (split='train').
    Validation: held-out development generator (split='val').
    """
    if held_out_generator in FORBIDDEN_TEST_GENERATORS:
        raise RuntimeError(f"FATAL: Attempted to run LOGO with forbidden test generator {held_out_generator}")

    set_seed(seed)
    train_gens = [g for g in DEV_GENERATORS if g != held_out_generator]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. GPU training required.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    raw_transform = get_raw_rgb_transform()

    # Load training data (4 generators, train split)
    train_dataset = DeepVisionDataset(
        split="train",
        generators=train_gens,
        transform=raw_transform,
    )
    assert_zero_test_leakage(train_dataset.records, f"Fold [{held_out_generator}] Train Dataset")

    # Load validation data (held-out generator, val split)
    val_dataset = DeepVisionDataset(
        split="val",
        generators=[held_out_generator],
        transform=raw_transform,
    )
    assert_zero_test_leakage(val_dataset.records, f"Fold [{held_out_generator}] Val Dataset")

    # Verify held-out generator is NOT in training records
    train_gen_set = {r.get("generator") for r in train_dataset.records}
    if held_out_generator in train_gen_set:
        raise RuntimeError(
            f"LEAKAGE VIOLATION: Held-out generator {held_out_generator} appeared in training data: {train_gen_set}"
        )

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

    print(f"\n[{norm_strategy.upper()} | Held-out: {held_out_generator}]")
    print(f"  Train samples: {len(train_dataset):,} (Generators: {train_gens})")
    print(f"  Val samples:   {len(val_dataset):,} (Held-out: {held_out_generator})")

    # Build E3 dual-branch model with specified norm strategy
    model = build_model(
        experiment="E3",
        pretrained=True,
        freq_norm_strategy=norm_strategy,
        freq_embedding_dim=256,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    param_groups = build_parameter_groups(
        model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

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
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
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
        bb_lr = optimizer.param_groups[0]["lr"]
        hd_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else bb_lr

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_roc_auc": round(val_auc, 4),
            "lr_backbone": float(f"{bb_lr:.2e}"),
            "lr_head": float(f"{hd_lr:.2e}"),
            "epoch_duration_sec": round(epoch_time, 1),
        }
        history.append(epoch_record)

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Best checkpoint selection strictly via validation ROC-AUC on held-out generator
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {
                "norm_strategy": norm_strategy,
                "held_out_generator": held_out_generator,
                "best_epoch": epoch,
                "val_roc_auc": round(val_auc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            }
            if save_dir is not None:
                checkpoint_file = save_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "norm_strategy": norm_strategy,
                    "held_out_generator": held_out_generator,
                    "train_generators": train_gens,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "experiment": "E3",
                        "norm_strategy": norm_strategy,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr_backbone": lr_backbone,
                        "lr_head": lr_head,
                        "weight_decay": weight_decay,
                        "seed": seed,
                        "held_out_generator": held_out_generator,
                        "train_generators": train_gens,
                    },
                }, checkpoint_file)

    total_time = time.time() - start_time
    print(f"  --> Best Epoch: {best_metrics.get('best_epoch')} | Best Val AUC: {best_metrics.get('val_roc_auc'):.4f} (Total {total_time:.1f}s)")

    # Load best checkpoint and compute full classification metrics on validation split
    best_model = build_model(
        experiment="E3",
        pretrained=False,
        freq_norm_strategy=norm_strategy,
        freq_embedding_dim=256,
    )
    if save_dir is not None and (save_dir / "best_model.pt").exists():
        ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
        best_model.load_state_dict(ckpt["model_state_dict"])
    else:
        best_model.load_state_dict(model.state_dict())

    val_metrics = evaluate_model_on_records(
        model=best_model,
        records=val_dataset.records,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        transform=raw_transform,
    )

    fold_result = {
        "norm_strategy": norm_strategy,
        "held_out_generator": held_out_generator,
        "train_generators": train_gens,
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "best_epoch": best_metrics.get("best_epoch"),
        "metrics": val_metrics,
        "training_time_sec": round(total_time, 1),
    }

    if save_dir is not None:
        with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(save_dir / "fold_summary.json", "w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    return fold_result


def run_full_ood_development(
    strategies: Optional[List[str]] = None,
    output_dir: str = "experiments/ood_development",
    batch_size: int = 16,
    epochs: int = 10,
    lr_backbone: float = 1e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 2,
    seed: int = 42,
    num_workers: int = 2,
    use_amp: bool = True,
) -> Dict:
    """
    Executes the complete Leave-One-Generator-Out (LOGO) cross-validation protocol
    across all 5 folds and candidate normalization strategies.
    
    Generates:
    - protocol_config.json
    - fold-level checkpoints and summaries
    - logo_results.json
    - logo_summary_table.csv
    """
    strategies = strategies or NORM_STRATEGIES
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DEEPVISION FORENSICS — LEAVE-ONE-GENERATOR-OUT (LOGO) DEVELOPMENT PROTOCOL")
    print("=" * 70)
    print(f"Development Generators (5): {DEV_GENERATORS}")
    print(f"Forbidden Test Generators:  {sorted(list(FORBIDDEN_TEST_GENERATORS))} (100% Isolated)")
    print(f"Normalization Strategies:   {strategies}")
    print(f"Random Seed:                {seed}")
    print(f"Batch Size:                 {batch_size}")
    print(f"Epochs per Fold:            {epochs}")
    print(f"Save Directory:             {out_path}")
    print("=" * 70)

    # Save protocol configuration
    protocol_config = {
        "protocol_name": "5-Fold Leave-One-Generator-Out (LOGO) Cross-Validation",
        "development_generators": DEV_GENERATORS,
        "forbidden_test_generators": sorted(list(FORBIDDEN_TEST_GENERATORS)),
        "strategies": strategies,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "weight_decay": weight_decay,
        "seed": seed,
        "base_architecture": "E3 (Spatial EfficientNet-B3 + Spectral 4-Block CNN)",
    }
    with open(out_path / "protocol_config.json", "w", encoding="utf-8") as f:
        json.dump(protocol_config, f, indent=2)

    folds = get_logo_folds(DEV_GENERATORS)
    all_results: Dict[str, Dict] = {}

    total_start = time.time()

    for strat in strategies:
        print(f"\n=======================================================")
        print(f"RUNNING STRATEGY: {strat.upper()}")
        print(f"=======================================================")
        strat_results = []
        strat_dir = out_path / strat
        strat_dir.mkdir(parents=True, exist_ok=True)

        for fold in folds:
            held_out = str(fold["held_out_generator"])
            fold_dir = strat_dir / f"fold_{held_out.lower()}"

            fold_result = run_single_logo_fold(
                held_out_generator=held_out,
                norm_strategy=strat,
                batch_size=batch_size,
                epochs=epochs,
                lr_backbone=lr_backbone,
                lr_head=lr_head,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                seed=seed,
                save_dir=fold_dir,
                num_workers=num_workers,
                use_amp=use_amp,
            )
            strat_results.append(fold_result)

        # Compute aggregate statistics for this normalization strategy
        auc_values = [f["metrics"]["roc_auc"] for f in strat_results]
        acc_values = [f["metrics"]["accuracy"] for f in strat_results]
        pr_auc_values = [f["metrics"]["pr_auc"] for f in strat_results]
        f1_values = [f["metrics"]["f1_score"] for f in strat_results]
        prec_values = [f["metrics"]["precision"] for f in strat_results]
        rec_values = [f["metrics"]["recall"] for f in strat_results]

        agg_stats = {
            "mean_roc_auc": round(float(np.mean(auc_values)), 4),
            "std_roc_auc": round(float(np.std(auc_values)), 4),
            "worst_case_roc_auc": round(float(np.min(auc_values)), 4),
            "best_case_roc_auc": round(float(np.max(auc_values)), 4),
            "mean_accuracy": round(float(np.mean(acc_values)), 4),
            "std_accuracy": round(float(np.std(acc_values)), 4),
            "mean_pr_auc": round(float(np.mean(pr_auc_values)), 4),
            "mean_f1": round(float(np.mean(f1_values)), 4),
            "mean_precision": round(float(np.mean(prec_values)), 4),
            "mean_recall": round(float(np.mean(rec_values)), 4),
        }

        all_results[strat] = {
            "strategy": strat,
            "folds": strat_results,
            "aggregate_statistics": agg_stats,
        }

    total_duration = time.time() - total_start

    # Save comprehensive machine-readable results
    full_output = {
        "protocol_config": protocol_config,
        "results": all_results,
        "total_duration_sec": round(total_duration, 1),
        "total_duration_min": round(total_duration / 60.0, 2),
    }

    json_path = out_path / "logo_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[*] Saved full LOGO results to {json_path}")

    # Generate summary CSV table
    csv_path = out_path / "logo_summary_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy", "Held_Out_Generator", "Train_Generators", "Train_N", "Val_N",
            "Best_Epoch", "Val_Accuracy", "Val_ROC_AUC", "Val_PR_AUC",
            "Val_Precision", "Val_Recall", "Val_F1", "TP", "FP", "TN", "FN"
        ])
        for strat in strategies:
            for fold in all_results[strat]["folds"]:
                m = fold["metrics"]
                writer.writerow([
                    strat,
                    fold["held_out_generator"],
                    "+".join(fold["train_generators"]),
                    fold["train_sample_count"],
                    fold["val_sample_count"],
                    fold["best_epoch"],
                    f"{m['accuracy']:.4f}",
                    f"{m['roc_auc']:.4f}",
                    f"{m['pr_auc']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1_score']:.4f}",
                    m["tp"], m["fp"], m["tn"], m["fn"]
                ])

    print(f"[*] Saved LOGO summary CSV table to {csv_path}")

    # Print summary leaderboard
    print("\n" + "=" * 75)
    print("LEAVE-ONE-GENERATOR-OUT (LOGO) DEVELOPMENT LEADERBOARD")
    print("=" * 75)
    print(f"{'Strategy':<14} | {'Mean ROC-AUC':<12} | {'Std AUC':<8} | {'Worst AUC':<10} | {'Best AUC':<10} | {'Mean Acc':<10} | {'Mean F1':<8}")
    print("-" * 75)

    ranked_strats = sorted(
        strategies,
        key=lambda s: all_results[s]["aggregate_statistics"]["mean_roc_auc"],
        reverse=True,
    )

    for strat in ranked_strats:
        st = all_results[strat]["aggregate_statistics"]
        print(
            f"{strat:<14} | "
            f"{st['mean_roc_auc']:<12.4f} | "
            f"{st['std_roc_auc']:<8.4f} | "
            f"{st['worst_case_roc_auc']:<10.4f} | "
            f"{st['best_case_roc_auc']:<10.4f} | "
            f"{st['mean_accuracy']:<10.4f} | "
            f"{st['mean_f1']:<8.4f}"
        )
    print("=" * 75)
    print(f"Top-Ranked Strategy: Candidate [{ranked_strats[0]}] with Mean LOGO ROC-AUC = {all_results[ranked_strats[0]]['aggregate_statistics']['mean_roc_auc']:.4f}")
    print(f"Total Execution Time: {total_duration/60.0:.2f} minutes")
    print("=" * 75)

    return full_output


def parse_args():
    parser = argparse.ArgumentParser(description="DeepVision-Forensics LOGO OOD Development Suite")
    parser.add_argument("--strategies", nargs="+", default=["minmax", "standardize", "none"],
                        choices=["minmax", "standardize", "instance_norm", "none"],
                        help="Normalization strategies to ablate")
    parser.add_argument("--output-dir", type=str, default="experiments/ood_development",
                        help="Output directory for development artifacts")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per fold")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Spatial backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Scratch/head learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_ood_development(
        strategies=args.strategies,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
