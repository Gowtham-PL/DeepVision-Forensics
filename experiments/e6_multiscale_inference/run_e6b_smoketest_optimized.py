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

PROJECT_ROOT = Path("d:/DeepVision-Forensics")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_5_crops(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    
    crops = []
    # View 0: Global image resized to 224x224
    crops.append(img.resize((224, 224), Image.Resampling.BILINEAR))
    # View 1: Top-Left
    crops.append(img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR))
    # View 2: Top-Right
    crops.append(img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR))
    # View 3: Bottom-Left
    crops.append(img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR))
    # View 4: Bottom-Right
    crops.append(img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR))
    
    return crops

class E6BMultiViewTrainDataset(Dataset):
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

            # Training spatial random horizontal flip applied consistently across all views
            if self.is_train and random.random() > 0.5:
                img_rgb = img_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            crops = generate_5_crops(img_rgb)
            view_tensors = [self.to_tensor(c) for c in crops]
            stacked_views = torch.stack(view_tensors, dim=0) # (5, 3, 224, 224)
            return stacked_views, torch.tensor(label, dtype=torch.float32), True
        except Exception as exc:
            dummy_views = torch.zeros((5, 3, 224, 224), dtype=torch.float32)
            return dummy_views, torch.tensor(label, dtype=torch.float32), False

def main():
    print("=" * 70, flush=True)
    print("E6-B: OPTIMIZED MULTI-VIEW TRAINING BENCHMARK (1 EPOCH / 500 IMAGES)", flush=True)
    print("=" * 70, flush=True)

    checkpoint_path = PROJECT_ROOT / "experiments/e5_generalization/best_model.pt"
    print(f"[*] Loading E5 checkpoint initialization from: {checkpoint_path}", flush=True)
    
    # Instantiate E6-B MultiView Model
    model = MultiViewE5Model(
        checkpoint_path=checkpoint_path,
        num_views=5,
        freq_norm_strategy="standardize",
        freq_embedding_dim=256,
    )
    model.to(DEVICE)
    model.train()

    # Load 500 training records from E5 manifest
    manifest_path = PROJECT_ROOT / "data/e5_external/manifests/e5_manifest.csv"
    train_records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                train_records.append({
                    "image_path": str(PROJECT_ROOT / row["image_path"]),
                    "label": int(row["label"])
                })

    print(f"[*] Total training manifest records available: {len(train_records):,}", flush=True)
    
    # Select deterministic subset of 500 images for smoke test
    random.seed(42)
    smoke_subset = random.sample(train_records, 500)
    print(f"[*] Smoke test subset selected: {len(smoke_subset)} images.", flush=True)

    batch_size = 8
    dataset = E6BMultiViewTrainDataset(smoke_subset, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    print(f"\n[*] Starting 1-epoch OPTIMIZED smoke test on {len(smoke_subset)} images...", flush=True)
    print(f"[*] Optimizations: CUDA AMP FP16, Pin Memory, Async Transfer, Stacked Tensor Batches.", flush=True)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
    t_start = time.time()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    gradient_checks = {
        "spatial_branch": False,
        "frequency_branch": False,
        "view_attention": False,
        "classifier": False
    }

    errors = []

    try:
        for batch_idx, (x_views, y_true, valid) in enumerate(loader):
            mask = valid.bool()
            if not mask.any():
                continue

            x_views = x_views[mask].to(DEVICE, non_blocking=True) # (B, 5, 3, 224, 224)
            y_true = y_true[mask].to(DEVICE, non_blocking=True).unsqueeze(1) # (B, 1)

            optimizer.zero_grad()

            # Forward pass under CUDA AMP autocast
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits, view_weights = model(x_views, return_view_weights=True)
                loss = criterion(logits, y_true)

            # Backward pass under GradScaler
            scaler.scale(loss).backward()

            # Unscale optimizer gradients prior to checking gradient magnitudes
            scaler.unscale_(optimizer)

            # Verify gradients across key components
            if not gradient_checks["spatial_branch"]:
                param = next(model.base_model.spatial_branch.parameters())
                if param.grad is not None and torch.abs(param.grad).sum() > 0:
                    gradient_checks["spatial_branch"] = True

            if not gradient_checks["frequency_branch"]:
                param = next(model.base_model.frequency_branch.parameters())
                if param.grad is not None and torch.abs(param.grad).sum() > 0:
                    gradient_checks["frequency_branch"] = True

            if not gradient_checks["view_attention"]:
                param = next(model.view_attention.parameters())
                if param.grad is not None and torch.abs(param.grad).sum() > 0:
                    gradient_checks["view_attention"] = True

            if not gradient_checks["classifier"]:
                param = next(model.classifier.parameters())
                if param.grad is not None and torch.abs(param.grad).sum() > 0:
                    gradient_checks["classifier"] = True

            # Optimizer step & scaler update
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y_true)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.50).float()
            total_correct += int((preds == y_true).sum().item())
            total_samples += len(y_true)
            num_batches += 1

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(loader):
                avg_b_loss = total_loss / max(1, total_samples)
                acc = total_correct / max(1, total_samples)
                vw = view_weights[0].detach().cpu().numpy().round(3)
                print(f"  Batch {batch_idx + 1:2d}/{len(loader)} | Loss: {avg_b_loss:.4f} | Acc: {acc*100:5.2f}% | View Weights: {vw}", flush=True)

    except Exception as exc:
        print(f"[!] Smoke test failed with exception: {exc}", flush=True)
        errors.append(str(exc))

    t_total_sec = time.time() - t_start
    t_total_min = round(t_total_sec / 60.0, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
    peak_vram_gb = round(peak_vram_mb / 1024.0, 2)
    images_per_sec = round(total_samples / max(0.001, t_total_sec), 2)

    # Estimate Full E6-B 10-Epoch Training Runtime across full 33,659 train set
    full_dataset_size = len(train_records) # 33,659
    total_images_10_epochs = full_dataset_size * 10 # 336,590 images
    est_full_training_sec = total_images_10_epochs / max(1.0, images_per_sec)
    est_full_training_hours = round(est_full_training_sec / 3600.0, 2)

    # Baseline comparison (Current unoptimized smoke test)
    baseline_runtime_sec = 1183.93
    baseline_throughput = 0.42
    baseline_vram_gb = 7.89
    baseline_10ep_hours = 64.0

    speedup_factor = round(images_per_sec / max(0.001, baseline_throughput), 2)
    vram_reduction_pct = round((1.0 - (peak_vram_mb / (baseline_vram_gb * 1024))) * 100, 1)

    print("\n" + "=" * 70, flush=True)
    print("E6-B OPTIMIZED SMOKE TEST BENCHMARK RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"1. 500-Image Benchmark Runtime:    {t_total_sec:.2f} s ({t_total_min} min) [Baseline: 1,183.93 s / 19.73 min]", flush=True)
    print(f"2. Processing Throughput:          {images_per_sec} img/s [Baseline: 0.42 img/s -> {speedup_factor}x Speedup]", flush=True)
    print(f"3. Peak GPU VRAM Usage:            {peak_vram_mb} MB ({peak_vram_gb} GB) [Baseline: 7.89 GB -> {vram_reduction_pct}% reduction]", flush=True)
    print(f"4. Est. Full E6-B 10-Epoch Runtime: {est_full_training_hours} hours [Baseline: ~64.0 hours]", flush=True)
    print(f"5. Implementation Errors:          {'None' if len(errors) == 0 else errors}", flush=True)
    print(f"6. Gradient Flow Verification:", flush=True)
    for comp, status in gradient_checks.items():
        print(f"   - {comp:<20}: {'PASS' if status else 'FAIL'}", flush=True)
    print(f"7. Numerical Behavior:", flush=True)
    print(f"   - Final Loss:                   {total_loss / max(1, total_samples):.4f}", flush=True)
    print(f"   - Accuracy:                     {total_correct / max(1, total_samples) * 100:.2f}%", flush=True)

    # Save Optimized Benchmark Results JSON
    out_json = PROJECT_ROOT / "experiments/e6_multiscale_inference/e6b_optimized_smoketest_results.json"
    results_data = {
        "benchmark_name": "E6-B Multi-View Training Optimized Benchmark",
        "checkpoint_init": str(checkpoint_path),
        "device": str(DEVICE),
        "n_smoke_images": total_samples,
        "batch_size": batch_size,
        "amp_enabled": True,
        "smoke_test_runtime_sec": round(t_total_sec, 2),
        "smoke_test_runtime_min": t_total_min,
        "images_per_sec": images_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "peak_vram_gb": peak_vram_gb,
        "baseline_runtime_sec": baseline_runtime_sec,
        "baseline_throughput": baseline_throughput,
        "baseline_vram_gb": baseline_vram_gb,
        "speedup_factor": speedup_factor,
        "vram_reduction_pct": vram_reduction_pct,
        "full_dataset_size": full_dataset_size,
        "est_full_10epoch_runtime_hours": est_full_training_hours,
        "implementation_errors": errors,
        "gradient_checks": gradient_checks,
        "numerical_validity": {
            "final_loss": round(total_loss / max(1, total_samples), 4),
            "accuracy_pct": round(total_correct / max(1, total_samples) * 100, 2),
            "gradients_valid": all(gradient_checks.values())
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[+] Saved optimized benchmark JSON to: {out_json}", flush=True)

if __name__ == "__main__":
    main()
