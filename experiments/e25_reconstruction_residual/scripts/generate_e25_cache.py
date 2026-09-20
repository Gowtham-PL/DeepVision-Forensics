import os
import sys
import time
import csv
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from diffusers import AutoencoderKL

PROJECT_ROOT = Path("d:/DeepVision-Forensics")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e25_reconstruction_residual.scripts.e25_model import create_radial_frequency_mask

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

def process_and_cache_manifest(
    manifest_path: Path,
    cache_dir: Path,
    vae: AutoencoderKL,
    mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 2, # 2 images = 10 views per VAE forward
    limit: int = None
) -> Dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    if limit is not None:
        records = records[:limit]
        
    total_records = len(records)
    print(f"[*] Processing {total_records} images from {manifest_path.name} -> {cache_dir}")
    
    to_tensor = transforms.ToTensor()
    t0 = time.time()
    processed_count = 0
    skipped_count = 0
    
    # Process in batches
    for b_start in range(0, total_records, batch_size):
        b_records = records[b_start : b_start + batch_size]
        batch_crops = []
        valid_records = []
        
        for r in b_records:
            out_file = cache_dir / f"{r['image_id']}.pt"
            if out_file.exists():
                skipped_count += 1
                continue
                
            img_path = PROJECT_ROOT / r["full_filepath"]
            if not img_path.exists():
                print(f"[!] Warning: missing image {img_path}")
                continue
                
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                crops = generate_5_crops(img_rgb)
                crop_tensors = torch.stack([to_tensor(c) for c in crops], dim=0) # (5, 3, 224, 224)
                batch_crops.append(crop_tensors)
                valid_records.append(r)
            except Exception as exc:
                print(f"[!] Error loading {img_path}: {exc}")
                
        if not batch_crops:
            continue
            
        # (B, 5, 3, 224, 224) -> flatten to (B * 5, 3, 224, 224)
        B_curr = len(batch_crops)
        x_views = torch.stack(batch_crops, dim=0).to(device=device, dtype=torch.float16) # (B, 5, 3, 224, 224)
        x_flat = x_views.view(B_curr * 5, 3, 224, 224)
        x_vae = x_flat * 2.0 - 1.0 # to [-1, 1]
        
        with torch.no_grad():
            posterior = vae.encode(x_vae).latent_dist
            latent = posterior.mean
            recon_vae = vae.decode(latent).sample
            recon_01 = (recon_vae / 2.0 + 0.5).clamp(0.0, 1.0)
            
            # RGB residual
            residual_rgb = torch.abs(x_flat - recon_01) # (B*5, 3, 224, 224)
            
            # Grayscale residual
            residual_gray = (
                0.2989 * residual_rgb[:, 0:1, :, :] +
                0.5870 * residual_rgb[:, 1:2, :, :] +
                0.1140 * residual_rgb[:, 2:3, :, :]
            ) # (B*5, 1, 224, 224)
            
            # Frequency representation
            fft_res = torch.fft.fft2(residual_gray.to(torch.float32), norm="ortho")
            fft_shift = torch.fft.fftshift(fft_res, dim=(-2, -1))
            fft_mag = torch.log1p(torch.abs(fft_shift)) # float32
            
            # Radial mid-frequency masked magnitude
            mid_freq_mag = fft_mag * mask
            
            # Instance standardization
            mid_freq_norm = (mid_freq_mag - mid_freq_mag.mean(dim=(-2, -1), keepdim=True)) / (
                mid_freq_mag.std(dim=(-2, -1), keepdim=True) + 1e-6
            )
            mid_freq_norm_fp16 = mid_freq_norm.to(torch.float16)
            
            # 5-channel residual input: [RGB (3), Grayscale (1), MidFreq (1)]
            res_5ch = torch.cat([residual_rgb, residual_gray, mid_freq_norm_fp16], dim=1) # (B*5, 5, 224, 224)
            
            # Scalar diagnostics
            res_mae = residual_rgb.mean(dim=(1, 2, 3)).cpu()
            res_std = residual_rgb.std(dim=(1, 2, 3)).cpu()
            tot_energy = torch.sum(residual_rgb**2, dim=(1, 2, 3)).cpu()
            mid_energy = torch.sum(mid_freq_mag**2, dim=(1, 2, 3)).cpu()
            
        # Reshape to (B, 5, 5, 224, 224)
        res_5ch = res_5ch.view(B_curr, 5, 5, 224, 224).cpu()
        res_mae = res_mae.view(B_curr, 5)
        res_std = res_std.view(B_curr, 5)
        tot_energy = tot_energy.view(B_curr, 5)
        mid_energy = mid_energy.view(B_curr, 5)
        
        # Save each sample
        for i, r in enumerate(valid_records):
            out_file = cache_dir / f"{r['image_id']}.pt"
            sample_data = {
                "residual_5views": res_5ch[i], # (5, 5, 224, 224) torch.float16
                "diagnostics": {
                    "residual_mae": res_mae[i].tolist(),
                    "residual_std": res_std[i].tolist(),
                    "total_energy": tot_energy[i].tolist(),
                    "mid_freq_energy": mid_energy[i].tolist()
                },
                "metadata": {
                    "image_id": r["image_id"],
                    "label": r["label"],
                    "source_manifest": str(manifest_path),
                    "full_filepath": r["full_filepath"],
                    "vae_model_id": "stabilityai/sd-vae-ft-mse",
                    "dtype": "torch.float16",
                    "shape": [5, 5, 224, 224]
                }
            }
            torch.save(sample_data, out_file)
            processed_count += 1
            
        if (processed_count + skipped_count) % 200 == 0 or (b_start + batch_size >= total_records):
            elapsed = time.time() - t0
            done = processed_count + skipped_count
            rate = done / max(1e-5, elapsed)
            rem = (total_records - done) / max(1e-5, rate)
            vram = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
            print(f"  [{done}/{total_records}] ({rate:.2f} img/s) Elapsed: {elapsed/60:.1f}m, ETA: {rem/60:.1f}m, VRAM: {vram:.1f} MB", flush=True)
            
    total_time = time.time() - t0
    return {
        "manifest": str(manifest_path),
        "total_records": total_records,
        "processed": processed_count,
        "skipped": skipped_count,
        "total_time_s": total_time,
        "images_per_sec": (processed_count + skipped_count) / max(1e-5, total_time),
        "peak_vram_mb": torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="both", choices=["train", "dev", "both"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for cache generation: {device}")
    
    print("[*] Loading VAE in FP16...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device=device, dtype=torch.float16)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
        
    mask = create_radial_frequency_mask(224, 224, low_cutoff=0.10, high_cutoff=0.50).to(device=device, dtype=torch.float32)
    
    base_cache_dir = PROJECT_ROOT / "experiments/e25_reconstruction_residual/cache"
    manifest_dir = PROJECT_ROOT / "experiments/e24_joint_multidomain/manifests"
    
    results = {}
    
    # 1. Dev split
    if args.split in ["dev", "both"]:
        dev_manifest = manifest_dir / "e24_dev_manifest.csv"
        dev_cache = base_cache_dir / "dev"
        res_dev = process_and_cache_manifest(
            manifest_path=dev_manifest,
            cache_dir=dev_cache,
            vae=vae,
            mask=mask,
            device=device,
            batch_size=2,
            limit=args.limit
        )
        results["dev"] = res_dev
        
    # 2. Train split
    if args.split in ["train", "both"]:
        train_manifest = manifest_dir / "e24_train_manifest.csv"
        train_cache = base_cache_dir / "train"
        res_train = process_and_cache_manifest(
            manifest_path=train_manifest,
            cache_dir=train_cache,
            vae=vae,
            mask=mask,
            device=device,
            batch_size=2,
            limit=args.limit
        )
        results["train"] = res_train
        
    # Save metadata summary
    summary_path = base_cache_dir / "cache_generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n=== CACHE GENERATION COMPLETED ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
