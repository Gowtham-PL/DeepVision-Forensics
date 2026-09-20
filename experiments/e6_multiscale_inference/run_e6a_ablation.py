import os
import sys
import time
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path("d:/DeepVision-Forensics")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion import build_model
from ml.train import compute_roc_auc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_pr_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]
    cum_tp = np.cumsum(y_true_sorted)
    cum_fp = np.cumsum(1 - y_true_sorted)
    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / max(np.sum(y_true), 1)
    precisions = np.r_[1, precisions]
    recalls = np.r_[0, recalls]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precisions, recalls))
    elif hasattr(np, "trapz"):
        return float(np.trapz(precisions, recalls))
    else:
        return float(np.sum((precisions[1:] + precisions[:-1]) * 0.5 * np.diff(recalls)))

def compute_full_metrics(y_true, y_prob, threshold=0.50):
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    n_real = int(np.sum(y_true == 0))
    n_ai = int(np.sum(y_true == 1))

    acc = float((tp + tn) / total) if total > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / n_real) if n_real > 0 else 0.0
    fnr = float(fn / n_ai) if n_ai > 0 else 0.0

    roc_auc = float(compute_roc_auc(y_true, y_prob))
    pr_auc = float(compute_pr_auc(y_true, y_prob))

    return {
        "n_samples": total,
        "n_real": n_real,
        "n_ai": n_ai,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }

def aspect_preserving_resize(img: Image.Image, target_size: int = 224) -> Image.Image:
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas

def generate_crops(img: Image.Image, num_crops: int) -> list[Image.Image]:
    w, h = img.size
    crops = []
    
    if num_crops == 2:
        if h > w:
            crops.append(img.crop((0, 0, w, w)))
            crops.append(img.crop((0, h - w, w, h)))
        else:
            crops.append(img.crop((0, 0, h, h)))
            crops.append(img.crop((w - h, 0, w, h)))
            
    elif num_crops == 4:
        w60, h60 = int(round(w * 0.6)), int(round(h * 0.6))
        crops.append(img.crop((0, 0, w60, h60)))                     # Top-Left
        crops.append(img.crop((w - w60, 0, w, h60)))                 # Top-Right
        crops.append(img.crop((0, h - h60, w60, h)))                 # Bottom-Left
        crops.append(img.crop((w - w60, h - h60, w, h)))             # Bottom-Right
        
    elif num_crops == 8:
        w60, h60 = int(round(w * 0.6)), int(round(h * 0.6))
        w20, h20 = int(round(w * 0.2)), int(round(h * 0.2))
        crops.append(img.crop((0, 0, w60, h60)))                     # 1. Top-Left
        crops.append(img.crop((w - w60, 0, w, h60)))                 # 2. Top-Right
        crops.append(img.crop((0, h - h60, w60, h)))                 # 3. Bottom-Left
        crops.append(img.crop((w - w60, h - h60, w, h)))             # 4. Bottom-Right
        crops.append(img.crop((w20, h20, w20 + w60, h20 + h60)))     # 5. Center
        crops.append(img.crop((w20, 0, w20 + w60, h60)))             # 6. Top-Center
        crops.append(img.crop((w20, h - h60, w20 + w60, h)))         # 7. Bottom-Center
        crops.append(img.crop((0, h20, w60, h20 + h60)))             # 8. Left-Center
        
    return crops

def load_manifest_validation():
    manifest_path = PROJECT_ROOT / "data/e5_external/manifests/e5_manifest.csv"
    val_records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                src_ds = row["source_dataset"]
                device = row["generator_or_device"]
                if "GenImage" in src_ds:
                    group = "GenImage"
                elif "Flux-1-Dev" in src_ds:
                    group = "FLUX dev"
                elif "Flux-1-Schnell" in src_ds:
                    group = "FLUX schnell"
                elif "Synthbuster" in src_ds:
                    group = "SDXL"
                elif "Apple" in device:
                    group = "VISION Apple"
                elif "OnePlus" in device or "Samsung" in device:
                    group = "VISION Android"
                else:
                    group = "Other"

                val_records.append({
                    "image_path": str(PROJECT_ROOT / row["image_path"]),
                    "label": int(row["label"]),
                    "group": group,
                    "generator_or_device": device,
                    "source_dataset": src_ds
                })
    return val_records

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
        group = rec["group"]

        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")

            view_tensors = []
            # Baseline
            img_base = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            view_tensors.append(self.to_tensor(img_base))

            # Padded
            img_pad = aspect_preserving_resize(img_rgb, 224)
            view_tensors.append(self.to_tensor(img_pad))

            # 2 crops
            for c in generate_crops(img_rgb, 2):
                view_tensors.append(self.to_tensor(c.resize((224, 224), Image.Resampling.BILINEAR)))

            # 4 crops
            for c in generate_crops(img_rgb, 4):
                view_tensors.append(self.to_tensor(c.resize((224, 224), Image.Resampling.BILINEAR)))

            # 8 crops
            for c in generate_crops(img_rgb, 8):
                view_tensors.append(self.to_tensor(c.resize((224, 224), Image.Resampling.BILINEAR)))

            stacked_views = torch.stack(view_tensors, dim=0) # (16, 3, 224, 224)
            return stacked_views, label, group, True
        except Exception as exc:
            dummy = torch.zeros((16, 3, 224, 224), dtype=torch.float32)
            return dummy, label, group, False

def main():
    print("=" * 70, flush=True)
    print("E6-A: MULTI-SCALE INFERENCE ABLATION EVALUATION (PARALLEL DATALOADER)", flush=True)
    print("=" * 70, flush=True)

    # Load E5 Model
    checkpoint_path = PROJECT_ROOT / "experiments/e5_generalization/best_model.pt"
    print(f"[*] Loading E5 model from: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = build_model(
        experiment="E3",
        pretrained=False,
        freq_norm_strategy="standardize",
        freq_embedding_dim=256,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    val_records = load_manifest_validation()
    print(f"[*] Loaded {len(val_records)} validation records from E5 manifest.", flush=True)

    dataset = MultiViewDataset(val_records)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)

    y_true_list = []
    groups_list = []

    probs_baseline = []
    probs_padded = []
    probs_g2_mean, probs_g2_max = [], []
    probs_g4_mean, probs_g4_max = [], []
    probs_g8_mean, probs_g8_max = [], []

    total_inference_time_ms = 0.0
    total_forward_passes = 0

    print("\n[*] Running batched evaluation loop with 4 parallel workers across 5,775 validation images...", flush=True)
    t_start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    with torch.no_grad():
        for idx, (batch_tensors, label, group, valid) in enumerate(loader):
            if not valid.item():
                continue

            # batch_tensors: shape (1, 16, 3, 224, 224)
            stacked_views = batch_tensors.squeeze(0).to(DEVICE) # (16, 3, 224, 224)
            lbl = label.item() if isinstance(label, torch.Tensor) else label
            grp = group[0] if isinstance(group, (list, tuple)) else group

            y_true_list.append(lbl)
            groups_list.append(grp)

            t0 = time.perf_counter()
            logits = model(stacked_views)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            total_forward_passes += 16
            t1 = time.perf_counter()

            total_inference_time_ms += (t1 - t0) * 1000

            p_base = float(probs[0])
            p_pad = float(probs[1])
            p_c2 = probs[2:4]
            p_c4 = probs[4:8]
            p_c8 = probs[8:16]

            # Global + 2 crops (3 views)
            g2_views = [p_base] + list(p_c2)
            p_g2_mean = float(np.mean(g2_views))
            p_g2_max = float(np.max(g2_views))

            # Global + 4 crops (5 views)
            g4_views = [p_base] + list(p_c4)
            p_g4_mean = float(np.mean(g4_views))
            p_g4_max = float(np.max(g4_views))

            # Global + 8 crops (9 views)
            g8_views = [p_base] + list(p_c8)
            p_g8_mean = float(np.mean(g8_views))
            p_g8_max = float(np.max(g8_views))

            probs_baseline.append(p_base)
            probs_padded.append(p_pad)
            probs_g2_mean.append(p_g2_mean)
            probs_g2_max.append(p_g2_max)
            probs_g4_mean.append(p_g4_mean)
            probs_g4_max.append(p_g4_max)
            probs_g8_mean.append(p_g8_mean)
            probs_g8_max.append(p_g8_max)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(val_records):
                elapsed = time.time() - t_start
                print(f"  Processed {idx + 1}/{len(val_records)} images ({elapsed:.1f}s elapsed)...", flush=True)

    t_total_sec = time.time() - t_start
    peak_vram_mb = round(torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

    print(f"\n[*] Completed evaluation loop in {t_total_sec:.2f}s!", flush=True)
    print(f"[*] Total forward passes: {total_forward_passes:,}", flush=True)
    print(f"[*] Peak GPU VRAM: {peak_vram_mb} MB", flush=True)

    # Compute Overall Metrics
    configs_to_eval = {
        "baseline_direct_224": probs_baseline,
        "aspect_preserving_padded_224": probs_padded,
        "global_plus_2crops_mean": probs_g2_mean,
        "global_plus_2crops_max": probs_g2_max,
        "global_plus_4crops_mean": probs_g4_mean,
        "global_plus_4crops_max": probs_g4_max,
        "global_plus_8crops_mean": probs_g8_mean,
        "global_plus_8crops_max": probs_g8_max,
    }

    overall_results = {}
    for cfg_name, prob_arr in configs_to_eval.items():
        overall_results[cfg_name] = compute_full_metrics(y_true_list, prob_arr, threshold=0.50)

    # Compute Group Breakdowns
    unique_groups = sorted(list(set(groups_list)))
    group_results = {}

    for grp in unique_groups:
        grp_indices = [i for i, g in enumerate(groups_list) if g == grp]
        grp_y_true = [y_true_list[i] for i in grp_indices]
        group_results[grp] = {}
        for cfg_name, prob_arr in configs_to_eval.items():
            grp_prob = [prob_arr[i] for i in grp_indices]
            group_results[grp][cfg_name] = compute_full_metrics(grp_y_true, grp_prob, threshold=0.50)

    # Evaluate Hard-Case Diagnostic Images
    hard_case_1 = PROJECT_ROOT / "data/e6_hard_cases/original_ai_edit.png"
    hard_case_2 = PROJECT_ROOT / "data/e6_hard_cases/whatsapp_download.jpeg"

    to_tensor = transforms.ToTensor()
    hard_case_evals = {}
    with torch.no_grad():
        for hc_path in [hard_case_1, hard_case_2]:
            hc_name = hc_path.name
            with Image.open(hc_path) as hc_img:
                hc_rgb = hc_img.convert("RGB")

            view_tensors = []
            view_tensors.append(to_tensor(hc_rgb.resize((224, 224), Image.Resampling.BILINEAR)))
            view_tensors.append(to_tensor(aspect_preserving_resize(hc_rgb, 224)))
            for c in generate_crops(hc_rgb, 2):
                view_tensors.append(to_tensor(c.resize((224, 224), Image.Resampling.BILINEAR)))
            for c in generate_crops(hc_rgb, 4):
                view_tensors.append(to_tensor(c.resize((224, 224), Image.Resampling.BILINEAR)))
            for c in generate_crops(hc_rgb, 8):
                view_tensors.append(to_tensor(c.resize((224, 224), Image.Resampling.BILINEAR)))

            batch_t = torch.stack(view_tensors, dim=0).to(DEVICE)
            probs = torch.sigmoid(model(batch_t)).squeeze(1).cpu().numpy()

            p_base = float(probs[0])
            p_pad = float(probs[1])
            p_c2 = list(probs[2:4])
            p_c4 = list(probs[4:8])
            p_c8 = list(probs[8:16])

            v2 = [p_base] + p_c2
            v4 = [p_base] + p_c4
            v8 = [p_base] + p_c8

            hard_case_evals[hc_name] = {
                "baseline_direct_224": round(p_base, 6),
                "aspect_preserving_padded_224": round(p_pad, 6),
                "global_plus_2crops_mean": round(float(np.mean(v2)), 6),
                "global_plus_2crops_max": round(float(np.max(v2)), 6),
                "global_plus_4crops_mean": round(float(np.mean(v4)), 6),
                "global_plus_4crops_max": round(float(np.max(v4)), 6),
                "global_plus_8crops_mean": round(float(np.mean(v8)), 6),
                "global_plus_8crops_max": round(float(np.max(v8)), 6),
            }

    # Compute Cost Metrics
    avg_infer_time_per_image_ms = round(total_inference_time_ms / len(val_records), 3)

    cost_metrics = {
        "n_validation_images": len(val_records),
        "total_forward_passes_all_configs": total_forward_passes,
        "total_evaluation_time_sec": round(t_total_sec, 2),
        "avg_total_inference_time_per_image_ms": avg_infer_time_per_image_ms,
        "peak_gpu_vram_mb": peak_vram_mb,
        "passes_per_image_breakdown": {
            "baseline_direct_224": 1,
            "aspect_preserving_padded_224": 1,
            "global_plus_2crops": 3,
            "global_plus_4crops": 5,
            "global_plus_8crops": 9
        },
        "relative_cost_vs_baseline": {
            "baseline_direct_224": 1.0,
            "aspect_preserving_padded_224": 1.0,
            "global_plus_2crops": 3.0,
            "global_plus_4crops": 5.0,
            "global_plus_8crops": 9.0
        }
    }

    # Print Summary Results Table
    print("\n" + "=" * 90, flush=True)
    print("E6-A ABLATION SUMMARY RESULTS TABLE (E5 Validation Set, N = 5,775)", flush=True)
    print("=" * 90, flush=True)
    headers = f"{'Configuration':<32} | {'AUC':<6} | {'PR-AUC':<6} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'FPR':<6} | {'FNR':<6}"
    print(headers, flush=True)
    print("-" * len(headers), flush=True)
    for cfg_name, res in overall_results.items():
        print(f"{cfg_name:<32} | {res['roc_auc']:<6.4f} | {res['pr_auc']:<6.4f} | {res['accuracy']:<6.4f} | {res['precision']:<6.4f} | {res['recall']:<6.4f} | {res['f1']:<6.4f} | {res['fpr']:<6.4f} | {res['fnr']:<6.4f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("HARD-CASE DIAGNOSTIC RESULTS", flush=True)
    print("=" * 90, flush=True)
    for hc_name, hc_res in hard_case_evals.items():
        print(f"\n--- {hc_name} ---", flush=True)
        for k, v in hc_res.items():
            print(f"  {k:<32}: {v:.6f} ({v*100:.2f}%)", flush=True)

    # Save Output JSON
    output_dir = PROJECT_ROOT / "experiments/e6_multiscale_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e6a_multiscale_results.json"

    full_results_data = {
        "metadata": {
            "ablation_name": "E6-A Multi-Scale Inference Ablation",
            "model_key": "e5_generalization",
            "model_name": "DeepVision-E5-Generalization",
            "checkpoint": "experiments/e5_generalization/best_model.pt",
            "manifest": "data/e5_external/manifests/e5_manifest.csv",
            "split": "val",
            "n_samples": len(val_records),
            "threshold": 0.50,
            "device": str(DEVICE),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "overall_results": overall_results,
        "validation_source_breakdown": group_results,
        "cost_metrics": cost_metrics,
        "hard_case_diagnostic": hard_case_evals
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results_data, f, indent=2)

    print(f"\n[+] Saved complete ablation results JSON to: {json_path}", flush=True)

if __name__ == "__main__":
    main()
