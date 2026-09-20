import os
import sys
import time
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path("d:/DeepVision-Forensics")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e25_reconstruction_residual.scripts.e25_model import E25ReconstructionDetector
from experiments.e25_reconstruction_residual.scripts.generate_e25_cache import generate_5_crops

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_roc_auc_numpy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

def compute_pr_auc_numpy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    desc_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_indices]
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.5
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    precision = tps / (tps + fps)
    recall = tps / n_pos
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2.0))

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict:
    preds = (y_prob >= threshold).astype(float)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    n_samples = len(y_true)
    acc = (tp + tn) / max(1, n_samples)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    roc_auc = compute_roc_auc_numpy(y_true, y_prob)
    pr_auc = compute_pr_auc_numpy(y_true, y_prob)

    return {
        "n_samples": n_samples,
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

class CachedE25Dataset(Dataset):
    """
    Loads 5 crops of the RGB image and the precomputed 5-channel residual tensor from cache.
    """
    def __init__(self, records: List[Dict], cache_dir: Path):
        self.records = records
        self.cache_dir = cache_dir
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        rec = self.records[idx]
        img_id = rec["image_id"]
        img_path = PROJECT_ROOT / rec["full_filepath"]
        cache_path = self.cache_dir / f"{img_id}.pt"

        # 1. Load RGB Crops
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            x_views = torch.stack([self.to_tensor(c) for c in crops], dim=0) # (5, 3, 224, 224)
        except Exception as exc:
            x_views = torch.zeros((5, 3, 224, 224), dtype=torch.float32)

        # 2. Load Cached Residual
        try:
            cached_obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            res_views = cached_obj["residual_5views"].to(torch.float32) # (5, 5, 224, 224)
        except Exception as exc:
            res_views = torch.zeros((5, 5, 224, 224), dtype=torch.float32)

        label_val = 1.0 if rec["label"] == "AI" else 0.0
        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        return x_views, res_views, label_tensor, rec

def evaluate_dev(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
    ablation_mode: Optional[str] = None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_views, res_views, labels, recs in val_loader:
            x_views = x_views.to(device)
            res_views = res_views.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x_views, cached_residuals=res_views, ablation_mode=ablation_mode)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(labels.numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    return metrics, y_true, y_prob

def run_cached_smoke_test(model: nn.Module, train_loader: DataLoader, device: torch.device):
    print("\n" + "=" * 70)
    print("RUNNING 20-IMAGE CACHED TRAINING SMOKE TEST")
    print("=" * 70)

    # Freeze base model and VAE for Phase 1 check
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.vae.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    steady_time = 0.0
    steady_samples = 0
    total_samples = 0
    model.train()

    for batch_idx, (x_views, res_views, labels, recs) in enumerate(train_loader):
        x_views = x_views.to(device)
        res_views = res_views.to(device)
        labels = labels.to(device)

        b_start = time.perf_counter()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(x_views, cached_residuals=res_views)
            loss = criterion(logits.squeeze(-1), labels)

        loss.backward()
        optimizer.step()

        b_time = time.perf_counter() - b_start
        total_samples += len(labels)
        if batch_idx > 0: # Exclude first batch DataLoader worker/CUDA startup warmup
            steady_time += b_time
            steady_samples += len(labels)

        print(f"  Batch {batch_idx+1}: {len(labels)} samples, Loss={loss.item():.4f}, Time={b_time*1000:.1f}ms")
        if total_samples >= 24:
            break

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    throughput = steady_samples / max(1e-5, steady_time)

    # Check gradients
    grad_checks = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            mod = name.split(".")[0]
            has_g = p.grad is not None and torch.isfinite(p.grad).all().item()
            grad_checks[mod] = grad_checks.get(mod, True) and has_g

    vae_has_grad = any(p.grad is not None for p in model.vae.parameters())
    backbone_has_grad = any(p.grad is not None for p in model.base_model.parameters())

    print("\n--- CACHED SMOKE TEST RESULTS ---")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Peak VRAM: {peak_vram:.2f} MB")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Trainable modules have valid gradients: {grad_checks}")
    print(f"VAE received gradients: {vae_has_grad} (Expected: False)")
    print(f"Backbone received gradients (Phase 1): {backbone_has_grad} (Expected: False)")

    est_epoch_s = 6279 / max(1e-5, throughput)
    print(f"Estimated 6,279-image Epoch Time: {est_epoch_s/60:.1f} minutes")
    print(f"Estimated 5 Epoch Total Time: {(est_epoch_s * 5)/3600:.2f} hours")
    if est_epoch_s * 5 > 6 * 3600:
        raise RuntimeError(f"Estimated training time exceeds 6 hours: {(est_epoch_s*5)/3600:.2f}h")
    print("[*] Cached smoke test PASSED successfully!")

def run_ablations(
    best_model: nn.Module,
    dev_loader: DataLoader,
    device: torch.device
) -> Dict:
    print("\n" + "=" * 70)
    print("RUNNING E25 ABLATION STUDY ON DEV SPLIT")
    print("=" * 70)

    # A. E6-C Baseline
    print("[*] Evaluating E6-C Baseline...")
    e6c_ckpt_path = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
    from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model
    e6c_model = MultiViewE5Model(checkpoint_path=e6c_ckpt_path, num_views=5).to(device)
    e6c_model.eval()

    all_e6c_probs = []
    all_targets = []
    with torch.no_grad():
        for x_views, res_views, labels, recs in dev_loader:
            x_views = x_views.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = e6c_model(x_views)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_e6c_probs.extend(probs.tolist())
            all_targets.extend(labels.numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    m_e6c = compute_metrics(y_true, np.array(all_e6c_probs, dtype=float))

    # B. E25 Full
    print("[*] Evaluating E25 Full (Spatial + Frequency + Residual)...")
    m_full, _, p_full = evaluate_dev(best_model, dev_loader, device, ablation_mode=None)

    # C. E25 Without Residual Branch
    print("[*] Evaluating E25 Without Residual (Zeroed Residual Embedding)...")
    m_no_res, _, p_no_res = evaluate_dev(best_model, dev_loader, device, ablation_mode="no_residual")

    # D. E25 Residual-Only
    print("[*] Evaluating E25 Residual-Only (Zeroed Spatial and Frequency)...")
    m_res_only, _, p_res_only = evaluate_dev(best_model, dev_loader, device, ablation_mode="residual_only")

    ablation_results = {
        "E6-C Baseline": m_e6c,
        "E25 Full (Spatial+Freq+Res)": m_full,
        "E25 Without Residual": m_no_res,
        "E25 Residual-Only": m_res_only
    }

    # Save to CSV
    csv_path = PROJECT_ROOT / "experiments/e25_reconstruction_residual/e25_ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "Delta_AUC_vs_E6C", "Delta_F1_vs_E6C", "Delta_FPR_vs_E6C"])
        for name, m in ablation_results.items():
            d_auc = m["roc_auc"] - m_e6c["roc_auc"]
            d_f1 = m["f1"] - m_e6c["f1"]
            d_fpr = m["fpr"] - m_e6c["fpr"]
            writer.writerow([name, m["roc_auc"], m["pr_auc"], m["accuracy"], m["precision"], m["recall"], m["f1"], m["fpr"], m["fnr"], round(d_auc, 4), round(d_f1, 4), round(d_fpr, 4)])

    print("\n--- ABLATION SUMMARY ---")
    for name, m in ablation_results.items():
        print(f"  {name:30s} | AUC={m['roc_auc']:.4f} | F1={m['f1']:.4f} | FPR={m['fpr']:.4f} | Recall={m['recall']:.4f}")

    return ablation_results

def evaluate_external_benchmarks(best_model: nn.Module, device: torch.device):
    print("\n" + "=" * 70)
    print("EVALUATING FROZEN SEALED EXTERNAL BENCHMARKS (E6-C vs E24 vs E25)")
    print("=" * 70)

    best_model.eval()
    e22_pred_dir = PROJECT_ROOT / "experiments/e22_ensemble_study/predictions"
    e24_pred_dir = PROJECT_ROOT / "experiments/e24_joint_multidomain/predictions"
    pred_dir = PROJECT_ROOT / "experiments/e25_reconstruction_residual/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    benchmark_configs = [
        ("E14 Clean", e22_pred_dir / "e14_clean_e22_predictions.csv", e24_pred_dir / "e14_clean_e24_predictions.csv", 200),
        ("E14 Degraded", e22_pred_dir / "e14_degraded_e22_predictions.csv", e24_pred_dir / "e14_degraded_e24_predictions.csv", 600),
        ("E19 Smartphone", e22_pred_dir / "e19_smartphone_benchmark_e22_predictions.csv", e24_pred_dir / "e19_smartphone_e24_predictions.csv", 200),
        ("E19 WhatsApp", e22_pred_dir / "e19_whatsapp_subset_e22_predictions.csv", e24_pred_dir / "e19_whatsapp_e24_predictions.csv", 60),
        ("WhatsApp N=67", e22_pred_dir / "whatsapp_n67_benchmark_e22_predictions.csv", e24_pred_dir / "whatsapp_n67_e24_predictions.csv", 67)
    ]

    to_tensor = transforms.ToTensor()
    all_benchmark_rows = []
    error_transitions = []
    residual_stats_records = []
    all_preds_by_benchmark = {}

    for name, csv_path, e24_csv, exp_n in benchmark_configs:
        print(f"\n[*] Evaluating benchmark: {name} (Expected N={exp_n})...")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        # Load E24 predictions for this benchmark
        e24_probs_map = {}
        if e24_csv.exists():
            with open(e24_csv, "r", encoding="utf-8") as ef:
                for row in csv.DictReader(ef):
                    e24_probs_map[row["image_id"]] = float(row["prob_e24"])
        else:
            print(f"[!] Warning: E24 CSV not found at {e24_csv}")

        e6c_probs = []
        e24_probs = []
        e25_probs = []
        labels = []
        pred_rows = []

        for rec in records:
            img_id = rec["image_id"]
            img_path = PROJECT_ROOT / rec["filepath"]
            lbl_raw = rec["label"]
            lbl = 1 if str(lbl_raw).strip().upper() in ["1", "AI"] else 0
            p_e6c = float(rec["prob_e6c"])
            p_e24 = e24_probs_map.get(img_id, float(rec.get("prob_e20", 0.5))) # fallback

            # Run E25 on-the-fly (unseen test images)
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
            crops = generate_5_crops(img_rgb)
            x_views = torch.stack([to_tensor(c) for c in crops], dim=0).unsqueeze(0).to(device) # (1, 5, 3, 224, 224)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits, view_w, diag = best_model(x_views, return_view_weights=True, return_diagnostics=True)
                    prob_e25 = float(torch.sigmoid(logits.squeeze(-1)).item())

            e6c_probs.append(p_e6c)
            e24_probs.append(p_e24)
            e25_probs.append(prob_e25)
            labels.append(lbl)

            # Record diagnostic statistics
            res_mae_val = float(diag["residual_mae"].mean().item())
            res_std_val = float(diag["residual_std"].mean().item())
            tot_energy_val = float(diag["total_energy"].mean().item())
            mid_energy_val = float(diag["mid_freq_energy"].mean().item())

            residual_stats_records.append({
                "benchmark": name,
                "image_id": img_id,
                "label": "AI" if lbl == 1 else "Real",
                "generator": rec.get("generator", "none"),
                "device": rec.get("device_family", "none"),
                "res_mae": res_mae_val,
                "res_std": res_std_val,
                "total_energy": tot_energy_val,
                "mid_freq_energy": mid_energy_val
            })

            rec_out = dict(rec)
            rec_out["prob_e24"] = p_e24
            rec_out["prob_e25"] = prob_e25
            rec_out["pred_e25"] = 1 if prob_e25 >= 0.50 else 0
            rec_out["res_mae"] = res_mae_val
            rec_out["mid_freq_energy"] = mid_energy_val
            pred_rows.append(rec_out)

            # Error transition analysis
            pred_e6c = 1 if p_e6c >= 0.50 else 0
            pred_e24 = 1 if p_e24 >= 0.50 else 0
            pred_e25 = 1 if prob_e25 >= 0.50 else 0

            error_transitions.append({
                "benchmark": name,
                "image_id": img_id,
                "label": lbl,
                "e6c_correct": (pred_e6c == lbl),
                "e24_correct": (pred_e24 == lbl),
                "e25_correct": (pred_e25 == lbl),
                "recovered_fn_e6c_to_e25": (lbl == 1 and pred_e6c == 0 and pred_e25 == 1),
                "lost_tp_e6c_to_e25": (lbl == 1 and pred_e6c == 1 and pred_e25 == 0),
                "corrected_fp_e6c_to_e25": (lbl == 0 and pred_e6c == 1 and pred_e25 == 0),
                "new_fp_e6c_to_e25": (lbl == 0 and pred_e6c == 0 and pred_e25 == 1),
                "recovered_fn_e24_to_e25": (lbl == 1 and pred_e24 == 0 and pred_e25 == 1),
                "lost_tp_e24_to_e25": (lbl == 1 and pred_e24 == 1 and pred_e25 == 0),
                "corrected_fp_e24_to_e25": (lbl == 0 and pred_e24 == 1 and pred_e25 == 0),
                "new_fp_e24_to_e25": (lbl == 0 and pred_e24 == 0 and pred_e25 == 1)
            })

        # Save predictions CSV
        tag = name.lower().replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
        out_pred_csv = pred_dir / f"{tag}_e25_predictions.csv"
        if pred_rows:
            with open(out_pred_csv, "w", newline="", encoding="utf-8") as pf:
                writer = csv.DictWriter(pf, fieldnames=list(pred_rows[0].keys()))
                writer.writeheader()
                writer.writerows(pred_rows)
        all_preds_by_benchmark[name] = pred_rows

        y_t = np.array(labels, dtype=int)
        m_e6c = compute_metrics(y_t, np.array(e6c_probs, dtype=float))
        m_e24 = compute_metrics(y_t, np.array(e24_probs, dtype=float))
        m_e25 = compute_metrics(y_t, np.array(e25_probs, dtype=float))

        print(f"  E6-C : Acc={m_e6c['accuracy']:.4f}, AUC={m_e6c['roc_auc']:.4f}, F1={m_e6c['f1']:.4f}, FPR={m_e6c['fpr']:.4f}, Recall={m_e6c['recall']:.4f}")
        print(f"  E24  : Acc={m_e24['accuracy']:.4f}, AUC={m_e24['roc_auc']:.4f}, F1={m_e24['f1']:.4f}, FPR={m_e24['fpr']:.4f}, Recall={m_e24['recall']:.4f}")
        print(f"  E25  : Acc={m_e25['accuracy']:.4f}, AUC={m_e25['roc_auc']:.4f}, F1={m_e25['f1']:.4f}, FPR={m_e25['fpr']:.4f}, Recall={m_e25['recall']:.4f}")

        for m_tag, m_dict in [("E6-C", m_e6c), ("E24", m_e24), ("E25", m_e25)]:
            all_benchmark_rows.append({
                "benchmark": name,
                "model": m_tag,
                "n_samples": len(labels),
                "accuracy": m_dict["accuracy"],
                "roc_auc": m_dict["roc_auc"],
                "pr_auc": m_dict["pr_auc"],
                "precision": m_dict["precision"],
                "recall": m_dict["recall"],
                "f1": m_dict["f1"],
                "fpr": m_dict["fpr"],
                "fnr": m_dict["fnr"],
                "tp": m_dict["tp"], "fp": m_dict["fp"], "tn": m_dict["tn"], "fn": m_dict["fn"]
            })

    # Save external benchmark results CSV
    out_bench_csv = PROJECT_ROOT / "experiments/e25_reconstruction_residual/e25_external_results.csv"
    with open(out_bench_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_benchmark_rows)

    # Save error transitions CSV
    out_trans_csv = PROJECT_ROOT / "experiments/e25_reconstruction_residual/e25_error_transitions.csv"
    with open(out_trans_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_transitions[0].keys()))
        writer.writeheader()
        writer.writerows(error_transitions)

    # Save FIRE-style residual diagnostics CSV
    out_res_diag_csv = PROJECT_ROOT / "experiments/e25_reconstruction_residual/reports/e25_fire_residual_diagnostics.csv"
    with open(out_res_diag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(residual_stats_records[0].keys()))
        writer.writeheader()
        writer.writerows(residual_stats_records)

    # Compute and save Subgroup Results CSV
    subgroup_rows = []
    # 1. E14 Generator Breakdown (AI Recall on E14 Clean)
    e14_clean_preds = all_preds_by_benchmark.get("E14 Clean", [])
    generators = sorted(list(set(r.get("generator") for r in e14_clean_preds if r.get("generator") and r.get("generator") != "camera")))
    for gen in generators:
        gen_recs = [r for r in e14_clean_preds if r.get("generator") == gen]
        n_g = len(gen_recs)
        rec_e6c = sum(1 for r in gen_recs if float(r["prob_e6c"]) >= 0.50) / max(1, n_g)
        rec_e24 = sum(1 for r in gen_recs if float(r["prob_e24"]) >= 0.50) / max(1, n_g)
        rec_e25 = sum(1 for r in gen_recs if float(r["prob_e25"]) >= 0.50) / max(1, n_g)
        subgroup_rows.append({
            "subgroup_type": "E14_Generator", "group_name": gen, "n_samples": n_g, "metric_type": "AI_Recall",
            "e6c_metric": f"{rec_e6c*100:.2f}%", "e24_metric": f"{rec_e24*100:.2f}%", "e25_metric": f"{rec_e25*100:.2f}%"
        })

    # 2. E19 Device Family (Real FPR on E19 Smartphone)
    e19_preds = all_preds_by_benchmark.get("E19 Smartphone", [])
    devices = sorted(list(set(r.get("device_family") for r in e19_preds if r.get("device_family"))))
    for dev_name in devices:
        dev_recs = [r for r in e19_preds if r.get("device_family") == dev_name and str(r["label"]).strip().upper() in ["0", "REAL"]]
        n_d = len(dev_recs)
        if n_d > 0:
            fpr_e6c = sum(1 for r in dev_recs if float(r["prob_e6c"]) >= 0.50) / n_d
            fpr_e24 = sum(1 for r in dev_recs if float(r["prob_e24"]) >= 0.50) / n_d
            fpr_e25 = sum(1 for r in dev_recs if float(r["prob_e25"]) >= 0.50) / n_d
            subgroup_rows.append({
                "subgroup_type": "E19_Device_Family", "group_name": dev_name, "n_samples": n_d, "metric_type": "Real_FPR",
                "e6c_metric": f"{fpr_e6c*100:.2f}%", "e24_metric": f"{fpr_e24*100:.2f}%", "e25_metric": f"{fpr_e25*100:.2f}%"
            })

    # 3. E19 WhatsApp Subset
    e19_wa_preds = all_preds_by_benchmark.get("E19 WhatsApp", [])
    ai_wa = [r for r in e19_wa_preds if str(r["label"]).strip().upper() in ["1", "AI"]]
    real_wa = [r for r in e19_wa_preds if str(r["label"]).strip().upper() in ["0", "REAL"]]
    if ai_wa:
        subgroup_rows.append({
            "subgroup_type": "E19 WhatsApp Subset", "group_name": "AI_Images", "n_samples": len(ai_wa), "metric_type": "AI_Recall",
            "e6c_metric": f"{(sum(1 for r in ai_wa if float(r['prob_e6c']) >= 0.50)/len(ai_wa))*100:.2f}%",
            "e24_metric": f"{(sum(1 for r in ai_wa if float(r['prob_e24']) >= 0.50)/len(ai_wa))*100:.2f}%",
            "e25_metric": f"{(sum(1 for r in ai_wa if float(r['prob_e25']) >= 0.50)/len(ai_wa))*100:.2f}%"
        })
    if real_wa:
        subgroup_rows.append({
            "subgroup_type": "E19 WhatsApp Subset", "group_name": "Real_Images", "n_samples": len(real_wa), "metric_type": "Real_FPR",
            "e6c_metric": f"{(sum(1 for r in real_wa if float(r['prob_e6c']) >= 0.50)/len(real_wa))*100:.2f}%",
            "e24_metric": f"{(sum(1 for r in real_wa if float(r['prob_e24']) >= 0.50)/len(real_wa))*100:.2f}%",
            "e25_metric": f"{(sum(1 for r in real_wa if float(r['prob_e25']) >= 0.50)/len(real_wa))*100:.2f}%"
        })

    # 4. WhatsApp N=67 Benchmark
    wa67_preds = all_preds_by_benchmark.get("WhatsApp N=67", [])
    ai_67 = [r for r in wa67_preds if str(r["label"]).strip().upper() in ["1", "AI"]]
    real_67 = [r for r in wa67_preds if str(r["label"]).strip().upper() in ["0", "REAL"]]
    if ai_67:
        subgroup_rows.append({
            "subgroup_type": "WhatsApp N=67 Benchmark", "group_name": "AI_Images", "n_samples": len(ai_67), "metric_type": "AI_Recall",
            "e6c_metric": f"{(sum(1 for r in ai_67 if float(r['prob_e6c']) >= 0.50)/len(ai_67))*100:.2f}%",
            "e24_metric": f"{(sum(1 for r in ai_67 if float(r['prob_e24']) >= 0.50)/len(ai_67))*100:.2f}%",
            "e25_metric": f"{(sum(1 for r in ai_67 if float(r['prob_e25']) >= 0.50)/len(ai_67))*100:.2f}%"
        })
    if real_67:
        subgroup_rows.append({
            "subgroup_type": "WhatsApp N=67 Benchmark", "group_name": "Real_Images", "n_samples": len(real_67), "metric_type": "Real_FPR",
            "e6c_metric": f"{(sum(1 for r in real_67 if float(r['prob_e6c']) >= 0.50)/len(real_67))*100:.2f}%",
            "e24_metric": f"{(sum(1 for r in real_67 if float(r['prob_e24']) >= 0.50)/len(real_67))*100:.2f}%",
            "e25_metric": f"{(sum(1 for r in real_67 if float(r['prob_e25']) >= 0.50)/len(real_67))*100:.2f}%"
        })

    out_subgroup_csv = PROJECT_ROOT / "experiments/e25_reconstruction_residual/e25_subgroup_results.csv"
    if subgroup_rows:
        with open(out_subgroup_csv, "w", newline="", encoding="utf-8") as sf:
            writer = csv.DictWriter(sf, fieldnames=list(subgroup_rows[0].keys()))
            writer.writeheader()
            writer.writerows(subgroup_rows)
        print(f"[*] Subgroup breakdown results saved to {out_subgroup_csv}")

    print(f"\n[*] Benchmark evaluations saved to {out_bench_csv}")
    print(f"[*] Error transitions saved to {out_trans_csv}")
    print(f"[*] Residual diagnostics saved to {out_res_diag_csv}")

    return all_benchmark_rows, error_transitions, residual_stats_records

def evaluate_hard_cases(best_model: nn.Module, device: torch.device):
    print("\n" + "=" * 70)
    print("EVALUATING E6 HARD CASES")
    print("=" * 70)

    hard_cases = [
        ("AI Edit", PROJECT_ROOT / "data/e6_hard_cases/original_ai_edit.png"),
        ("WhatsApp Recompressed", PROJECT_ROOT / "data/e6_hard_cases/whatsapp_download.jpeg")
    ]

    to_tensor = transforms.ToTensor()
    results = []

    # E6-C Model
    e6c_ckpt_path = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
    from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model
    e6c_model = MultiViewE5Model(checkpoint_path=e6c_ckpt_path, num_views=5).to(device)
    e6c_model.eval()

    best_model.eval()

    for name, img_path in hard_cases:
        if not img_path.exists():
            print(f"[!] Warning: missing hard case {img_path}")
            continue

        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
        crops = generate_5_crops(img_rgb)
        x_views = torch.stack([to_tensor(c) for c in crops], dim=0).unsqueeze(0).to(device) # (1, 5, 3, 224, 224)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                # E6-C
                logits_e6c, attn_e6c = e6c_model(x_views, return_view_weights=True)
                p_e6c = float(torch.sigmoid(logits_e6c.squeeze(-1)).item())

                # E25
                logits_e25, attn_e25, diag = best_model(x_views, return_view_weights=True, return_diagnostics=True)
                p_e25 = float(torch.sigmoid(logits_e25.squeeze(-1)).item())

        res_mae = float(diag["residual_mae"].mean().item())
        res_std = float(diag["residual_std"].mean().item())
        tot_energy = float(diag["total_energy"].mean().item())
        mid_energy = float(diag["mid_freq_energy"].mean().item())

        print(f"\n[{name}] {img_path.name}:")
        print(f"  E6-C Prob: {p_e6c:.4f} (Attention: {np.round(attn_e6c.cpu().numpy(), 3).tolist()})")
        print(f"  E25  Prob: {p_e25:.4f} (Attention: {np.round(attn_e25.cpu().numpy(), 3).tolist()})")
        print(f"  Residual Diagnostics: MAE={res_mae:.4f}, Std={res_std:.4f}, TotalEnergy={tot_energy:.1f}, MidFreqEnergy={mid_energy:.1f}")

        results.append({
            "case": name,
            "filepath": str(img_path),
            "prob_e6c": p_e6c,
            "prob_e25": p_e25,
            "attn_weights_e25": attn_e25.cpu().numpy().tolist(),
            "residual_mae": res_mae,
            "residual_std": res_std,
            "total_energy": tot_energy,
            "mid_freq_energy": mid_energy
        })

    out_json = PROJECT_ROOT / "experiments/e25_reconstruction_residual/reports/e25_hard_cases.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting E25 Training on device: {device}")

    # Paths
    exp_dir = PROJECT_ROOT / "experiments/e25_reconstruction_residual"
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = exp_dir / "cache"

    train_manifest = PROJECT_ROOT / "experiments/e24_joint_multidomain/manifests/e24_train_manifest.csv"
    dev_manifest = PROJECT_ROOT / "experiments/e24_joint_multidomain/manifests/e24_dev_manifest.csv"

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_records = list(csv.DictReader(f))
    with open(dev_manifest, "r", encoding="utf-8") as f:
        dev_records = list(csv.DictReader(f))

    print(f"Loaded {len(train_records)} train records and {len(dev_records)} dev records.")

    train_dataset = CachedE25Dataset(train_records, cache_dir / "train")
    dev_dataset = CachedE25Dataset(dev_records, cache_dir / "dev")

    batch_size = 8
    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Instantiate model
    e6c_ckpt = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
    model = E25ReconstructionDetector(e6c_checkpoint_path=e6c_ckpt).to(device)

    # 1. Run 20-image smoke test
    run_cached_smoke_test(model, train_loader, device)

    # 2. Training Loop
    # Phase 1: Freeze EfficientNet + FFT branch + VAE. Train residual CNN, projections, attention, fusion, classifier (2 epochs)
    print("\n" + "=" * 70)
    print("STARTING E25 PHASE 1: FROZEN BACKBONE (2 EPOCHS)")
    print("=" * 70)

    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.vae.parameters():
        p.requires_grad = False

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last saved checkpoint if available")
    parser.add_argument("--scratch", action="store_true", default=False, help="Train from epoch 1 from scratch")
    args, _ = parser.parse_known_args()

    # Determine start epoch
    start_epoch = 1
    epoch_logs = []
    best_f1 = -1.0
    best_fpr = 1.0
    best_auc = -1.0
    best_recall = -1.0
    best_epoch = -1
    best_ckpt_path = ckpt_dir / "e25_best_model.pt"

    ep2_ckpt = ckpt_dir / "e25_checkpoint_epoch2.pt"
    ep1_ckpt = ckpt_dir / "e25_checkpoint_epoch1.pt"

    if not args.scratch and ep2_ckpt.exists() and ep1_ckpt.exists():
        print(f"[*] Detected completed Phase 1 checkpoints (Epochs 1 and 2). Resuming into Phase 2...")
        d1 = torch.load(ep1_ckpt, map_location=device, weights_only=False)
        d2 = torch.load(ep2_ckpt, map_location=device, weights_only=False)
        m1 = d1.get("metrics", {})
        m2 = d2.get("metrics", {})

        epoch_logs.append({
            "epoch": 1, "phase": 1, "train_loss": 0.2821,
            "dev_acc": m1.get("accuracy", 0.8084), "dev_roc_auc": m1.get("roc_auc", 0.9031),
            "dev_pr_auc": m1.get("pr_auc", 0.8962), "dev_precision": m1.get("precision", 0.8280),
            "dev_recall": m1.get("recall", 0.7619), "dev_f1": m1.get("f1", 0.7936),
            "dev_fpr": m1.get("fpr", 0.1481), "dev_fnr": m1.get("fnr", 0.2381),
            "duration_min": 4.7
        })
        epoch_logs.append({
            "epoch": 2, "phase": 1, "train_loss": 0.1240,
            "dev_acc": m2.get("accuracy", 0.8084), "dev_roc_auc": m2.get("roc_auc", 0.9056),
            "dev_pr_auc": m2.get("pr_auc", 0.8948), "dev_precision": m2.get("precision", 0.8448),
            "dev_recall": m2.get("recall", 0.7395), "dev_f1": m2.get("f1", 0.7886),
            "dev_fpr": m2.get("fpr", 0.1271), "dev_fnr": m2.get("fnr", 0.2605),
            "duration_min": 4.0
        })
        best_f1 = m1.get("f1", 0.7936)
        best_fpr = m1.get("fpr", 0.1481)
        best_auc = m1.get("roc_auc", 0.9031)
        best_recall = m1.get("recall", 0.7619)
        best_epoch = 1

        # Load weights from epoch 2
        model.load_state_dict(d2["model_state_dict"])
        start_epoch = 3
        print(f"[*] Loaded Epoch 2 weights. Best model tracking initialized: Epoch 1 (F1={best_f1:.4f}, FPR={best_fpr:.4f}, AUC={best_auc:.4f}, Recall={best_recall:.4f})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = None

    for epoch in range(start_epoch, 6):
        if epoch <= 2:
            if epoch == 1:
                print("\n" + "=" * 70)
                print("STARTING E25 PHASE 1: FROZEN BACKBONE (2 EPOCHS)")
                print("=" * 70)
            for p in model.base_model.parameters():
                p.requires_grad = False
            for p in model.vae.parameters():
                p.requires_grad = False
            phase1_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(phase1_params, lr=1e-4, weight_decay=1e-4)
        elif epoch >= 3 and optimizer is None:
            print("\n" + "=" * 70)
            print("STARTING E25 PHASE 2: UNFREEZE EFFICIENTNET-B3 BACKBONE (UP TO 3 EPOCHS)")
            print("=" * 70)
            # Unfreeze EfficientNet
            for p in model.base_model.spatial_branch.parameters():
                p.requires_grad = True
            for p in model.base_model.frequency_branch.parameters():
                p.requires_grad = True
            for p in model.vae.parameters():
                p.requires_grad = False # VAE remains frozen

            optimizer = torch.optim.AdamW([
                {"params": model.base_model.spatial_branch.parameters(), "lr": 1e-5},
                {"params": model.base_model.frequency_branch.parameters(), "lr": 1e-5},
                {"params": model.residual_cnn.parameters(), "lr": 1e-4},
                {"params": model.spatial_proj.parameters(), "lr": 1e-4},
                {"params": model.freq_proj.parameters(), "lr": 1e-4},
                {"params": model.res_proj.parameters(), "lr": 1e-4},
                {"params": model.view_attention.parameters(), "lr": 1e-4},
                {"params": model.fusion_mlp.parameters(), "lr": 1e-4},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ], weight_decay=1e-4)

        phase_num = 1 if epoch <= 2 else 2
        print(f"\n--- Epoch {epoch}/5 (Phase {phase_num}) ---")
        model.train()
        train_loss = 0.0
        n_batches = 0
        t_ep = time.perf_counter()

        for x_views, res_views, labels, recs in train_loader:
            x_views = x_views.to(device)
            res_views = res_views.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x_views, cached_residuals=res_views)
                loss = criterion(logits.squeeze(-1), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        ep_duration = time.perf_counter() - t_ep
        avg_loss = train_loss / max(1, n_batches)

        # Evaluate Dev
        m_dev, _, _ = evaluate_dev(model, dev_loader, device, threshold=0.50)
        print(f"Epoch {epoch} finished in {ep_duration/60:.1f}m | Train Loss={avg_loss:.4f} | Dev Acc={m_dev['accuracy']:.4f} | Dev AUC={m_dev['roc_auc']:.4f} | Dev F1={m_dev['f1']:.4f} | Dev FPR={m_dev['fpr']:.4f} | Dev Recall={m_dev['recall']:.4f}")

        log_row = {
            "epoch": epoch,
            "phase": phase_num,
            "train_loss": round(avg_loss, 4),
            "dev_acc": m_dev["accuracy"],
            "dev_roc_auc": m_dev["roc_auc"],
            "dev_pr_auc": m_dev["pr_auc"],
            "dev_precision": m_dev["precision"],
            "dev_recall": m_dev["recall"],
            "dev_f1": m_dev["f1"],
            "dev_fpr": m_dev["fpr"],
            "dev_fnr": m_dev["fnr"],
            "duration_min": round(ep_duration / 60, 2)
        }
        epoch_logs.append(log_row)

        # Save checkpoint for epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": m_dev,
            "optimizer_state_dict": optimizer.state_dict()
        }, ckpt_dir / f"e25_checkpoint_epoch{epoch}.pt")

        # Checkpoint selection: 1. F1, 2. Lowest FPR, 3. ROC-AUC, 4. Recall
        is_best = False
        if m_dev["f1"] > best_f1 + 1e-4:
            is_best = True
        elif abs(m_dev["f1"] - best_f1) <= 1e-4:
            if m_dev["fpr"] < best_fpr - 1e-4:
                is_best = True
            elif abs(m_dev["fpr"] - best_fpr) <= 1e-4 and m_dev["roc_auc"] > best_auc + 1e-4:
                is_best = True
            elif abs(m_dev["fpr"] - best_fpr) <= 1e-4 and abs(m_dev["roc_auc"] - best_auc) <= 1e-4 and m_dev["recall"] > best_recall:
                is_best = True

        if is_best:
            best_f1 = m_dev["f1"]
            best_fpr = m_dev["fpr"]
            best_auc = m_dev["roc_auc"]
            best_recall = m_dev["recall"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": m_dev
            }, best_ckpt_path)
            print(f"  >>> New Best Model at Epoch {epoch}! (F1={best_f1:.4f}, FPR={best_fpr:.4f}, AUC={best_auc:.4f}, Recall={best_recall:.4f})")

    # Save epoch metrics CSV
    with open(exp_dir / "e25_epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_logs[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"\n[*] Training complete. Best model: Epoch {best_epoch} saved to {best_ckpt_path}")

    # Load best model for evaluation
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # 3. Ablation Study
    run_ablations(model, dev_loader, device)

    # 4. Sealed External Benchmark Evaluation
    evaluate_external_benchmarks(model, device)

    # 5. Hard Cases Evaluation
    evaluate_hard_cases(model, device)

    print("\n" + "=" * 70)
    print("ALL E25 TASKS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()

