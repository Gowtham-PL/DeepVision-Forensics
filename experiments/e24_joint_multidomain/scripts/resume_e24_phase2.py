"""
E24 Phase 2 Resume Script.

Resumes E24 training from the existing best checkpoint (Phase 1 / Epoch 1),
runs Phase 2 (Epochs 3-5), then executes the threshold study and sealed
external benchmark evaluation.
"""

import os
import sys
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion import build_model, DeepVisionFusionModel
from experiments.e22_ensemble_study.scripts.run_e22_study import compute_metrics

EXP_DIR = PROJECT_ROOT / "experiments/e24_joint_multidomain"
MANIFEST_DIR = EXP_DIR / "manifests"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
SUBGROUP_DIR = EXP_DIR / "subgroups"
PRED_DIR = EXP_DIR / "predictions"

E22_PRED_DIR = PROJECT_ROOT / "experiments/e22_ensemble_study/predictions"
TRAIN_MANIFEST = MANIFEST_DIR / "e24_train_manifest.csv"
DEV_MANIFEST = MANIFEST_DIR / "e24_dev_manifest.csv"
BEST_MODEL_PATH = CKPT_DIR / "e24_best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_5_crops(img):
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    return [
        img.resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR),
    ]


class MultiDomainE24Dataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = PROJECT_ROOT / rec["full_filepath"]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                crops = generate_5_crops(img)
                x_views = torch.stack([self.to_tensor(c) for c in crops], dim=0)
        except Exception:
            x_views = torch.zeros((5, 3, 224, 224), dtype=torch.float32)

        y_det = torch.tensor([1.0 if rec["label"] == "AI" else 0.0], dtype=torch.float32)
        y_dom = torch.tensor([
            float(rec.get("is_legacy_ai", 0)),
            float(rec.get("is_modern_ai", 0)),
            float(rec.get("is_smartphone_real", 0)),
            float(rec.get("is_camera_real", 0)),
            float(rec.get("is_compressed", 0)),
        ], dtype=torch.float32)
        return x_views, y_det, y_dom, rec


class MultiDomainE24Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = build_model(
            experiment="E3", pretrained=False,
            freq_norm_strategy="standardize", freq_embedding_dim=256,
        )
        self.fused_dim = self.base_model.fused_dim  # 1792
        self.shared_forensic_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, 256)
        )
        self.local_attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        nn.init.zeros_(self.local_attention[2].weight)
        nn.init.zeros_(self.local_attention[2].bias)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, 256)
        )
        self.classifier = nn.Linear(256, 1)
        self.domain_head = nn.Linear(256, 5)

    def unfreeze_backbones(self):
        for p in self.base_model.spatial_branch.parameters():
            p.requires_grad = True
        for p in self.base_model.frequency_branch.parameters():
            p.requires_grad = True
        print("[*] Backbones UNFROZEN.")

    def forward(self, x_views):
        B, V, C, H, W = x_views.shape
        x_flat = x_views.view(B * V, C, H, W)
        feats = self.base_model(x_flat, return_features=True)
        e_fused = feats["fused_embedding"]
        e_freq = feats["frequency_embedding"]
        z_views = self.shared_forensic_mlp(e_fused).view(B, V, 256)
        z_global = z_views[:, 0, :]
        e_freq_global = e_freq.view(B, V, 256)[:, 0, :]
        z_local = z_views[:, 1:, :]
        attn_weights = torch.softmax(self.local_attention(z_local), dim=1)
        z_agg_local = torch.sum(attn_weights * z_local, dim=1)
        combined = torch.cat([z_global, z_agg_local, e_freq_global], dim=1)
        h_forensic = self.fusion_mlp(combined)
        return self.classifier(h_forensic), self.domain_head(h_forensic), h_forensic


@torch.no_grad()
def evaluate_dev(model, loader):
    model.eval()
    criterion_det = nn.BCEWithLogitsLoss()
    criterion_dom = nn.BCEWithLogitsLoss()
    total_loss, all_y_true, all_p_det, item_records = 0.0, [], [], []

    for x_views, y_det, y_dom, rec_list in loader:
        x_views = x_views.to(DEVICE, non_blocking=True)
        y_det_d = y_det.to(DEVICE, non_blocking=True)
        y_dom_d = y_dom.to(DEVICE, non_blocking=True)
        det_logits, dom_logits, _ = model(x_views)
        loss = criterion_det(det_logits, y_det_d) + 0.05 * criterion_dom(dom_logits, y_dom_d)
        total_loss += loss.item() * len(y_det)
        p_det = torch.sigmoid(det_logits).cpu().numpy().flatten()
        all_y_true.extend(y_det.numpy().flatten().tolist())
        all_p_det.extend(p_det.tolist())
        for i in range(len(p_det)):
            item = {k: rec_list[k][i] for k in rec_list.keys()}
            item["prob_det"] = p_det[i]
            item_records.append(item)

    y_arr = np.array(all_y_true)
    p_arr = np.array(all_p_det)
    m = compute_metrics(y_arr, p_arr, threshold=0.50)
    m["val_loss"] = total_loss / len(y_arr)
    return m, item_records


def main():
    for d in [EXP_DIR, MANIFEST_DIR, CKPT_DIR, REPORTS_DIR, SUBGROUP_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("E24 RESUME: Phase 2 Training + External Evaluation")
    print("=" * 50)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {gpu}")

    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_records = list(csv.DictReader(f))
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        dev_records = list(csv.DictReader(f))
    print(f"Train: {len(train_records)}, Dev: {len(dev_records)}")

    train_dataset = MultiDomainE24Dataset(train_records)
    dev_dataset = MultiDomainE24Dataset(dev_records)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nLoading Phase 1 best checkpoint: {BEST_MODEL_PATH}")
    model = MultiDomainE24Model()
    ckpt_p1 = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt_p1["model_state_dict"])
    saved_epoch = ckpt_p1["epoch"]
    saved_dm = ckpt_p1["dev_metrics"]
    best_dev_f1 = saved_dm["f1"]
    best_epoch = saved_epoch
    print(f"[*] Epoch {saved_epoch}: Dev F1={best_dev_f1:.4f}, Acc={saved_dm['accuracy']*100:.2f}%, AUC={saved_dm['roc_auc']:.4f}")
    model.to(DEVICE)

    criterion_det = nn.BCEWithLogitsLoss()
    criterion_dom = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None

    epoch_logs = []

    print("\n>>> STARTING PHASE 2: Backbones Unfrozen (Epochs 3-5) <<<")
    model.unfreeze_backbones()

    optimizer_p2 = torch.optim.AdamW([
        {"params": model.base_model.spatial_branch.parameters(), "lr": 1e-5},
        {"params": model.base_model.frequency_branch.parameters(), "lr": 1e-5},
        {"params": model.shared_forensic_mlp.parameters(), "lr": 1e-4},
        {"params": model.local_attention.parameters(), "lr": 1e-4},
        {"params": model.fusion_mlp.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4},
        {"params": model.domain_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    for epoch in range(3, 6):
        t0 = time.time()
        model.train()
        running_loss = running_det = running_dom = 0.0

        for x_views, y_det, y_dom, _ in train_loader:
            x_views = x_views.to(DEVICE, non_blocking=True)
            y_det = y_det.to(DEVICE, non_blocking=True)
            y_dom = y_dom.to(DEVICE, non_blocking=True)
            optimizer_p2.zero_grad()
            with autocast("cuda"):
                det_logits, dom_logits, _ = model(x_views)
                l_det = criterion_det(det_logits, y_det)
                l_dom = criterion_dom(dom_logits, y_dom)
                loss = l_det + 0.05 * l_dom
            scaler.scale(loss).backward()
            scaler.step(optimizer_p2)
            scaler.update()
            running_loss += loss.item() * len(y_det)
            running_det += l_det.item() * len(y_det)
            running_dom += l_dom.item() * len(y_det)

        n = len(train_records)
        train_loss = running_loss / n
        train_det = running_det / n
        train_dom = running_dom / n
        dev_m, _ = evaluate_dev(model, dev_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/5 (P2) [{elapsed:.1f}s] | TrLoss:{train_loss:.4f}(Det:{train_det:.4f} Dom:{train_dom:.4f}) | "
              f"DevAcc:{dev_m['accuracy']*100:.2f}% F1:{dev_m['f1']:.4f} AUC:{dev_m['roc_auc']:.4f} "
              f"AIRec:{dev_m['recall']*100:.2f}% FPR:{dev_m['fpr']*100:.2f}%")

        epoch_logs.append({
            "epoch": epoch, "phase": "Phase2",
            "train_loss": train_loss, "train_det_loss": train_det, "train_dom_loss": train_dom,
            "dev_loss": dev_m["val_loss"], "dev_accuracy": dev_m["accuracy"],
            "dev_f1": dev_m["f1"], "dev_roc_auc": dev_m["roc_auc"],
            "dev_pr_auc": dev_m["pr_auc"], "dev_precision": dev_m["precision"],
            "dev_recall": dev_m["recall"], "dev_fpr": dev_m["fpr"], "dev_fnr": dev_m["fnr"],
            "tp": dev_m["tp"], "tn": dev_m["tn"], "fp": dev_m["fp"], "fn": dev_m["fn"]
        })

        if dev_m["f1"] > best_dev_f1:
            best_dev_f1 = dev_m["f1"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "dev_metrics": dev_m}, BEST_MODEL_PATH)
            print(f"  [*] New Best: Epoch {epoch} F1={dev_m['f1']:.4f}")

    print(f"\nPhase 2 complete. Best epoch: {best_epoch}, Best Dev F1: {best_dev_f1:.4f}")

    epoch_csv_path = EXP_DIR / "e24_phase2_epoch_metrics.csv"
    with open(epoch_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_logs[0].keys()))
        writer.writeheader()
        for r in epoch_logs: writer.writerow(r)
    print(f"Saved Phase 2 metrics: {epoch_csv_path}")

    # Load best for evaluation
    print(f"\nLoading best (Epoch {best_epoch}, F1={best_dev_f1:.4f}) for eval...")
    best_ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Dev Threshold Study
    print("\n--- Dev Threshold Study (tau=0.20..0.80) ---")
    _, dev_items = evaluate_dev(model, dev_loader)
    y_dev_arr = np.array([1 if r["label"] == "AI" else 0 for r in dev_items])
    p_dev_arr = np.array([float(r["prob_det"]) for r in dev_items])

    threshold_rows, best_thresh_5pct_fpr, best_f1_5pct_fpr = [], 0.50, -1.0
    best_thresh_unconstrained, best_f1_unconstrained = 0.50, -1.0
    for tau in np.arange(0.20, 0.81, 0.05):
        tau = round(float(tau), 2)
        tm = compute_metrics(y_dev_arr, p_dev_arr, threshold=tau)
        threshold_rows.append({"threshold": tau, "accuracy": tm["accuracy"], "f1": tm["f1"],
            "precision": tm["precision"], "ai_recall": tm["recall"], "real_fpr": tm["fpr"],
            "ai_fnr": tm["fnr"], "tp": tm["tp"], "tn": tm["tn"], "fp": tm["fp"], "fn": tm["fn"]})
        if tm["f1"] > best_f1_unconstrained:
            best_f1_unconstrained = tm["f1"]; best_thresh_unconstrained = tau
        if tm["fpr"] <= 0.05 and tm["f1"] > best_f1_5pct_fpr:
            best_f1_5pct_fpr = tm["f1"]; best_thresh_5pct_fpr = tau

    dev_thresh_csv = EXP_DIR / "e24_dev_thresholds.csv"
    with open(dev_thresh_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        for r in threshold_rows: writer.writerow(r)
    print(f"Saved threshold study: {dev_thresh_csv}")
    print(f"Best unconstrained: tau={best_thresh_unconstrained} (F1={best_f1_unconstrained:.4f})")
    print(f"Best FPR<=5%: tau={best_thresh_5pct_fpr} (F1={best_f1_5pct_fpr:.4f})")

    # Sealed External Benchmarks
    print("\n" + "=" * 50)
    print("SEALED EXTERNAL BENCHMARK EVALUATION")
    print("=" * 50)

    benchmark_configs = [
        {"name": "E14 Clean External", "file": E22_PRED_DIR / "e14_clean_e22_predictions.csv", "key": "e14_clean", "n": 200},
        {"name": "E14 Degraded External", "file": E22_PRED_DIR / "e14_degraded_e22_predictions.csv", "key": "e14_degraded", "n": 600},
        {"name": "E19 Smartphone Benchmark", "file": E22_PRED_DIR / "e19_smartphone_benchmark_e22_predictions.csv", "key": "e19_smartphone", "n": 200},
        {"name": "E19 WhatsApp Subset", "file": E22_PRED_DIR / "e19_whatsapp_subset_e22_predictions.csv", "key": "e19_whatsapp", "n": 60},
        {"name": "WhatsApp N=67 Benchmark", "file": E22_PRED_DIR / "whatsapp_n67_benchmark_e22_predictions.csv", "key": "whatsapp_n67", "n": 67},
    ]

    all_ext_results, all_decisions, all_transitions = [], [], []
    to_tensor = transforms.ToTensor()

    for bcfg in benchmark_configs:
        bname = bcfg["name"]
        print(f"\nEvaluating {bname} (N={bcfg['n']})...")
        with open(bcfg["file"], "r", encoding="utf-8") as f:
            b_rows = list(csv.DictReader(f))
        assert len(b_rows) == bcfg["n"], f"Expected {bcfg['n']}, got {len(b_rows)}"

        p_e24_list = []
        for i, r in enumerate(b_rows):
            if i % 100 == 0: print(f"  {i}/{len(b_rows)}...")
            fp = PROJECT_ROOT / r["filepath"]
            try:
                with Image.open(fp) as img:
                    img = img.convert("RGB")
                    crops = generate_5_crops(img)
                    x_v = torch.stack([to_tensor(c) for c in crops], dim=0).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        with autocast("cuda"):
                            det_logit, _, _ = model(x_v)
                        p_val = torch.sigmoid(det_logit).item()
            except Exception as e:
                print(f"  Warning: {fp}: {e}")
                p_val = 0.5
            p_e24_list.append(p_val)

        p_e24 = np.array(p_e24_list)
        y_true = np.array([1 if r["label"] == "AI" else 0 for r in b_rows])
        p_e6c = np.array([float(r["prob_e6c"]) for r in b_rows])
        p_e20 = np.array([float(r["prob_e20"]) for r in b_rows])
        p_e21 = np.array([float(r["prob_e21"]) for r in b_rows])

        # Save E24 predictions
        pred_csv = PRED_DIR / f"{bcfg['key']}_e24_predictions.csv"
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "filepath", "label", "prob_e24"])
            for i, r in enumerate(b_rows):
                writer.writerow([r.get("image_id", ""), r.get("filepath", ""), r["label"], f"{p_e24_list[i]:.6f}"])
        print(f"  Saved predictions: {pred_csv}")

        models_to_eval = [
            ("E6-C Baseline (tau=0.50)", p_e6c, 0.50),
            ("E20 Baseline (tau=0.50)", p_e20, 0.50),
            ("E21 Baseline (tau=0.50)", p_e21, 0.50),
            ("E24 Joint Detector (tau=0.50)", p_e24, 0.50),
            (f"E24 Joint Detector (Research tau={best_thresh_5pct_fpr})", p_e24, best_thresh_5pct_fpr),
        ]

        for mname, p_arr, t_val in models_to_eval:
            m = compute_metrics(y_true, p_arr, threshold=t_val)
            all_ext_results.append({
                "benchmark": bname, "n_samples": len(y_true), "model": mname, "threshold": t_val,
                "accuracy": m["accuracy"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                "precision": m["precision"], "ai_recall": m["recall"], "real_fpr": m["fpr"],
                "ai_fnr": m["fnr"], "f1": m["f1"], "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            })
            print(f"  {mname}: Acc={m['accuracy']*100:.2f}% AUC={m['roc_auc']:.4f} AIRec={m['recall']*100:.2f}% FPR={m['fpr']*100:.2f}%")

        for idx, row in enumerate(b_rows):
            y_i = y_true[idx]
            p20_i, p21_i, p24_i = p_e20[idx], p_e21[idx], p_e24[idx]
            pred_20 = 1 if p20_i >= 0.50 else 0
            pred_21 = 1 if p21_i >= 0.50 else 0
            pred_24 = 1 if p24_i >= 0.50 else 0
            corr_20, corr_21, corr_24 = (pred_20 == y_i), (pred_21 == y_i), (pred_24 == y_i)

            t20_24 = ("E20_correct__E24_correct" if corr_20 and corr_24 else
                      "E20_correct__E24_wrong" if corr_20 and not corr_24 else
                      "E20_wrong__E24_correct" if not corr_20 and corr_24 else "E20_wrong__E24_wrong")
            t21_24 = ("E21_correct__E24_correct" if corr_21 and corr_24 else
                      "E21_correct__E24_wrong" if corr_21 and not corr_24 else
                      "E21_wrong__E24_correct" if not corr_21 and corr_24 else "E21_wrong__E24_wrong")

            all_decisions.append({
                "benchmark": bname, "image_id": row.get("image_id", ""), "label": row["label"],
                "prob_e6c": row["prob_e6c"], "prob_e20": f"{p20_i:.6f}", "pred_e20": pred_20,
                "prob_e21": f"{p21_i:.6f}", "pred_e21": pred_21,
                "prob_e24": f"{p24_i:.6f}", "pred_e24": pred_24,
                "e20_vs_e24_transition": t20_24, "e21_vs_e24_transition": t21_24,
                "all_three_wrong": 1 if (not corr_20 and not corr_21 and not corr_24) else 0,
                "generator": row.get("generator", "unknown"),
                "device_family": row.get("device_family", "unknown"),
                "filepath": row.get("filepath", "")
            })
            all_transitions.append({
                "benchmark": bname, "image_id": row.get("image_id", ""), "label": row["label"],
                "e20_vs_e24_transition": t20_24, "e21_vs_e24_transition": t21_24,
                "generator": row.get("generator", "unknown"), "device_family": row.get("device_family", "unknown"),
            })

    # Save CSVs
    ext_csv_path = EXP_DIR / "e24_external_results.csv"
    with open(ext_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_ext_results[0].keys()))
        writer.writeheader()
        for r in all_ext_results: writer.writerow(r)
    print(f"\nSaved external results: {ext_csv_path}")

    trans_csv_path = EXP_DIR / "e24_error_transitions.csv"
    with open(trans_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_transitions[0].keys()))
        writer.writeheader()
        for r in all_transitions: writer.writerow(r)
    print(f"Saved transitions: {trans_csv_path}")

    # Subgroup Analysis
    subgroup_rows = []
    e14_ai = [d for d in all_decisions if d["benchmark"] == "E14 Clean External" and d["label"] == "AI"]
    for g in sorted(set(d["generator"] for d in e14_ai)):
        sub = [d for d in e14_ai if d["generator"] == g]
        n_g = len(sub)
        subgroup_rows.append({
            "subgroup_type": "E14_Generator", "group_name": g, "n_samples": n_g, "metric_type": "AI_Recall",
            "e6c_metric": f"{sum(1 for d in sub if float(d['prob_e6c'])>=0.5)/n_g*100:.2f}%",
            "e20_metric": f"{sum(1 for d in sub if d['pred_e20']==1)/n_g*100:.2f}%",
            "e21_metric": f"{sum(1 for d in sub if d['pred_e21']==1)/n_g*100:.2f}%",
            "e24_metric": f"{sum(1 for d in sub if d['pred_e24']==1)/n_g*100:.2f}%",
        })

    e19_real = [d for d in all_decisions if d["benchmark"] == "E19 Smartphone Benchmark" and d["label"] == "Real"]
    for dev in sorted(set(d["device_family"] for d in e19_real)):
        sub = [d for d in e19_real if d["device_family"] == dev]
        n_d = len(sub)
        subgroup_rows.append({
            "subgroup_type": "E19_Device_Family", "group_name": dev, "n_samples": n_d, "metric_type": "Real_FPR",
            "e6c_metric": f"{sum(1 for d in sub if float(d['prob_e6c'])>=0.5)/n_d*100:.2f}%",
            "e20_metric": f"{sum(1 for d in sub if d['pred_e20']==1)/n_d*100:.2f}%",
            "e21_metric": f"{sum(1 for d in sub if d['pred_e21']==1)/n_d*100:.2f}%",
            "e24_metric": f"{sum(1 for d in sub if d['pred_e24']==1)/n_d*100:.2f}%",
        })

    for wa_bench in ["E19 WhatsApp Subset", "WhatsApp N=67 Benchmark"]:
        sub_wa = [d for d in all_decisions if d["benchmark"] == wa_bench]
        wa_ai = [d for d in sub_wa if d["label"] == "AI"]
        wa_real = [d for d in sub_wa if d["label"] == "Real"]
        if wa_ai:
            subgroup_rows.append({
                "subgroup_type": wa_bench, "group_name": "AI_Images", "n_samples": len(wa_ai), "metric_type": "AI_Recall",
                "e6c_metric": f"{sum(1 for d in wa_ai if float(d['prob_e6c'])>=0.5)/len(wa_ai)*100:.2f}%",
                "e20_metric": f"{sum(1 for d in wa_ai if d['pred_e20']==1)/len(wa_ai)*100:.2f}%",
                "e21_metric": f"{sum(1 for d in wa_ai if d['pred_e21']==1)/len(wa_ai)*100:.2f}%",
                "e24_metric": f"{sum(1 for d in wa_ai if d['pred_e24']==1)/len(wa_ai)*100:.2f}%",
            })
        if wa_real:
            subgroup_rows.append({
                "subgroup_type": wa_bench, "group_name": "Real_Images", "n_samples": len(wa_real), "metric_type": "Real_FPR",
                "e6c_metric": f"{sum(1 for d in wa_real if float(d['prob_e6c'])>=0.5)/len(wa_real)*100:.2f}%",
                "e20_metric": f"{sum(1 for d in wa_real if d['pred_e20']==1)/len(wa_real)*100:.2f}%",
                "e21_metric": f"{sum(1 for d in wa_real if d['pred_e21']==1)/len(wa_real)*100:.2f}%",
                "e24_metric": f"{sum(1 for d in wa_real if d['pred_e24']==1)/len(wa_real)*100:.2f}%",
            })

    subgroup_csv_path = EXP_DIR / "e24_subgroup_results.csv"
    with open(subgroup_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
        writer.writeheader()
        for r in subgroup_rows: writer.writerow(r)
    print(f"Saved subgroup breakdown: {subgroup_csv_path}")

    print("\n" + "=" * 50)
    print("E24 PHASE 2 + EVALUATION COMPLETE")
    print(f"Best Epoch: {best_epoch} | Best Dev F1: {best_dev_f1:.4f}")
    print(f"Research threshold (FPR<=5%): tau={best_thresh_5pct_fpr}")
    print("=" * 50)


if __name__ == "__main__":
    main()
