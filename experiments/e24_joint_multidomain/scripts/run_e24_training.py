"""
Experiment 24 (E24): Joint Multi-Domain Forensic Detector Training and Sealed Benchmark Evaluation.

Architecture:
- MultiDomainE24Model starting from E20 checkpoint weights.
- 5-view multi-scale input (1 global + 4 corner crops).
- Spatial branch (EfficientNet-B3, 1536-D) + Frequency branch (FFT + CNN, 256-D).
- 256-D shared forensic representation per view.
- 4-view learned attention aggregation for local crops.
- Domain-aware fusion block (global + agg_local + freq_global -> 256-D final forensic vector).
- Multi-task output heads:
  1. Binary Detection Head (Linear 256 -> 1)
  2. Multi-Label Auxiliary Domain Head (Linear 256 -> 5):
     [is_legacy_ai, is_modern_ai, is_smartphone_real, is_camera_real, is_compressed]

Loss:
L_total = BCEWithLogitsLoss(det_logits, y_det) + 0.05 * BCEWithLogitsLoss(dom_logits, y_dom)

Training Protocol:
- Phase 1 (Epochs 1-2): Freeze backbones, train fusion/attention/heads (LR=1e-4).
- Phase 2 (Epochs 3-5): Unfreeze backbones (Backbone LR=1e-5, Heads LR=1e-4).
- Optimizer: AdamW, Weight Decay: 1e-4, AMP FP16.
- Strict Dev Checkpoint Selection + Threshold Study + Sealed External Evaluation.
"""

import os
import sys
import time
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

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

# Directories and Paths
EXP_DIR = PROJECT_ROOT / "experiments/e24_joint_multidomain"
MANIFEST_DIR = EXP_DIR / "manifests"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
SUBGROUP_DIR = EXP_DIR / "subgroups"
PRED_DIR = EXP_DIR / "predictions"

E20_CHECKPOINT = PROJECT_ROOT / "experiments/e20_training/checkpoints/e20_best_model.pt"
E21_CHECKPOINT = PROJECT_ROOT / "experiments/e21_targeted_data/checkpoints/e21_best_model.pt"
E6C_CHECKPOINT = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"

E22_PRED_DIR = PROJECT_ROOT / "experiments/e22_ensemble_study/predictions"

TRAIN_MANIFEST = MANIFEST_DIR / "e24_train_manifest.csv"
DEV_MANIFEST = MANIFEST_DIR / "e24_dev_manifest.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset & Crop Generation ---

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

class MultiDomainE24Dataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        rec = self.records[idx]
        img_path = PROJECT_ROOT / rec["full_filepath"]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                crops = generate_5_crops(img)
                tensors = [self.to_tensor(c) for c in crops]
                x_views = torch.stack(tensors, dim=0) # (5, 3, 224, 224)
        except Exception as e:
            # Fallback black tensor
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


# --- 2. Model Architecture ---

class MultiDomainE24Model(nn.Module):
    def __init__(self, starting_checkpoint: Optional[Path] = None):
        super().__init__()
        # Base dual-domain fusion model
        self.base_model: DeepVisionFusionModel = build_model(
            experiment="E3",
            pretrained=False,
            freq_norm_strategy="standardize",
            freq_embedding_dim=256,
        )

        if starting_checkpoint is not None and starting_checkpoint.exists():
            ckpt = torch.load(starting_checkpoint, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            bm_sd = {k.replace("base_model.", ""): v for k, v in sd.items() if k.startswith("base_model.")}
            self.base_model.load_state_dict(bm_sd, strict=True)
            print(f"[*] MultiDomainE24Model: Loaded E20 base model weights from {starting_checkpoint}")

        self.fused_dim = self.base_model.fused_dim # 1792

        # 1. Shared Forensic Representation per view: 1792 -> 512 -> LN -> GELU -> Dropout -> 256
        self.shared_forensic_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # 2. Multi-view attention over the 4 local views (views 1..4)
        self.local_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        nn.init.zeros_(self.local_attention[2].weight)
        nn.init.zeros_(self.local_attention[2].bias)

        # 3. Domain-Aware Fusion Block: [z_global (256) + z_agg_local (256) + e_freq_global (256)] = 768
        self.fusion_mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # 4. Main Binary AI/Real Classifier
        self.classifier = nn.Linear(256, 1)

        # 5. Auxiliary Multi-Label Domain Head (5 independent outputs)
        self.domain_head = nn.Linear(256, 5)

    def freeze_backbones(self):
        for p in self.base_model.spatial_branch.parameters():
            p.requires_grad = False
        for p in self.base_model.frequency_branch.parameters():
            p.requires_grad = False
        print("[*] Backbones (EfficientNet-B3 + Frequency CNN) FROZEN.")

    def unfreeze_backbones(self):
        for p in self.base_model.spatial_branch.parameters():
            p.requires_grad = True
        for p in self.base_model.frequency_branch.parameters():
            p.requires_grad = True
        print("[*] Backbones (EfficientNet-B3 + Frequency CNN) UNFROZEN.")

    def forward(self, x_views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_views: (B, 5, 3, 224, 224)
        Returns:
            det_logits: (B, 1)
            dom_logits: (B, 5)
            h_forensic: (B, 256)
        """
        B, V, C, H, W = x_views.shape
        x_flat = x_views.view(B * V, C, H, W)

        feats = self.base_model(x_flat, return_features=True)
        e_fused = feats["fused_embedding"] # (B * V, 1792)
        e_freq = feats["frequency_embedding"] # (B * V, 256)

        # Pass each view through shared forensic representation
        z_views = self.shared_forensic_mlp(e_fused) # (B * V, 256)
        z_views = z_views.view(B, V, 256) # (B, 5, 256)

        # Global view representation and global frequency representation
        z_global = z_views[:, 0, :] # (B, 256)
        e_freq_global = e_freq.view(B, V, 256)[:, 0, :] # (B, 256)

        # Local views (views 1..4)
        z_local = z_views[:, 1:, :] # (B, 4, 256)
        attn_scores = self.local_attention(z_local) # (B, 4, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, 4, 1)
        z_agg_local = torch.sum(attn_weights * z_local, dim=1) # (B, 256)

        # Domain-Aware Fusion Block
        combined = torch.cat([z_global, z_agg_local, e_freq_global], dim=1) # (B, 768)
        h_forensic = self.fusion_mlp(combined) # (B, 256)

        # Output heads
        det_logits = self.classifier(h_forensic) # (B, 1)
        dom_logits = self.domain_head(h_forensic) # (B, 5)

        return det_logits, dom_logits, h_forensic


# --- 3. Evaluation Helper ---

@torch.no_grad()
def evaluate_dev(model: MultiDomainE24Model, loader: DataLoader) -> Tuple[Dict, List[Dict]]:
    model.eval()
    criterion_det = nn.BCEWithLogitsLoss()
    criterion_dom = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_y_true = []
    all_p_det = []
    all_y_dom = []
    all_p_dom = []
    item_records = []

    for x_views, y_det, y_dom, rec_list in loader:
        x_views = x_views.to(DEVICE, non_blocking=True)
        y_det = y_det.to(DEVICE, non_blocking=True)
        y_dom = y_dom.to(DEVICE, non_blocking=True)

        det_logits, dom_logits, _ = model(x_views)
        loss_det = criterion_det(det_logits, y_det)
        loss_dom = criterion_dom(dom_logits, y_dom)
        loss = loss_det + 0.05 * loss_dom

        total_loss += loss.item() * len(y_det)

        p_det = torch.sigmoid(det_logits).cpu().numpy().flatten()
        p_dom = torch.sigmoid(dom_logits).cpu().numpy()

        all_y_true.extend(y_det.cpu().numpy().flatten().tolist())
        all_p_det.extend(p_det.tolist())
        all_y_dom.extend(y_dom.cpu().numpy().tolist())
        all_p_dom.extend(p_dom.tolist())

        # Unpack dict of lists from DataLoader
        b_size = len(p_det)
        for i in range(b_size):
            item = {k: rec_list[k][i] for k in rec_list.keys()}
            item["prob_det"] = p_det[i]
            item_records.append(item)

    y_arr = np.array(all_y_true)
    p_arr = np.array(all_p_det)
    m = compute_metrics(y_arr, p_arr, threshold=0.50)
    m["val_loss"] = total_loss / len(y_arr)

    return m, item_records


# --- 4. Main Training and Evaluation Routine ---

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBGROUP_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STARTING EXPERIMENT 24: JOINT MULTI-DOMAIN DETECTOR")
    print("==================================================")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load Manifests
    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_records = list(csv.DictReader(f))
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        dev_records = list(csv.DictReader(f))

    print(f"Loaded Train records: {len(train_records)}, Dev records: {len(dev_records)}")

    train_dataset = MultiDomainE24Dataset(train_records)
    dev_dataset = MultiDomainE24Dataset(dev_records)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    # 2. Instantiate Model
    model = MultiDomainE24Model(starting_checkpoint=E20_CHECKPOINT)
    model.to(DEVICE)

    criterion_det = nn.BCEWithLogitsLoss()
    criterion_dom = nn.BCEWithLogitsLoss()
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None

    epoch_logs = []
    best_dev_f1 = -1.0
    best_epoch = -1
    best_model_path = CKPT_DIR / "e24_best_model.pt"

    # --- PHASE 1: Epochs 1-2 (Backbones Frozen) ---
    print("\n>>> STARTING PHASE 1: Backbones Frozen (Epochs 1-2, LR=1e-4) <<<")
    model.freeze_backbones()

    optimizer_p1 = torch.optim.AdamW([
        {"params": model.shared_forensic_mlp.parameters(), "lr": 1e-4},
        {"params": model.local_attention.parameters(), "lr": 1e-4},
        {"params": model.fusion_mlp.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-4},
        {"params": model.domain_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    for epoch in range(1, 3):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_det_loss = 0.0
        running_dom_loss = 0.0

        for x_views, y_det, y_dom, _ in train_loader:
            x_views = x_views.to(DEVICE, non_blocking=True)
            y_det = y_det.to(DEVICE, non_blocking=True)
            y_dom = y_dom.to(DEVICE, non_blocking=True)

            optimizer_p1.zero_grad()
            with autocast("cuda"):
                det_logits, dom_logits, _ = model(x_views)
                l_det = criterion_det(det_logits, y_det)
                l_dom = criterion_dom(dom_logits, y_dom)
                loss = l_det + 0.05 * l_dom

            scaler.scale(loss).backward()
            scaler.step(optimizer_p1)
            scaler.update()

            running_loss += loss.item() * len(y_det)
            running_det_loss += l_det.item() * len(y_det)
            running_dom_loss += l_dom.item() * len(y_det)

        train_loss = running_loss / len(train_records)
        train_det = running_det_loss / len(train_records)
        train_dom = running_dom_loss / len(train_records)

        dev_m, _ = evaluate_dev(model, dev_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/5 (Phase 1) [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} (Det: {train_det:.4f}, Dom: {train_dom:.4f}) | Dev Acc: {dev_m['accuracy']*100:.2f}%, F1: {dev_m['f1']:.4f}, AUC: {dev_m['roc_auc']:.4f}, Recall: {dev_m['recall']*100:.2f}%, FPR: {dev_m['fpr']*100:.2f}%")

        log_row = {
            "epoch": epoch,
            "phase": "Phase 1 (Backbones Frozen)",
            "train_loss": train_loss,
            "train_det_loss": train_det,
            "train_dom_loss": train_dom,
            "dev_loss": dev_m["val_loss"],
            "dev_accuracy": dev_m["accuracy"],
            "dev_f1": dev_m["f1"],
            "dev_roc_auc": dev_m["roc_auc"],
            "dev_pr_auc": dev_m["pr_auc"],
            "dev_precision": dev_m["precision"],
            "dev_recall": dev_m["recall"],
            "dev_fpr": dev_m["fpr"],
            "dev_fnr": dev_m["fnr"],
            "tp": dev_m["tp"],
            "tn": dev_m["tn"],
            "fp": dev_m["fp"],
            "fn": dev_m["fn"]
        }
        epoch_logs.append(log_row)

        if dev_m["f1"] > best_dev_f1:
            best_dev_f1 = dev_m["f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_metrics": dev_m
            }, best_model_path)
            print(f"  [*] New Best Model saved at Epoch {epoch} (F1 = {dev_m['f1']:.4f})")

    # --- PHASE 2: Epochs 3-5 (Backbones Unfrozen) ---
    print("\n>>> STARTING PHASE 2: Backbones Unfrozen (Epochs 3-5, Backbone LR=1e-5, Heads LR=1e-4) <<<")
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
        running_loss = 0.0
        running_det_loss = 0.0
        running_dom_loss = 0.0

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
            running_det_loss += l_det.item() * len(y_det)
            running_dom_loss += l_dom.item() * len(y_det)

        train_loss = running_loss / len(train_records)
        train_det = running_det_loss / len(train_records)
        train_dom = running_dom_loss / len(train_records)

        dev_m, _ = evaluate_dev(model, dev_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/5 (Phase 2) [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} (Det: {train_det:.4f}, Dom: {train_dom:.4f}) | Dev Acc: {dev_m['accuracy']*100:.2f}%, F1: {dev_m['f1']:.4f}, AUC: {dev_m['roc_auc']:.4f}, Recall: {dev_m['recall']*100:.2f}%, FPR: {dev_m['fpr']*100:.2f}%")

        log_row = {
            "epoch": epoch,
            "phase": "Phase 2 (Backbones Unfrozen)",
            "train_loss": train_loss,
            "train_det_loss": train_det,
            "train_dom_loss": train_dom,
            "dev_loss": dev_m["val_loss"],
            "dev_accuracy": dev_m["accuracy"],
            "dev_f1": dev_m["f1"],
            "dev_roc_auc": dev_m["roc_auc"],
            "dev_pr_auc": dev_m["pr_auc"],
            "dev_precision": dev_m["precision"],
            "dev_recall": dev_m["recall"],
            "dev_fpr": dev_m["fpr"],
            "dev_fnr": dev_m["fnr"],
            "tp": dev_m["tp"],
            "tn": dev_m["tn"],
            "fp": dev_m["fp"],
            "fn": dev_m["fn"]
        }
        epoch_logs.append(log_row)

        if dev_m["f1"] > best_dev_f1:
            best_dev_f1 = dev_m["f1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_metrics": dev_m
            }, best_model_path)
            print(f"  [*] New Best Model saved at Epoch {epoch} (F1 = {dev_m['f1']:.4f})")

    # Save epoch metrics CSV
    epoch_csv_path = EXP_DIR / "e24_epoch_metrics.csv"
    with open(epoch_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_logs[0].keys()))
        writer.writeheader()
        for r in epoch_logs:
            writer.writerow(r)
    print(f"\nSaved Epoch Metrics to {epoch_csv_path}")

    # --- 5. Load Frozen Best Checkpoint for Threshold Study & External Benchmarking ---
    print(f"\n>>> Loading Best Model from Epoch {best_epoch} (F1 = {best_dev_f1:.4f}) <<<")
    best_ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # --- 6. Separate Dev Threshold Study (Dev ONLY) ---
    print("\n--- Running Dev Threshold Study on E24 Dev ONLY (tau = 0.20 to 0.80) ---")
    dev_m_final, dev_items = evaluate_dev(model, dev_loader)
    y_dev_arr = np.array([1 if r["label"] == "AI" else 0 for r in dev_items])
    p_dev_arr = np.array([float(r["prob_det"]) for r in dev_items])

    threshold_rows = []
    best_thresh_unconstrained = 0.50
    best_f1_unconstrained = -1.0
    best_thresh_5pct_fpr = 0.50
    best_f1_5pct_fpr = -1.0

    for tau in np.arange(0.20, 0.81, 0.05):
        tau = round(float(tau), 2)
        tm = compute_metrics(y_dev_arr, p_dev_arr, threshold=tau)
        row = {
            "threshold": tau,
            "accuracy": tm["accuracy"],
            "f1": tm["f1"],
            "precision": tm["precision"],
            "ai_recall": tm["recall"],
            "real_fpr": tm["fpr"],
            "ai_fnr": tm["fnr"],
            "tp": tm["tp"],
            "tn": tm["tn"],
            "fp": tm["fp"],
            "fn": tm["fn"]
        }
        threshold_rows.append(row)

        if tm["f1"] > best_f1_unconstrained:
            best_f1_unconstrained = tm["f1"]
            best_thresh_unconstrained = tau

        if tm["fpr"] <= 0.05 and tm["f1"] > best_f1_5pct_fpr:
            best_f1_5pct_fpr = tm["f1"]
            best_thresh_5pct_fpr = tau

    dev_thresh_csv = EXP_DIR / "e24_dev_thresholds.csv"
    with open(dev_thresh_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        for r in threshold_rows:
            writer.writerow(r)
    print(f"Saved Dev Threshold Study to {dev_thresh_csv}")
    print(f"Best Unconstrained Dev Threshold: tau = {best_thresh_unconstrained} (F1 = {best_f1_unconstrained:.4f})")
    print(f"Best Research Dev Threshold (Real FPR <= 5%): tau = {best_thresh_5pct_fpr} (F1 = {best_f1_5pct_fpr:.4f})")

    # --- 7. Sealed External Benchmark Evaluation ---
    print("\n==================================================")
    print("PHASE B: SEALED EXTERNAL BENCHMARK EVALUATION")
    print("==================================================")

    benchmark_configs = [
        {
            "name": "E14 Clean External",
            "file": E22_PRED_DIR / "e14_clean_e22_predictions.csv",
            "benchmark_key": "e14_clean",
            "n_expected": 200
        },
        {
            "name": "E14 Degraded External",
            "file": E22_PRED_DIR / "e14_degraded_e22_predictions.csv",
            "benchmark_key": "e14_degraded",
            "n_expected": 600
        },
        {
            "name": "E19 Smartphone Benchmark",
            "file": E22_PRED_DIR / "e19_smartphone_benchmark_e22_predictions.csv",
            "benchmark_key": "e19_smartphone",
            "n_expected": 200
        },
        {
            "name": "E19 WhatsApp Subset",
            "file": E22_PRED_DIR / "e19_whatsapp_subset_e22_predictions.csv",
            "benchmark_key": "e19_whatsapp",
            "n_expected": 60
        },
        {
            "name": "WhatsApp N=67 Benchmark",
            "file": E22_PRED_DIR / "whatsapp_n67_benchmark_e22_predictions.csv",
            "benchmark_key": "whatsapp_n67",
            "n_expected": 67
        },
    ]

    all_ext_results = []
    all_decisions = []
    all_transitions = []

    to_tensor = transforms.ToTensor()

    for bcfg in benchmark_configs:
        bname = bcfg["name"]
        print(f"\nEvaluating {bname}...")

        with open(bcfg["file"], "r", encoding="utf-8") as f:
            b_rows = list(csv.DictReader(f))
        assert len(b_rows) == bcfg["n_expected"]

        # Run E24 inference on external benchmark
        p_e24_list = []
        for r in b_rows:
            fp = PROJECT_ROOT / r["filepath"]
            try:
                with Image.open(fp) as img:
                    img = img.convert("RGB")
                    crops = generate_5_crops(img)
                    x_v = torch.stack([to_tensor(c) for c in crops], dim=0).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        det_logit, _, _ = model(x_v)
                        p_val = torch.sigmoid(det_logit).item()
            except Exception as e:
                p_val = 0.5
            p_e24_list.append(p_val)

        p_e24 = np.array(p_e24_list)
        y_true = np.array([1 if r["label"] == "AI" else 0 for r in b_rows])
        p_e6c = np.array([float(r["prob_e6c"]) for r in b_rows])
        p_e20 = np.array([float(r["prob_e20"]) for r in b_rows])
        p_e21 = np.array([float(r["prob_e21"]) for r in b_rows])

        # Baseline & E24 metrics at tau=0.50
        models = [
            ("E6-C Baseline (tau=0.50)", p_e6c, 0.50),
            ("E20 Baseline (tau=0.50)", p_e20, 0.50),
            ("E21 Baseline (tau=0.50)", p_e21, 0.50),
            ("E24 Joint Detector (tau=0.50)", p_e24, 0.50),
            (f"E24 Joint Detector (Research tau={best_thresh_5pct_fpr})", p_e24, best_thresh_5pct_fpr),
        ]

        for mname, p_arr, t_val in models:
            m = compute_metrics(y_true, p_arr, threshold=t_val)
            all_ext_results.append({
                "benchmark": bname,
                "n_samples": len(y_true),
                "model": mname,
                "threshold": t_val,
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "precision": m["precision"],
                "ai_recall": m["recall"],
                "real_fpr": m["fpr"],
                "ai_fnr": m["fnr"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
            })

        # Process per-image error transitions against E20 and E21
        for idx, row in enumerate(b_rows):
            y_i = y_true[idx]
            p20_i = p_e20[idx]
            p21_i = p_e21[idx]
            p24_i = p_e24[idx]

            pred_20 = 1 if p20_i >= 0.50 else 0
            pred_21 = 1 if p21_i >= 0.50 else 0
            pred_24 = 1 if p24_i >= 0.50 else 0

            corr_20 = (pred_20 == y_i)
            corr_21 = (pred_21 == y_i)
            corr_24 = (pred_24 == y_i)

            # Transitions
            if corr_20 and corr_24: trans_20_24 = "E20_correct__E24_correct"
            elif corr_20 and not corr_24: trans_20_24 = "E20_correct__E24_wrong"
            elif not corr_20 and corr_24: trans_20_24 = "E20_wrong__E24_correct"
            else: trans_20_24 = "E20_wrong__E24_wrong"

            if corr_21 and corr_24: trans_21_24 = "E21_correct__E24_correct"
            elif corr_21 and not corr_24: trans_21_24 = "E21_correct__E24_wrong"
            elif not corr_21 and corr_24: trans_21_24 = "E21_wrong__E24_correct"
            else: trans_21_24 = "E21_wrong__E24_wrong"

            all_decisions.append({
                "benchmark": bname,
                "image_id": row["image_id"],
                "label": row["label"],
                "prob_e6c": row["prob_e6c"],
                "prob_e20": f"{p20_i:.6f}",
                "pred_e20": pred_20,
                "prob_e21": f"{p21_i:.6f}",
                "pred_e21": pred_21,
                "prob_e24": f"{p24_i:.6f}",
                "pred_e24": pred_24,
                "e20_vs_e24_transition": trans_20_24,
                "e21_vs_e24_transition": trans_21_24,
                "all_three_wrong": 1 if (not corr_20 and not corr_21 and not corr_24) else 0,
                "generator": row.get("generator", "unknown"),
                "device_family": row.get("device_family", "unknown"),
                "filepath": row.get("filepath", "")
            })

            all_transitions.append({
                "benchmark": bname,
                "image_id": row["image_id"],
                "label": row["label"],
                "e20_vs_e24_transition": trans_20_24,
                "e21_vs_e24_transition": trans_21_24,
                "generator": row.get("generator", "unknown"),
                "device_family": row.get("device_family", "unknown"),
            })

    # Save external results CSV
    ext_csv_path = EXP_DIR / "e24_external_results.csv"
    with open(ext_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_ext_results[0].keys()))
        writer.writeheader()
        for r in all_ext_results:
            writer.writerow(r)
    print(f"\nSaved External Results to {ext_csv_path}")

    # Save transitions CSV
    trans_csv_path = EXP_DIR / "e24_error_transitions.csv"
    with open(trans_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_transitions[0].keys()))
        writer.writeheader()
        for r in all_transitions:
            writer.writerow(r)
    print(f"Saved Error Transitions to {trans_csv_path}")

    # --- 8. Subgroup Breakdown Generation ---
    subgroup_rows = []

    # E14 Clean Generators
    e14_ai = [d for d in all_decisions if d["benchmark"] == "E14 Clean External" and d["label"] == "AI"]
    gens = sorted(list(set(d["generator"] for d in e14_ai)))
    for g in gens:
        sub = [d for d in e14_ai if d["generator"] == g]
        n_g = len(sub)
        rec_e6c = sum(1 for d in sub if float(d["prob_e6c"]) >= 0.50) / n_g * 100.0
        rec_e20 = sum(1 for d in sub if d["pred_e20"] == 1) / n_g * 100.0
        rec_e21 = sum(1 for d in sub if d["pred_e21"] == 1) / n_g * 100.0
        rec_e24 = sum(1 for d in sub if d["pred_e24"] == 1) / n_g * 100.0
        subgroup_rows.append({
            "subgroup_type": "E14_Generator",
            "group_name": g,
            "n_samples": n_g,
            "metric_type": "AI_Recall",
            "e6c_metric": f"{rec_e6c:.2f}%",
            "e20_metric": f"{rec_e20:.2f}%",
            "e21_metric": f"{rec_e21:.2f}%",
            "e24_metric": f"{rec_e24:.2f}%",
        })

    # E19 Smartphone Devices
    e19_real = [d for d in all_decisions if d["benchmark"] == "E19 Smartphone Benchmark" and d["label"] == "Real"]
    devs = sorted(list(set(d["device_family"] for d in e19_real)))
    for dev in devs:
        sub = [d for d in e19_real if d["device_family"] == dev]
        n_d = len(sub)
        fpr_e6c = sum(1 for d in sub if float(d["prob_e6c"]) >= 0.50) / n_d * 100.0
        fpr_e20 = sum(1 for d in sub if d["pred_e20"] == 1) / n_d * 100.0
        fpr_e21 = sum(1 for d in sub if d["pred_e21"] == 1) / n_d * 100.0
        fpr_e24 = sum(1 for d in sub if d["pred_e24"] == 1) / n_d * 100.0
        subgroup_rows.append({
            "subgroup_type": "E19_Device_Family",
            "group_name": dev,
            "n_samples": n_d,
            "metric_type": "Real_FPR",
            "e6c_metric": f"{fpr_e6c:.2f}%",
            "e20_metric": f"{fpr_e20:.2f}%",
            "e21_metric": f"{fpr_e21:.2f}%",
            "e24_metric": f"{fpr_e24:.2f}%",
        })

    # WhatsApp Subgroups
    for wa_bench in ["E19 WhatsApp Subset", "WhatsApp N=67 Benchmark"]:
        sub_wa = [d for d in all_decisions if d["benchmark"] == wa_bench]
        wa_ai = [d for d in sub_wa if d["label"] == "AI"]
        wa_real = [d for d in sub_wa if d["label"] == "Real"]

        # AI Recall
        subgroup_rows.append({
            "subgroup_type": wa_bench,
            "group_name": "AI_Images",
            "n_samples": len(wa_ai),
            "metric_type": "AI_Recall",
            "e6c_metric": f"{sum(1 for d in wa_ai if float(d['prob_e6c'])>=0.5)/len(wa_ai)*100:.2f}%",
            "e20_metric": f"{sum(1 for d in wa_ai if d['pred_e20']==1)/len(wa_ai)*100:.2f}%",
            "e21_metric": f"{sum(1 for d in wa_ai if d['pred_e21']==1)/len(wa_ai)*100:.2f}%",
            "e24_metric": f"{sum(1 for d in wa_ai if d['pred_e24']==1)/len(wa_ai)*100:.2f}%",
        })

        # Real FPR
        subgroup_rows.append({
            "subgroup_type": wa_bench,
            "group_name": "Real_Images",
            "n_samples": len(wa_real),
            "metric_type": "Real_FPR",
            "e6c_metric": f"{sum(1 for d in wa_real if float(d['prob_e6c'])>=0.5)/len(wa_real)*100:.2f}%",
            "e20_metric": f"{sum(1 for d in wa_real if d['pred_e20']==1)/len(wa_real)*100:.2f}%",
            "e21_metric": f"{sum(1 for d in wa_real if d['pred_e21']==1)/len(wa_real)*100:.2f}%",
            "e24_metric": f"{sum(1 for d in wa_real if d['pred_e24']==1)/len(wa_real)*100:.2f}%",
        })

    subgroup_csv_path = EXP_DIR / "e24_subgroup_results.csv"
    with open(subgroup_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
        writer.writeheader()
        for r in subgroup_rows:
            writer.writerow(r)
    print(f"Saved Subgroup Breakdown to {subgroup_csv_path}")

    print("\n==================================================")
    print("EXPERIMENT 24 EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
