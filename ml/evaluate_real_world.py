"""
Real-World Generalization Evaluation for DeepVision-Forensics.

Evaluates and compares two existing frozen models on unconstrained real-world test images:
1. E1 Spatial Baseline: experiments/e1_spatial/best_model.pt
2. E3-Std Dual-Domain Candidate: experiments/candidate_standardize/best_model.pt

Dataset structure expected:
data/real_world_test/
    real/ (Label 0: authentic / real photographs)
    ai/   (Label 1: AI-generated images)

Discovers .png, .jpg, .jpeg, .webp recursively with zero source image modification.
Computes accuracy, precision, recall, F1, confusion matrix, FPR on real, FNR on AI, and ROC-AUC.
Saves results to:
- experiments/real_world_evaluation/results.json
- experiments/real_world_evaluation/summary.csv
- experiments/real_world_evaluation/REAL_WORLD_REPORT.md
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.data import config
from ml.evaluate import compute_metrics, get_raw_rgb_transform, load_model_from_checkpoint


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def discover_images(data_dir: Path) -> List[Dict]:
    """
    Recursively discovers all valid image files in real/ and ai/ subdirectories.
    
    Args:
        data_dir: Path to directory containing real/ and ai/ folders.
        
    Returns:
        List of dictionaries with image path, ground truth label (0 or 1), class name, and relative path.
    """
    images = []
    real_dir = data_dir / "real"
    ai_dir = data_dir / "ai"

    if real_dir.exists():
        for p in sorted(real_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append({
                    "path": p,
                    "label": 0,
                    "class_name": "real",
                    "filename": p.name,
                    "rel_path": str(p.relative_to(data_dir)).replace("\\", "/"),
                })

    if ai_dir.exists():
        for p in sorted(ai_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append({
                    "path": p,
                    "label": 1,
                    "class_name": "ai",
                    "filename": p.name,
                    "rel_path": str(p.relative_to(data_dir)).replace("\\", "/"),
                })

    return images


class RealWorldDataset(Dataset):
    """Dataset for loading and preprocessing real-world test images."""
    
    def __init__(self, image_records: List[Dict], transform: Optional[transforms.Compose] = None):
        self.records = image_records
        self.transform = transform or get_raw_rgb_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        image_path = rec["path"]
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            tensor = self.transform(img_rgb)
        return tensor, rec["label"], rec["rel_path"]


def compute_real_world_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Computes comprehensive evaluation metrics for real-world test sets.
    
    Includes accuracy, precision, recall, F1, confusion matrix, FPR on real, FNR on AI, and ROC-AUC.
    """
    total = len(y_true)
    if total == 0:
        return {}

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n_real = int(np.sum(y_true == 0))
    n_ai = int(np.sum(y_true == 1))

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    fpr_on_real = fp / n_real if n_real > 0 else 0.0
    fnr_on_ai = fn / n_ai if n_ai > 0 else 0.0

    # ROC-AUC (only valid if both classes are present)
    if n_real > 0 and n_ai > 0:
        base_metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        roc_auc = base_metrics.get("roc_auc", 0.5)
        pr_auc = base_metrics.get("pr_auc", 0.5)
    else:
        roc_auc = None
        pr_auc = None

    return {
        "sample_count": total,
        "n_real": n_real,
        "n_ai": n_ai,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "fpr_on_real": float(fpr_on_real),
        "fnr_on_ai": float(fnr_on_ai),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


@torch.no_grad()
def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: RealWorldDataset,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> Tuple[np.ndarray, List[Dict]]:
    """Runs deterministic inference across the dataset and collects predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs = []
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device.type) if device.type == "cuda" else torch.no_grad():
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logit"]
                probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

    probs_np = np.array(all_probs)
    
    per_image_predictions = []
    for i, rec in enumerate(dataset.records):
        prob = float(probs_np[i])
        pred_label = 1 if prob >= 0.5 else 0
        is_correct = (pred_label == rec["label"])
        per_image_predictions.append({
            "filename": rec["filename"],
            "rel_path": rec["rel_path"],
            "ground_truth_label": rec["label"],
            "ground_truth_class": rec["class_name"],
            "ai_probability": round(prob, 4),
            "predicted_label": pred_label,
            "predicted_class": "ai" if pred_label == 1 else "real",
            "is_correct": is_correct,
        })

    return probs_np, per_image_predictions


def run_real_world_evaluation(
    data_dir: str = "data/real_world_test",
    save_dir: str = "experiments/real_world_evaluation",
    e1_checkpoint: str = "experiments/e1_spatial/best_model.pt",
    e3_checkpoint: str = "experiments/candidate_standardize/best_model.pt",
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    report_filename: Optional[str] = None,
) -> Dict:
    """
    Executes real-world generalization evaluation on E1 and E3-Std.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(data_dir)
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DEEPVISION FORENSICS — REAL-WORLD GENERALIZATION EVALUATION")
    print("=" * 80)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Dataset Dir:      {data_path.resolve()}")
    print(f"Output Dir:       {out_path.resolve()}")
    print(f"E1 Checkpoint:    {e1_checkpoint}")
    print(f"E3-Std Checkpoint:{e3_checkpoint}")
    print("=" * 80)

    # 1. Discover images
    image_records = discover_images(data_path)
    n_total = len(image_records)
    n_real = sum(1 for r in image_records if r["label"] == 0)
    n_ai = sum(1 for r in image_records if r["label"] == 1)

    print(f"\n[*] Discovered Images Summary:")
    print(f"    Total Images:  {n_total}")
    print(f"    Real Images:   {n_real} (Label 0)")
    print(f"    AI Images:     {n_ai} (Label 1)")

    if n_total == 0:
        raise ValueError(f"No valid images found in {data_path}/real or {data_path}/ai")

    dataset = RealWorldDataset(image_records)
    y_true = np.array([r["label"] for r in image_records])

    # 2. Evaluate Model 1: E1 Spatial
    print(f"\n[*] Evaluating E1 Spatial Baseline...")
    e1_path = Path(e1_checkpoint)
    if not e1_path.exists():
        raise FileNotFoundError(f"E1 checkpoint not found at {e1_path}")
    
    e1_model = load_model_from_checkpoint(str(e1_path), experiment="E1", device=device)
    e1_probs, e1_per_image = evaluate_model_on_dataset(e1_model, dataset, batch_size=batch_size, device=device)
    e1_metrics = compute_real_world_metrics(y_true, e1_probs, threshold=0.5)

    # 3. Evaluate Model 2: E3-Std Candidate
    print(f"[*] Evaluating E3-Std Dual-Domain Candidate...")
    e3_path = Path(e3_checkpoint)
    if not e3_path.exists():
        raise FileNotFoundError(f"E3-Std checkpoint not found at {e3_path}")
    
    e3_model = load_model_from_checkpoint(str(e3_path), experiment="E3", device=device)
    e3_probs, e3_per_image = evaluate_model_on_dataset(e3_model, dataset, batch_size=batch_size, device=device)
    e3_metrics = compute_real_world_metrics(y_true, e3_probs, threshold=0.5)

    # 4. Compute Comparison Deltas (E3-Std - E1)
    deltas = {
        "accuracy": round(e3_metrics["accuracy"] - e1_metrics["accuracy"], 4),
        "precision": round(e3_metrics["precision"] - e1_metrics["precision"], 4),
        "recall": round(e3_metrics["recall"] - e1_metrics["recall"], 4),
        "f1_score": round(e3_metrics["f1_score"] - e1_metrics["f1_score"], 4),
        "roc_auc": round(e3_metrics["roc_auc"] - e1_metrics["roc_auc"], 4) if (e3_metrics["roc_auc"] is not None and e1_metrics["roc_auc"] is not None) else None,
        "pr_auc": round(e3_metrics["pr_auc"] - e1_metrics["pr_auc"], 4) if (e3_metrics["pr_auc"] is not None and e1_metrics["pr_auc"] is not None) else None,
        "fpr_on_real": round(e3_metrics["fpr_on_real"] - e1_metrics["fpr_on_real"], 4),
        "fnr_on_ai": round(e3_metrics["fnr_on_ai"] - e1_metrics["fnr_on_ai"], 4),
    }

    # 5. Build Combined Results Payload
    combined_results = {
        "metadata": {
            "evaluation_name": "Real-World Generalization Evaluation V2" if "v2" in str(save_dir).lower() else "Real-World Generalization Evaluation",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "dataset_directory": str(data_path),
            "total_images": n_total,
            "real_images_count": n_real,
            "ai_images_count": n_ai,
            "decision_threshold": 0.5,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "models": {
            "E1_Spatial": {
                "model_name": "E1 Spatial Baseline",
                "architecture": "EfficientNet-B3 (Pretrained)",
                "checkpoint": e1_checkpoint,
                "metrics": e1_metrics,
                "per_image_predictions": e1_per_image,
            },
            "E3_Std": {
                "model_name": "E3-Std Dual-Domain Candidate",
                "architecture": "Dual-Domain (Spatial + Frequency Standardization)",
                "checkpoint": e3_checkpoint,
                "metrics": e3_metrics,
                "per_image_predictions": e3_per_image,
            },
        },
        "comparisons": {
            "deltas_e3_std_minus_e1": deltas,
        },
    }

    # 6. Save JSON
    json_file = out_path / "results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\n[*] Saved JSON results to {json_file}")

    # 7. Save CSV Summary
    csv_file = out_path / "summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model_ID", "Model_Name", "Architecture", "Total_Images", "Real_Count", "AI_Count",
            "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC",
            "FPR_on_Real", "FNR_on_AI", "TP", "FP", "TN", "FN"
        ])
        for model_key, model_data in combined_results["models"].items():
            m = model_data["metrics"]
            writer.writerow([
                model_key,
                model_data["model_name"],
                model_data["architecture"],
                m["sample_count"],
                m["n_real"],
                m["n_ai"],
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A",
                f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A",
                f"{m['fpr_on_real']:.4f}",
                f"{m['fnr_on_ai']:.4f}",
                m["tp"],
                m["fp"],
                m["tn"],
                m["fn"],
            ])
    print(f"[*] Saved CSV summary to {csv_file}")

    # 8. Generate Markdown Report
    if report_filename is None:
        report_filename = "REAL_WORLD_V2_REPORT.md" if "v2" in str(save_dir).lower() else "REAL_WORLD_REPORT.md"
    md_file = out_path / report_filename
    generate_markdown_report(combined_results, md_file)
    print(f"[*] Saved Markdown report to {md_file}")

    # 9. Print Terminal Leaderboard
    print("\n" + "=" * 80)
    print("REAL-WORLD TEST EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<18} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'FPR (Real)':<10} | {'FNR (AI)':<9}")
    print("-" * 80)
    for model_key, model_data in combined_results["models"].items():
        m = model_data["metrics"]
        roc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        print(f"{model_data['model_name']:<18} | {m['accuracy']*100:<8.2f}% | {m['precision']*100:<8.2f}% | {m['recall']*100:<8.2f}% | {m['f1_score']:<9.4f} | {roc_str:<9} | {m['fpr_on_real']*100:<9.2f}% | {m['fnr_on_ai']*100:<8.2f}%")
    roc_delta_str = f"{deltas['roc_auc']:+.4f}" if deltas['roc_auc'] is not None else "N/A"
    print(f"Deltas (E3-Std - E1): Accuracy Delta={deltas['accuracy']*100:+.2f}% | Precision Delta={deltas['precision']*100:+.2f}% | Recall Delta={deltas['recall']*100:+.2f}% | ROC-AUC Delta={roc_delta_str}")
    print("=" * 80)

    return combined_results


def generate_markdown_report(results: Dict, output_path: Path) -> None:
    """Generates a detailed Markdown report for real-world evaluation."""
    meta = results["metadata"]
    models = results["models"]
    e1_m = models["E1_Spatial"]["metrics"]
    e3_m = models["E3_Std"]["metrics"]
    deltas = results["comparisons"]["deltas_e3_std_minus_e1"]

    md_lines = [
        "# DeepVision Forensics — Real-World Generalization Evaluation Report",
        "",
        "This report provides an empirical evaluation of **E1 Spatial Baseline** versus **E3-Std Dual-Domain Candidate** on completely unconstrained, real-world images (camera/smartphone photographs, web images, and state-of-the-art AI-synthesized media).",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"- **Dataset Path:** `{meta['dataset_directory']}`",
        f"- **Total Images Evaluated:** **{meta['total_images']}**",
        f"- **Authentic Photographs (`real/`):** **{meta['real_images_count']}** (Ground Truth = 0)",
        f"- **AI-Generated Images (`ai/`):** **{meta['ai_images_count']}** (Ground Truth = 1, e.g., Gemini, Stable Diffusion, Midjourney, Flux)",
        f"- **Decision Threshold:** `0.50` (Fixed, non-tuned)",
        "",
        "---",
        "",
        "## 2. Summary Leaderboard",
        "",
        "| Model ID | Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positive Rate (Real) | False Negative Rate (AI) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_key, model_data in models.items():
        m = model_data["metrics"]
        roc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A"
        md_lines.append(
            f"| **{model_data['model_name']}** | {model_data['architecture']} | "
            f"**{m['accuracy']*100:.2f}%** | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | "
            f"{m['f1_score']:.4f} | **{roc_str}** | {pr_str} | "
            f"{m['fpr_on_real']*100:.2f}% ({m['fp']}/{m['n_real']}) | "
            f"{m['fnr_on_ai']*100:.2f}% ({m['fn']}/{m['n_ai']}) |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Confusion Matrix Breakdown",
        "",
        "| Model | True Positive (AI detected as AI) | False Positive (Real flagged as AI) | True Negative (Real detected as Real) | False Negative (AI missed as Real) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **E1 Spatial** | {e1_m['tp']} ({e1_m['tp']/e1_m['n_ai']*100:.1f}%) | {e1_m['fp']} ({e1_m['fp']/e1_m['n_real']*100:.1f}%) | {e1_m['tn']} ({e1_m['tn']/e1_m['n_real']*100:.1f}%) | {e1_m['fn']} ({e1_m['fn']/e1_m['n_ai']*100:.1f}%) |",
        f"| **E3-Std Dual** | {e3_m['tp']} ({e3_m['tp']/e3_m['n_ai']*100:.1f}%) | {e3_m['fp']} ({e3_m['fp']/e3_m['n_real']*100:.1f}%) | {e3_m['tn']} ({e3_m['tn']/e3_m['n_real']*100:.1f}%) | {e3_m['fn']} ({e3_m['fn']/e3_m['n_ai']*100:.1f}%) |",
        "",
        "---",
        "",
        "## 4. Performance Deltas & Comparative Insights",
        "",
        f"- **Accuracy Delta (E3-Std - E1):** `{deltas['accuracy']*100:+.2f}%` ({e3_m['accuracy']*100:.2f}% vs {e1_m['accuracy']*100:.2f}%)",
        f"- **Precision Delta (E3-Std - E1):** `{deltas['precision']*100:+.2f}%` ({e3_m['precision']*100:.2f}% vs {e1_m['precision']*100:.2f}%)",
        f"- **Recall Delta (E3-Std - E1):** `{deltas['recall']*100:+.2f}%` ({e3_m['recall']*100:.2f}% vs {e1_m['recall']*100:.2f}%)",
        f"- **F1-Score Delta (E3-Std - E1):** `{deltas['f1_score']:+.4f}` ({e3_m['f1_score']:.4f} vs {e1_m['f1_score']:.4f})",
    ])

    auc_delta_str = f"{deltas['roc_auc']:+.4f}" if deltas["roc_auc"] is not None else "N/A"
    e3_auc_str = f"{e3_m['roc_auc']:.4f}" if e3_m["roc_auc"] is not None else "N/A"
    e1_auc_str = f"{e1_m['roc_auc']:.4f}" if e1_m["roc_auc"] is not None else "N/A"

    md_lines.extend([
        f"- **ROC-AUC Delta (E3-Std - E1):** `{auc_delta_str}` ({e3_auc_str} vs {e1_auc_str})",
        f"- **False Positive Rate on Real Images (E3-Std - E1):** `{deltas['fpr_on_real']*100:+.2f}%` ({e3_m['fpr_on_real']*100:.2f}% vs {e1_m['fpr_on_real']*100:.2f}%)",
        f"- **False Negative Rate on AI Images (E3-Std - E1):** `{deltas['fnr_on_ai']*100:+.2f}%` ({e3_m['fnr_on_ai']*100:.2f}% vs {e1_m['fnr_on_ai']*100:.2f}%)",
        "",
        "---",
        "",
        "## 5. Per-Image Prediction Registry",
        "",
        "| Filename | Ground Truth | E1 Prob (AI) | E1 Pred | E1 Correct | E3-Std Prob (AI) | E3-Std Pred | E3-Std Correct |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    e1_preds = {p["rel_path"]: p for p in models["E1_Spatial"]["per_image_predictions"]}
    e3_preds = {p["rel_path"]: p for p in models["E3_Std"]["per_image_predictions"]}

    for rel_path, p1 in e1_preds.items():
        p3 = e3_preds.get(rel_path, {})
        gt_class = p1["ground_truth_class"].upper()
        p1_corr = "PASS" if p1["is_correct"] else "FAIL"
        p3_corr = "PASS" if p3.get("is_correct", False) else "FAIL"
        md_lines.append(
            f"| `{p1['filename']}` | **{gt_class}** | "
            f"`{p1['ai_probability']:.4f}` | {p1['predicted_class'].upper()} | {p1_corr} | "
            f"`{p3.get('ai_probability', 0.0):.4f}` | {p3.get('predicted_class', '').upper()} | {p3_corr} |"
        )

    md_lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Real-World Generalization Evaluation Script")
    parser.add_argument("--data-dir", type=str, default="data/real_world_test", help="Path to real-world test data directory")
    parser.add_argument("--save-dir", type=str, default="experiments/real_world_evaluation", help="Directory to save evaluation artifacts")
    parser.add_argument("--e1-checkpoint", type=str, default="experiments/e1_spatial/best_model.pt", help="Path to E1 model checkpoint")
    parser.add_argument("--e3-checkpoint", type=str, default="experiments/candidate_standardize/best_model.pt", help="Path to E3-Std model checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--report-filename", type=str, default=None, help="Custom filename for Markdown report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_real_world_evaluation(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        e1_checkpoint=args.e1_checkpoint,
        e3_checkpoint=args.e3_checkpoint,
        batch_size=args.batch_size,
        report_filename=args.report_filename,
    )
