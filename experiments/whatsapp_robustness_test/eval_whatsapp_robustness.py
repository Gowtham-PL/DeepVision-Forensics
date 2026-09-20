"""
Controlled WhatsApp Robustness Evaluation of E6-C Epoch-2 Checkpoint.

Strict Constraints:
- Production checkpoint: experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt
- 5 views (1 global 224x224 + 4 corner crops 60% resized to 224x224)
- Learned-attention aggregation
- Threshold: 0.50
- Frozen weights, model.eval(), torch.no_grad()
- CUDA if available
- Dataset: data/whatsapp_robustness_test/ (real/ and ai/)
- No training, no threshold tuning, no model modifications, no V2 access
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model

CHECKPOINT_PATH = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
DATA_DIR = PROJECT_ROOT / "data/whatsapp_robustness_test"
OUT_DIR = PROJECT_ROOT / "experiments/whatsapp_robustness_test"
THRESHOLD = 0.50


def generate_5_crops(img: Image.Image) -> list:
    """Exact production 5-view crop generation."""
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    crops = []
    crops.append(img.resize((224, 224), Image.Resampling.BILINEAR))  # View 0: Global
    crops.append(img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR))  # View 1: Top-Left
    crops.append(img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR))  # View 2: Top-Right
    crops.append(img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR))  # View 3: Bottom-Left
    crops.append(img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR))  # View 4: Bottom-Right
    return crops


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Exact Mann-Whitney U statistic ROC-AUC with tie averaging."""
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
    """PR-AUC via trapezoidal integration over precision-recall curve."""
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


def main():
    print("=" * 70)
    print("CONTROLLED WHATSAPP ROBUSTNESS EVALUATION: E6-C EPOCH-2")
    print("=" * 70)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 2. Checkpoint check
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    print(f"[Checkpoint] Loading from: {CHECKPOINT_PATH}")

    # 3. Model construction
    model = MultiViewE5Model(
        checkpoint_path=None,
        num_views=5,
        freq_norm_strategy="standardize",
        freq_embedding_dim=256,
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    print("[Model] E6-C MultiView architecture loaded, weights frozen, in eval mode.")

    # 4. Preprocessing transform (Bilinear resize to 224x224, ToTensor)
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # 5. Gather test images
    real_dir = DATA_DIR / "real"
    ai_dir = DATA_DIR / "ai"

    real_files = sorted([f for f in real_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    ai_files = sorted([f for f in ai_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

    print(f"[Dataset] Found {len(real_files)} REAL images and {len(ai_files)} AI images.")

    all_samples = []
    for f in real_files:
        all_samples.append({"path": f, "label": 0, "label_str": "REAL"})
    for f in ai_files:
        all_samples.append({"path": f, "label": 1, "label_str": "AI"})

    results = []
    y_true = []
    y_prob = []

    print("\n[Inference] Running 5-view production forward pass on all images...")
    with torch.no_grad():
        for idx, sample in enumerate(all_samples, 1):
            img_path = sample["path"]
            label = sample["label"]
            label_str = sample["label_str"]

            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                orig_w, orig_h = img_rgb.size
                crops = generate_5_crops(img_rgb)

            crop_tensors = [preprocess_transform(c) for c in crops]
            x_views = torch.stack(crop_tensors, dim=0).unsqueeze(0).to(device)  # (1, 5, 3, 224, 224)

            # Production pipeline forward pass
            B, V, C, H, W = x_views.shape
            x_flat = x_views.view(B * V, C, H, W)
            feats = model.base_model(x_flat, return_features=True)
            e_fused = feats["fused_embedding"]  # (5, 1792)
            e_fused_views = e_fused.view(B, V, model.fused_dim)

            attn_scores = model.view_attention(e_fused_views)  # (1, 5, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (1, 5, 1)

            # Learned attention pooled embedding
            pooled_embedding = torch.sum(attn_weights * e_fused_views, dim=1)  # (1, 1792)
            logit = model.classifier(pooled_embedding)  # (1, 1)
            overall_prob = float(torch.sigmoid(logit).item())

            # Per-view probabilities
            view_logits = model.classifier(e_fused)  # (5, 1)
            view_probs = torch.sigmoid(view_logits).view(5).cpu().tolist()
            view_w = attn_weights.view(5).cpu().tolist()

            strongest_local = float(max(view_probs[1:]))

            is_pred_ai = overall_prob >= THRESHOLD
            pred_label_str = "AI" if is_pred_ai else "REAL"
            is_correct = (pred_label_str == label_str)

            error_type = "None"
            if not is_correct:
                error_type = "FP" if is_pred_ai else "FN"

            y_true.append(label)
            y_prob.append(overall_prob)

            res_entry = {
                "index": idx,
                "filename": img_path.name,
                "true_label": label_str,
                "true_numeric": label,
                "overall_ai_prob": round(overall_prob, 6),
                "predicted_label": pred_label_str,
                "threshold": THRESHOLD,
                "is_correct": is_correct,
                "error_type": error_type,
                "strongest_local_prob": round(strongest_local, 6),
                "global_prob": round(view_probs[0], 6),
                "top_left_prob": round(view_probs[1], 6),
                "top_right_prob": round(view_probs[2], 6),
                "bottom_left_prob": round(view_probs[3], 6),
                "bottom_right_prob": round(view_probs[4], 6),
                "attn_weight_global": round(view_w[0], 4),
                "attn_weight_tl": round(view_w[1], 4),
                "attn_weight_tr": round(view_w[2], 4),
                "attn_weight_bl": round(view_w[3], 4),
                "attn_weight_br": round(view_w[4], 4),
                "image_width": orig_w,
                "image_height": orig_h,
            }
            results.append(res_entry)

    y_true_np = np.array(y_true, dtype=int)
    y_prob_np = np.array(y_prob, dtype=float)

    # Calculate overall metrics
    tp = int(sum(1 for r in results if r["true_label"] == "AI" and r["predicted_label"] == "AI"))
    tn = int(sum(1 for r in results if r["true_label"] == "REAL" and r["predicted_label"] == "REAL"))
    fp = int(sum(1 for r in results if r["true_label"] == "REAL" and r["predicted_label"] == "AI"))
    fn = int(sum(1 for r in results if r["true_label"] == "AI" and r["predicted_label"] == "REAL"))

    n_total = len(results)
    n_real = len(real_files)
    n_ai = len(ai_files)

    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    roc_auc = compute_roc_auc(y_true_np, y_prob_np)
    pr_auc = compute_pr_auc(y_true_np, y_prob_np)

    # Probability distributions
    real_probs = [r["overall_ai_prob"] for r in results if r["true_label"] == "REAL"]
    ai_probs = [r["overall_ai_prob"] for r in results if r["true_label"] == "AI"]

    real_mean_prob = float(np.mean(real_probs))
    real_median_prob = float(np.median(real_probs))
    ai_mean_prob = float(np.mean(ai_probs))
    ai_median_prob = float(np.median(ai_probs))

    # Strongest local evidence statistics
    real_local_probs = [r["strongest_local_prob"] for r in results if r["true_label"] == "REAL"]
    ai_local_probs = [r["strongest_local_prob"] for r in results if r["true_label"] == "AI"]

    real_local_mean = float(np.mean(real_local_probs))
    real_local_median = float(np.median(real_local_probs))
    real_local_min = float(np.min(real_local_probs))
    real_local_max = float(np.max(real_local_probs))

    ai_local_mean = float(np.mean(ai_local_probs))
    ai_local_median = float(np.median(ai_local_probs))
    ai_local_min = float(np.min(ai_local_probs))
    ai_local_max = float(np.max(ai_local_probs))

    # Subsets metrics
    real_accuracy = tn / n_real if n_real > 0 else 0.0
    ai_accuracy = tp / n_ai if n_ai > 0 else 0.0

    # Save CSV with per-image predictions
    csv_path = OUT_DIR / "per_image_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[Output] Saved CSV predictions: {csv_path}")

    # Summary dictionary for JSON
    summary_data = {
        "evaluation_name": "Controlled WhatsApp Robustness Evaluation",
        "model_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "dataset_path": str(DATA_DIR.relative_to(PROJECT_ROOT)),
        "threshold": THRESHOLD,
        "n_samples": n_total,
        "n_real": n_real,
        "n_ai": n_ai,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
        },
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "probability_distribution": {
            "real": {
                "mean_ai_prob": round(real_mean_prob, 4),
                "median_ai_prob": round(real_median_prob, 4),
                "min_ai_prob": round(float(np.min(real_probs)), 4),
                "max_ai_prob": round(float(np.max(real_probs)), 4),
            },
            "ai": {
                "mean_ai_prob": round(ai_mean_prob, 4),
                "median_ai_prob": round(ai_median_prob, 4),
                "min_ai_prob": round(float(np.min(ai_probs)), 4),
                "max_ai_prob": round(float(np.max(ai_probs)), 4),
            }
        },
        "strongest_local_evidence": {
            "real": {
                "mean": round(real_local_mean, 4),
                "median": round(real_local_median, 4),
                "min": round(real_local_min, 4),
                "max": round(real_local_max, 4),
            },
            "ai": {
                "mean": round(ai_local_mean, 4),
                "median": round(ai_local_median, 4),
                "min": round(ai_local_min, 4),
                "max": round(ai_local_max, 4),
            }
        },
        "subsets": {
            "real_whatsapp": {
                "n": n_real,
                "correct_tn": tn,
                "false_positives": fp,
                "accuracy": round(real_accuracy, 4),
                "fpr": round(fpr, 4),
            },
            "ai_whatsapp": {
                "n": n_ai,
                "correct_tp": tp,
                "false_negatives": fn,
                "accuracy_recall": round(ai_accuracy, 4),
                "fnr": round(fnr, 4),
            }
        },
        "false_positives": [r for r in results if r["error_type"] == "FP"],
        "false_negatives": [r for r in results if r["error_type"] == "FN"],
    }

    json_path = OUT_DIR / "whatsapp_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Output] Saved JSON report: {json_path}")

    # Generate Markdown Report
    md_path = OUT_DIR / "WHATSAPP_ROBUSTNESS_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WhatsApp Robustness Evaluation Report: DeepVision E6-C\n\n")
        f.write("**Evaluation Type**: Controlled WhatsApp Robustness Evaluation (No Training, Frozen Weights)\n\n")
        f.write(f"- **Evaluated Checkpoint**: `{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}`\n")
        f.write(f"- **Dataset**: `{DATA_DIR.relative_to(PROJECT_ROOT)}`\n")
        f.write(f"- **Classification Threshold**: `{THRESHOLD}`\n")
        f.write(f"- **Pipeline Architecture**: 5-View Multi-Scale Spatial + Frequency Fusion with Learned Attention\n")
        f.write(f"- **Execution Hardware**: `{device}`\n\n")

        f.write("## 1. Executive Summary & Core Metrics\n\n")
        f.write("| Metric | Value | Interpretation |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **Total Sample Size (N)** | **{n_total}** | {n_real} Real + {n_ai} AI WhatsApp images |\n")
        f.write(f"| **Overall Accuracy** | **{accuracy * 100:.2f}%** | ({tp + tn} / {n_total} correct) |\n")
        f.write(f"| **Precision** | **{precision * 100:.2f}%** | Precision on AI detection |\n")
        f.write(f"| **Recall (Sensitivity)** | **{recall * 100:.2f}%** | Recall on AI detection ({tp} / {n_ai}) |\n")
        f.write(f"| **F1-Score** | **{f1:.4f}** | Harmonic mean of Precision & Recall |\n")
        f.write(f"| **ROC-AUC** | **{roc_auc:.4f}** | Area under ROC Curve (Ranking discriminability) |\n")
        f.write(f"| **PR-AUC** | **{pr_auc:.4f}** | Area under Precision-Recall Curve |\n")
        f.write(f"| **False Positive Rate (FPR)** | **{fpr * 100:.2f}%** | Real images misclassified as AI ({fp} / {n_real}) |\n")
        f.write(f"| **False Negative Rate (FNR)** | **{fnr * 100:.2f}%** | AI images misclassified as Real ({fn} / {n_ai}) |\n\n")

        f.write("## 2. Confusion Matrix\n\n")
        f.write("| | Predicted Real | Predicted AI | Total |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Actual Real** | **TN = {tn}** ({tn/n_real*100:.1f}%) | **FP = {fp}** ({fp/n_real*100:.1f}%) | {n_real} |\n")
        f.write(f"| **Actual AI** | **FN = {fn}** ({fn/n_ai*100:.1f}%) | **TP = {tp}** ({tp/n_ai*100:.1f}%) | {n_ai} |\n")
        f.write(f"| **Total** | {tn + fn} | {fp + tp} | {n_total} |\n\n")

        f.write("## 3. Probability Distribution Analysis\n\n")
        f.write("| Class | Mean AI Probability | Median AI Probability | Min AI Probability | Max AI Probability |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **REAL Images** (N={n_real}) | {real_mean_prob:.4f} ({real_mean_prob*100:.2f}%) | {real_median_prob:.4f} ({real_median_prob*100:.2f}%) | {min(real_probs):.4f} | {max(real_probs):.4f} |\n")
        f.write(f"| **AI Images** (N={n_ai}) | {ai_mean_prob:.4f} ({ai_mean_prob*100:.2f}%) | {ai_median_prob:.4f} ({ai_median_prob*100:.2f}%) | {min(ai_probs):.4f} | {max(ai_probs):.4f} |\n\n")

        f.write("## 4. Strongest Local Evidence Statistics\n\n")
        f.write("The strongest local evidence reflects the highest AI probability among the 4 deterministic 60% corner crops (Top-Left, Top-Right, Bottom-Left, Bottom-Right):\n\n")
        f.write("| Class | Mean Local Evidence | Median Local Evidence | Min Local Evidence | Max Local Evidence |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **REAL Images** | {real_local_mean:.4f} | {real_local_median:.4f} | {real_local_min:.4f} | {real_local_max:.4f} |\n")
        f.write(f"| **AI Images** | {ai_local_mean:.4f} | {ai_local_median:.4f} | {ai_local_min:.4f} | {ai_local_max:.4f} |\n\n")

        f.write("## 5. Subgroup Breakdown\n\n")
        f.write(f"### A. REAL WhatsApp Images\n")
        f.write(f"- Total Evaluated: **{n_real}**\n")
        f.write(f"- Correctly Identified as Real: **{tn}** ({real_accuracy * 100:.2f}%)\n")
        f.write(f"- False Positives: **{fp}** ({fpr * 100:.2f}%)\n")
        f.write(f"- Mean AI Probability: **{real_mean_prob:.4f}**\n\n")

        f.write(f"### B. AI WhatsApp Images\n")
        f.write(f"- Total Evaluated: **{n_ai}**\n")
        f.write(f"- Correctly Identified as AI: **{tp}** ({ai_accuracy * 100:.2f}%)\n")
        f.write(f"- False Negatives: **{fn}** ({fnr * 100:.2f}%)\n")
        f.write(f"- Mean AI Probability: **{ai_mean_prob:.4f}**\n\n")

        f.write("## 6. Detailed Error Inspection (False Positives & False Negatives)\n\n")
        if fp > 0:
            f.write(f"### False Positives (Real misclassified as AI: {fp})\n\n")
            f.write("| Filename | Overall Prob | Strongest Local Prob | View Probs [Global, TL, TR, BL, BR] | Attn Weights [Global, TL, TR, BL, BR] |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for r in summary_data["false_positives"]:
                vp = [r["global_prob"], r["top_left_prob"], r["top_right_prob"], r["bottom_left_prob"], r["bottom_right_prob"]]
                aw = [r["attn_weight_global"], r["attn_weight_tl"], r["attn_weight_tr"], r["attn_weight_bl"], r["attn_weight_br"]]
                f.write(f"| `{r['filename']}` | **{r['overall_ai_prob']:.4f}** | {r['strongest_local_prob']:.4f} | {vp} | {aw} |\n")
            f.write("\n")
        else:
            f.write("### False Positives\n\n**None! Zero False Positives observed.** All real WhatsApp images were correctly classified as Real.\n\n")

        if fn > 0:
            f.write(f"### False Negatives (AI misclassified as Real: {fn})\n\n")
            f.write("| Filename | Overall Prob | Strongest Local Prob | View Probs [Global, TL, TR, BL, BR] | Attn Weights [Global, TL, TR, BL, BR] |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for r in summary_data["false_negatives"]:
                vp = [r["global_prob"], r["top_left_prob"], r["top_right_prob"], r["bottom_left_prob"], r["bottom_right_prob"]]
                aw = [r["attn_weight_global"], r["attn_weight_tl"], r["attn_weight_tr"], r["attn_weight_bl"], r["attn_weight_br"]]
                f.write(f"| `{r['filename']}` | **{r['overall_ai_prob']:.4f}** | {r['strongest_local_prob']:.4f} | {vp} | {aw} |\n")
            f.write("\n")
        else:
            f.write("### False Negatives\n\n**None! Zero False Negatives observed.** All AI WhatsApp images were correctly classified as AI.\n\n")

        f.write("## 7. Full Per-Image Predictions Table\n\n")
        f.write("Complete per-image tabular evaluation results:\n\n")
        f.write("| # | Filename | Ground Truth | Prediction | AI Prob | Strongest Local | Correct? | Error |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            correct_icon = "PASS" if r["is_correct"] else "FAIL"
            err_str = r["error_type"] if r["error_type"] != "None" else "-"
            f.write(f"| {r['index']} | `{r['filename']}` | {r['true_label']} | {r['predicted_label']} | {r['overall_ai_prob']:.4f} | {r['strongest_local_prob']:.4f} | {correct_icon} | {err_str} |\n")

    print(f"[Output] Saved Markdown report: {md_path}")
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
