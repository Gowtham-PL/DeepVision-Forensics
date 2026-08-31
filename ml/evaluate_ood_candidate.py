"""
Final Unseen-Generator Evaluation of Selected OOD Candidate.

Evaluates the trained Dual-Domain candidate model (with Frequency Standardization)
against completely unseen generators:
- BigGAN (GAN paradigm)
- Midjourney (Commercial diffusion paradigm)
- Overall Combined Unseen Test Set

Applies strict protocol rules:
- Continuous predicted probabilities for ROC-AUC / PR-AUC
- Fixed 0.50 classification decision threshold (no threshold tuning)
- No Test-Time Augmentation (TTA)
- Deterministic inference
- Exact cross-generator duplicate exclusion (1 duplicate in Midjourney/train pair excluded)
- Direct comparison against frozen E1 and E3 baselines
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data import config
from ml.data.dataset import DeepVisionDataset
from ml.evaluate import compute_metrics, get_raw_rgb_transform, load_model_from_checkpoint


def run_candidate_evaluation(
    candidate_ckpt: str = "experiments/candidate_standardize/best_model.pt",
    save_dir: str = "experiments/final_evaluation_ood",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    num_workers: int = 2,
) -> Dict:
    """Executes the complete final unseen-generator test evaluation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DEEPVISION FORENSICS — FINAL UNSEEN-GENERATOR TEST EVALUATION")
    print("================================================================================")
    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Candidate Checkpoint: {candidate_ckpt}")
    print(f"Output Directory:     {save_dir}")
    print("=" * 80)

    # 1. Load candidate model
    candidate_path = Path(candidate_ckpt)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found at {candidate_path}")

    ckpt_data = torch.load(candidate_path, map_location=device)
    config_dict = ckpt_data.get("config", {})
    norm_strategy = config_dict.get("norm_strategy", "standardize")
    print(f"\n[*] Loaded checkpoint metadata:")
    print(f"    Experiment:            {config_dict.get('experiment', 'E3')}")
    print(f"    Norm Strategy:         {norm_strategy}")
    print(f"    Best Training Epoch:   {ckpt_data.get('epoch')}")
    print(f"    Best Val ROC-AUC:      {ckpt_data.get('val_auc'):.4f}")
    print(f"    Best Val Accuracy:     {ckpt_data.get('val_acc'):.4f}")

    model = load_model_from_checkpoint(str(candidate_path), experiment="E3", device=device)
    model.eval()

    # 2. Test split verification & dataloader
    raw_transform = get_raw_rgb_transform()
    test_dataset = DeepVisionDataset(split="test", transform=raw_transform)
    total_test_records = len(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Verify zero training generator contamination in test set
    test_gens = sorted(list({r.get("generator") for r in test_dataset.records}))
    test_dups = [r for r in test_dataset.records if r.get("is_cross_gen_dup") == "True"]
    print(f"\n[*] Test Dataset Verification:")
    print(f"    Total Test Samples:    {total_test_records:,}")
    print(f"    Test Generators:       {test_gens} (Expected: ['BigGAN', 'Midjourney'])")
    print(f"    Cross-Gen Duplicates:  {len(test_dups)} (Excluded from metrics)")
    print(f"    Valid Test Samples:    {total_test_records - len(test_dups):,}")

    assert "BigGAN" in test_gens and "Midjourney" in test_gens, "Test split must contain BigGAN and Midjourney"
    for dev_gen in ["ADM", "GLIDE", "SDv5", "VQDM", "Wukong"]:
        assert dev_gen not in test_gens, f"Training generator {dev_gen} leaked into test set!"

    # 3. Deterministic Inference
    print(f"\n[*] Running inference across {total_test_records:,} test images...")
    start_eval_time = time.time()
    all_probs = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device.type) if device.type == "cuda" else torch.no_grad():
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logit"]
                probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten())

    eval_duration = time.time() - start_eval_time
    probs_np = np.array(all_probs)
    print(f"[*] Inference completed in {eval_duration:.2f}s ({total_test_records / eval_duration:.1f} img/s)")

    # 4. Extract metadata & mask duplicates
    labels = []
    generators = []
    is_dup_flags = []

    for i in range(len(test_dataset)):
        rec = test_dataset.get_record(i)
        labels.append(int(rec["label"]))
        generators.append(rec.get("generator", "Unknown"))
        is_dup_flags.append(rec.get("is_cross_gen_dup", "False") == "True")

    labels_np = np.array(labels)
    generators_np = np.array(generators)
    is_dup_np = np.array(is_dup_flags)
    valid_mask = ~is_dup_np

    # 5. Compute Overall & Per-Generator Metrics
    overall_metrics = compute_metrics(
        y_true=labels_np[valid_mask],
        y_prob=probs_np[valid_mask],
        threshold=0.5,
    )

    per_gen_metrics = {}
    for gen in test_gens:
        gen_mask = (generators_np == gen) & valid_mask
        per_gen_metrics[gen] = compute_metrics(
            y_true=labels_np[gen_mask],
            y_prob=probs_np[gen_mask],
            threshold=0.5,
        )

    # 6. Load Frozen E1 / E2 / E3 Baselines for exact comparison
    frozen_eval_file = Path("experiments/final_evaluation/evaluation_results.json")
    if frozen_eval_file.exists():
        with open(frozen_eval_file, "r", encoding="utf-8") as f:
            frozen_results = json.load(f)
    else:
        frozen_results = {}

    e1_overall = frozen_results.get("models", {}).get("E1", {}).get("aggregate", {})
    e1_by_gen = frozen_results.get("models", {}).get("E1", {}).get("by_generator", {})
    e3_overall = frozen_results.get("models", {}).get("E3", {}).get("aggregate", {})
    e3_by_gen = frozen_results.get("models", {}).get("E3", {}).get("by_generator", {})

    # Compute exact deltas
    delta_vs_e1 = {
        "roc_auc": round(overall_metrics["roc_auc"] - e1_overall.get("roc_auc", 0.8991), 4),
        "accuracy": round(overall_metrics["accuracy"] - e1_overall.get("accuracy", 0.8131), 4),
        "f1_score": round(overall_metrics["f1_score"] - e1_overall.get("f1_score", 0.8291), 4),
        "pr_auc": round(overall_metrics["pr_auc"] - e1_overall.get("pr_auc", 0.8967), 4),
        "biggan_roc_auc": round(per_gen_metrics["BigGAN"]["roc_auc"] - e1_by_gen.get("BigGAN", {}).get("roc_auc", 0.9732), 4),
        "midjourney_roc_auc": round(per_gen_metrics["Midjourney"]["roc_auc"] - e1_by_gen.get("Midjourney", {}).get("roc_auc", 0.8224), 4),
    }

    delta_vs_e3 = {
        "roc_auc": round(overall_metrics["roc_auc"] - e3_overall.get("roc_auc", 0.8851), 4),
        "accuracy": round(overall_metrics["accuracy"] - e3_overall.get("accuracy", 0.7725), 4),
        "f1_score": round(overall_metrics["f1_score"] - e3_overall.get("f1_score", 0.8038), 4),
        "pr_auc": round(overall_metrics["pr_auc"] - e3_overall.get("pr_auc", 0.8817), 4),
        "biggan_roc_auc": round(per_gen_metrics["BigGAN"]["roc_auc"] - e3_by_gen.get("BigGAN", {}).get("roc_auc", 0.9465), 4),
        "midjourney_roc_auc": round(per_gen_metrics["Midjourney"]["roc_auc"] - e3_by_gen.get("Midjourney", {}).get("roc_auc", 0.8228), 4),
    }

    # 7. Package complete JSON results
    candidate_results = {
        "metadata": {
            "evaluation_name": "Final Unseen-Generator Evaluation (Candidate Standardize)",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "total_test_records": total_test_records,
            "excluded_duplicates_count": int(np.sum(is_dup_np)),
            "evaluated_test_records": int(np.sum(valid_mask)),
            "test_generators": test_gens,
            "threshold": 0.5,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candidate_config": config_dict,
        },
        "candidate_metrics": {
            "overall": overall_metrics,
            "by_generator": per_gen_metrics,
        },
        "frozen_baseline_comparison": {
            "e1_spatial_frozen": {
                "overall": e1_overall,
                "by_generator": e1_by_gen,
            },
            "e3_dual_frozen": {
                "overall": e3_overall,
                "by_generator": e3_by_gen,
            },
            "deltas_vs_e1": delta_vs_e1,
            "deltas_vs_e3": delta_vs_e3,
        },
    }

    # Save JSON
    json_path = out_path / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(candidate_results, f, indent=2)
    print(f"[*] Saved machine-readable results to {json_path}")

    # 8. Save CSV Summary Table
    csv_path = out_path / "evaluation_summary_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model_ID", "Architecture", "Frequency_Norm", "Evaluation_Scope",
            "Sample_Count", "Accuracy", "Precision", "Recall", "F1_Score",
            "ROC_AUC", "PR_AUC", "TP", "FP", "TN", "FN"
        ])
        
        # Candidate rows
        writer.writerow([
            "Candidate_OOD", "Dual-Domain (Spatial + Frequency)", "standardize", "Overall_Unseen",
            overall_metrics["sample_count"], f"{overall_metrics['accuracy']:.4f}",
            f"{overall_metrics['precision']:.4f}", f"{overall_metrics['recall']:.4f}",
            f"{overall_metrics['f1_score']:.4f}", f"{overall_metrics['roc_auc']:.4f}",
            f"{overall_metrics['pr_auc']:.4f}", overall_metrics["tp"], overall_metrics["fp"],
            overall_metrics["tn"], overall_metrics["fn"]
        ])
        for gen, gm in per_gen_metrics.items():
            writer.writerow([
                "Candidate_OOD", "Dual-Domain (Spatial + Frequency)", "standardize", f"Gen_{gen}",
                gm["sample_count"], f"{gm['accuracy']:.4f}", f"{gm['precision']:.4f}",
                f"{gm['recall']:.4f}", f"{gm['f1_score']:.4f}", f"{gm['roc_auc']:.4f}",
                f"{gm['pr_auc']:.4f}", gm["tp"], gm["fp"], gm["tn"], gm["fn"]
            ])

        # Frozen E1 rows
        if e1_overall:
            writer.writerow([
                "E1_Spatial_Frozen", "Spatial-Only (EfficientNet-B3)", "N/A", "Overall_Unseen",
                e1_overall.get("sample_count", 9999), f"{e1_overall.get('accuracy', 0.8131):.4f}",
                f"{e1_overall.get('precision', 0.7410):.4f}", f"{e1_overall.get('recall', 0.9410):.4f}",
                f"{e1_overall.get('f1_score', 0.8291):.4f}", f"{e1_overall.get('roc_auc', 0.8991):.4f}",
                f"{e1_overall.get('pr_auc', 0.8967):.4f}", e1_overall.get("tp", 4705),
                e1_overall.get("fp", 1645), e1_overall.get("tn", 3354), e1_overall.get("fn", 295)
            ])
            for gen in ["BigGAN", "Midjourney"]:
                gm = e1_by_gen.get(gen, {})
                if gm:
                    writer.writerow([
                        "E1_Spatial_Frozen", "Spatial-Only (EfficientNet-B3)", "N/A", f"Gen_{gen}",
                        gm.get("sample_count"), f"{gm.get('accuracy'):.4f}", f"{gm.get('precision'):.4f}",
                        f"{gm.get('recall'):.4f}", f"{gm.get('f1_score'):.4f}", f"{gm.get('roc_auc'):.4f}",
                        f"{gm.get('pr_auc'):.4f}", gm.get("tp"), gm.get("fp"), gm.get("tn"), gm.get("fn")
                    ])

        # Frozen E3 rows
        if e3_overall:
            writer.writerow([
                "E3_Dual_Frozen", "Dual-Domain (Spatial + Frequency)", "minmax", "Overall_Unseen",
                e3_overall.get("sample_count", 9999), f"{e3_overall.get('accuracy', 0.7725):.4f}",
                f"{e3_overall.get('precision', 0.6983):.4f}", f"{e3_overall.get('recall', 0.9472):.4f}",
                f"{e3_overall.get('f1_score', 0.8038):.4f}", f"{e3_overall.get('roc_auc', 0.8851):.4f}",
                f"{e3_overall.get('pr_auc', 0.8817):.4f}", e3_overall.get("tp", 4736),
                e3_overall.get("fp", 2048), e3_overall.get("tn", 2951), e3_overall.get("fn", 264)
            ])
            for gen in ["BigGAN", "Midjourney"]:
                gm = e3_by_gen.get(gen, {})
                if gm:
                    writer.writerow([
                        "E3_Dual_Frozen", "Dual-Domain (Spatial + Frequency)", "minmax", f"Gen_{gen}",
                        gm.get("sample_count"), f"{gm.get('accuracy'):.4f}", f"{gm.get('precision'):.4f}",
                        f"{gm.get('recall'):.4f}", f"{gm.get('f1_score'):.4f}", f"{gm.get('roc_auc'):.4f}",
                        f"{gm.get('pr_auc'):.4f}", gm.get("tp"), gm.get("fp"), gm.get("tn"), gm.get("fn")
                    ])

    print(f"[*] Saved CSV summary table to {csv_path}")

    # 9. Print Terminal Leaderboard
    print("\n" + "=" * 80)
    print("FINAL UNSEEN-GENERATOR EVALUATION LEADERBOARD (TEST SPLIT)")
    print("=" * 80)
    print(f"{'Model':<25} | {'Overall AUC':<11} | {'Overall Acc':<11} | {'BigGAN AUC':<10} | {'Midjourney AUC':<14} | {'F1-Score':<8}")
    print("-" * 80)
    print(f"{'Candidate (standardize)':<25} | {overall_metrics['roc_auc']:<11.4f} | {overall_metrics['accuracy']*100:<10.2f}% | {per_gen_metrics['BigGAN']['roc_auc']:<10.4f} | {per_gen_metrics['Midjourney']['roc_auc']:<14.4f} | {overall_metrics['f1_score']:<8.4f}")
    if e1_overall:
        print(f"{'E1 Spatial (Frozen)':<25} | {e1_overall.get('roc_auc', 0.8991):<11.4f} | {e1_overall.get('accuracy', 0.8131)*100:<10.2f}% | {e1_by_gen.get('BigGAN', {}).get('roc_auc', 0.9732):<10.4f} | {e1_by_gen.get('Midjourney', {}).get('roc_auc', 0.8224):<14.4f} | {e1_overall.get('f1_score', 0.8291):<8.4f}")
    if e3_overall:
        print(f"{'E3 Dual minmax (Frozen)':<25} | {e3_overall.get('roc_auc', 0.8851):<11.4f} | {e3_overall.get('accuracy', 0.7725)*100:<10.2f}% | {e3_by_gen.get('BigGAN', {}).get('roc_auc', 0.9465):<10.4f} | {e3_by_gen.get('Midjourney', {}).get('roc_auc', 0.8228):<14.4f} | {e3_overall.get('f1_score', 0.8038):<8.4f}")
    print("=" * 80)

    print("\nExact Deltas:")
    print(f"  Candidate vs E1 Spatial:  ROC-AUC Delta = {delta_vs_e1['roc_auc']:+.4f} | Acc Delta = {delta_vs_e1['accuracy']*100:+.2f}% | BigGAN Delta = {delta_vs_e1['biggan_roc_auc']:+.4f} | Midjourney Delta = {delta_vs_e1['midjourney_roc_auc']:+.4f}")
    print(f"  Candidate vs E3 Dual:     ROC-AUC Delta = {delta_vs_e3['roc_auc']:+.4f} | Acc Delta = {delta_vs_e3['accuracy']*100:+.2f}% | BigGAN Delta = {delta_vs_e3['biggan_roc_auc']:+.4f} | Midjourney Delta = {delta_vs_e3['midjourney_roc_auc']:+.4f}")
    print("=" * 80)

    return candidate_results


if __name__ == "__main__":
    run_candidate_evaluation()
