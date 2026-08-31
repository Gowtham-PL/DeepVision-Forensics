"""
Generates publication-quality standalone SVG visualization figures for the DeepVision-Forensics
Final Research Results Package using only the frozen experiment JSON artifacts.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
E1_HIST = PROJECT_ROOT / "experiments" / "e1_spatial" / "training_history.json"
E2_HIST = PROJECT_ROOT / "experiments" / "e2_frequency" / "training_history.json"
E3_HIST = PROJECT_ROOT / "experiments" / "e3_dual_domain" / "training_history.json"
EVAL_RES = PROJECT_ROOT / "experiments" / "final_evaluation" / "evaluation_results.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "final_results"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(E1_HIST, "r", encoding="utf-8") as f:
        e1_h = json.load(f)
    with open(E2_HIST, "r", encoding="utf-8") as f:
        e2_h = json.load(f)
    with open(E3_HIST, "r", encoding="utf-8") as f:
        e3_h = json.load(f)
    with open(EVAL_RES, "r", encoding="utf-8") as f:
        eval_r = json.load(f)
    return e1_h, e2_h, e3_h, eval_r


def create_line_chart_svg(
    title: str,
    y_label: str,
    series_data: List[Tuple[str, str, List[float]]],
    y_range: Tuple[float, float],
    filename: str,
):
    width, height = 700, 420
    margin_l, margin_r, margin_t, margin_b = 80, 160, 60, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    min_y, max_y = y_range
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def x_px(ep):
        return margin_l + (ep - 1) / 9.0 * plot_w

    def y_px(val):
        norm = (val - min_y) / (max_y - min_y + 1e-8)
        return margin_t + plot_h - norm * plot_h

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 600; fill: #1e293b; }',
        '  .axis { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; }',
        '  .label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 500; fill: #334155; }',
        '  .legend { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 600; }',
        '  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4 4; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8" />',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">{title}</text>',
    ]

    # Grid & Y ticks (5 ticks)
    for i in range(5):
        frac = i / 4.0
        val = min_y + frac * (max_y - min_y)
        y = margin_t + plot_h - frac * plot_h
        svg_lines.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" y2="{y}" class="grid" />')
        svg_lines.append(f'<text x="{margin_l - 10}" y="{y + 4}" text-anchor="end" class="axis">{val:.2f}</text>')

    # X ticks
    for ep in epochs:
        x = x_px(ep)
        svg_lines.append(f'<line x1="{x}" y1="{margin_t}" x2="{x}" y2="{margin_t + plot_h}" class="grid" />')
        svg_lines.append(f'<text x="{x}" y="{margin_t + plot_h + 20}" text-anchor="middle" class="axis">{ep}</text>')

    # Axis Lines
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')

    # Axis Labels
    svg_lines.append(f'<text x="{margin_l + plot_w / 2}" y="{height - 15}" text-anchor="middle" class="label">Training Epoch</text>')
    svg_lines.append(f'<text x="24" y="{margin_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {margin_t + plot_h / 2})" class="label">{y_label}</text>')

    # Series
    legend_y = margin_t + 20
    for name, color, values in series_data:
        points = [f"{x_px(ep):.1f},{y_px(v):.1f}" for ep, v in zip(epochs, values)]
        polyline = " ".join(points)
        svg_lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{polyline}" />')

        # Marker dots
        for ep, v in zip(epochs, values):
            svg_lines.append(f'<circle cx="{x_px(ep):.1f}" cy="{y_px(v):.1f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5" />')

        # Legend
        svg_lines.append(f'<circle cx="{margin_l + plot_w + 25}" cy="{legend_y}" r="5" fill="{color}" />')
        svg_lines.append(f'<text x="{margin_l + plot_w + 38}" y="{legend_y + 4}" class="legend" fill="{color}">{name}</text>')
        legend_y += 26

    svg_lines.append('</svg>')
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def create_bar_chart_svg(
    title: str,
    categories: List[str],
    models: List[Tuple[str, str, List[float]]],
    y_range: Tuple[float, float],
    filename: str,
):
    width, height = 750, 420
    margin_l, margin_r, margin_t, margin_b = 80, 180, 60, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    min_y, max_y = y_range
    num_cats = len(categories)
    num_models = len(models)
    cat_width = plot_w / num_cats
    bar_width = (cat_width * 0.7) / num_models

    def y_px(val):
        norm = (val - min_y) / (max_y - min_y + 1e-8)
        return margin_t + plot_h - norm * plot_h

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 600; fill: #1e293b; }',
        '  .axis { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; }',
        '  .cat { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 600; fill: #334155; }',
        '  .legend { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 600; }',
        '  .bar-val { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10px; font-weight: 600; fill: #1e293b; }',
        '  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4 4; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8" />',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">{title}</text>',
    ]

    # Grid & Y ticks
    for i in range(6):
        frac = i / 5.0
        val = min_y + frac * (max_y - min_y)
        y = margin_t + plot_h - frac * plot_h
        svg_lines.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" y2="{y}" class="grid" />')
        svg_lines.append(f'<text x="{margin_l - 10}" y="{y + 4}" text-anchor="end" class="axis">{val:.2f}</text>')

    # Axis Lines
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<text x="24" y="{margin_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {margin_t + plot_h / 2})" class="cat">ROC-AUC</text>')

    # Draw bars
    for c_idx, cat in enumerate(categories):
        c_center = margin_l + c_idx * cat_width + cat_width / 2.0
        svg_lines.append(f'<text x="{c_center}" y="{margin_t + plot_h + 24}" text-anchor="middle" class="cat">{cat}</text>')

        group_start = c_center - (num_models * bar_width) / 2.0
        for m_idx, (m_name, color, values) in enumerate(models):
            val = values[c_idx]
            bx = group_start + m_idx * bar_width
            by = y_px(val)
            bh = (margin_t + plot_h) - by
            svg_lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 4:.1f}" height="{bh:.1f}" fill="{color}" rx="3" />')
            svg_lines.append(f'<text x="{bx + (bar_width - 4)/2:.1f}" y="{by - 6:.1f}" text-anchor="middle" class="bar-val">{val:.4f}</text>')

    # Legend
    legend_y = margin_t + 20
    for m_name, color, _ in models:
        svg_lines.append(f'<rect x="{margin_l + plot_w + 20}" y="{legend_y - 10}" width="14" height="14" fill="{color}" rx="3" />')
        svg_lines.append(f'<text x="{margin_l + plot_w + 40}" y="{legend_y + 2}" class="legend" fill="{color}">{m_name}</text>')
        legend_y += 28

    svg_lines.append('</svg>')
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def create_confusion_matrices_svg(eval_r: dict, filename: str):
    width, height = 800, 360
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 600; fill: #1e293b; }',
        '  .subhead { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 600; fill: #334155; }',
        '  .cell-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 700; fill: #0f172a; }',
        '  .cell-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #64748b; }',
        '  .axis-lbl { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 500; fill: #475569; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8" />',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Unseen Test Confusion Matrices (Threshold = 0.50, N = 9,999)</text>',
    ]

    models_info = [
        ("E1: Spatial Baseline", eval_r["models"]["E1"]["aggregate"], 60),
        ("E2: Frequency Baseline", eval_r["models"]["E2"]["aggregate"], 310),
        ("E3: Dual-Domain Fusion", eval_r["models"]["E3"]["aggregate"], 560),
    ]

    for model_name, agg, start_x in models_info:
        tp, fp, tn, fn = agg["tp"], agg["fp"], agg["tn"], agg["fn"]
        acc = agg["accuracy"] * 100
        f1 = agg["f1_score"]

        # Subtitle
        svg_lines.append(f'<text x="{start_x + 90}" y="70" text-anchor="middle" class="subhead">{model_name}</text>')
        svg_lines.append(f'<text x="{start_x + 90}" y="88" text-anchor="middle" class="cell-sub">Acc: {acc:.2f}% | F1: {f1:.4f}</text>')

        # Axis labels
        svg_lines.append(f'<text x="{start_x + 50}" y="115" text-anchor="middle" class="axis-lbl">Pred: Nature</text>')
        svg_lines.append(f'<text x="{start_x + 130}" y="115" text-anchor="middle" class="axis-lbl">Pred: AI</text>')
        svg_lines.append(f'<text x="{start_x - 12}" y="160" text-anchor="middle" class="axis-lbl" transform="rotate(-90 {start_x - 12} 160)">True: Nature</text>')
        svg_lines.append(f'<text x="{start_x - 12}" y="240" text-anchor="middle" class="axis-lbl" transform="rotate(-90 {start_x - 12} 240)">True: AI</text>')

        # 2x2 grid
        # TN (Top-Left)
        svg_lines.append(f'<rect x="{start_x + 10}" y="125" width="75" height="75" fill="#dcfce7" stroke="#86efac" stroke-width="1.5" rx="4" />')
        svg_lines.append(f'<text x="{start_x + 47}" y="160" text-anchor="middle" class="cell-text">{tn:,}</text>')
        svg_lines.append(f'<text x="{start_x + 47}" y="178" text-anchor="middle" class="cell-sub">TN ({tn/4999*100:.1f}%)</text>')

        # FP (Top-Right)
        svg_lines.append(f'<rect x="{start_x + 95}" y="125" width="75" height="75" fill="#fee2e2" stroke="#fca5a5" stroke-width="1.5" rx="4" />')
        svg_lines.append(f'<text x="{start_x + 132}" y="160" text-anchor="middle" class="cell-text">{fp:,}</text>')
        svg_lines.append(f'<text x="{start_x + 132}" y="178" text-anchor="middle" class="cell-sub">FP ({fp/4999*100:.1f}%)</text>')

        # FN (Bottom-Left)
        svg_lines.append(f'<rect x="{start_x + 10}" y="205" width="75" height="75" fill="#fee2e2" stroke="#fca5a5" stroke-width="1.5" rx="4" />')
        svg_lines.append(f'<text x="{start_x + 47}" y="240" text-anchor="middle" class="cell-text">{fn:,}</text>')
        svg_lines.append(f'<text x="{start_x + 47}" y="258" text-anchor="middle" class="cell-sub">FN ({fn/5000*100:.1f}%)</text>')

        # TP (Bottom-Right)
        svg_lines.append(f'<rect x="{start_x + 95}" y="205" width="75" height="75" fill="#dcfce7" stroke="#86efac" stroke-width="1.5" rx="4" />')
        svg_lines.append(f'<text x="{start_x + 132}" y="240" text-anchor="middle" class="cell-text">{tp:,}</text>')
        svg_lines.append(f'<text x="{start_x + 132}" y="258" text-anchor="middle" class="cell-sub">TP ({tp/5000*100:.1f}%)</text>')

    svg_lines.append('</svg>')
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def main():
    e1_h, e2_h, e3_h, eval_r = load_data()

    # 1. Training Loss Curves
    create_line_chart_svg(
        title="Training Loss vs Epoch (AdamW Differential LR, Batch Size 16)",
        y_label="BCE Training Loss",
        series_data=[
            ("E1: Spatial Baseline", "#2563eb", [h["train_loss"] for h in e1_h]),
            ("E2: Frequency Baseline", "#059669", [h["train_loss"] for h in e2_h]),
            ("E3: Dual-Domain Model", "#d97706", [h["train_loss"] for h in e3_h]),
        ],
        y_range=(0.10, 0.70),
        filename="training_loss_curves.svg",
    )

    # 2. Validation Loss Curves
    create_line_chart_svg(
        title="Validation Loss vs Epoch (In-Distribution Validation, N = 5,000)",
        y_label="BCE Validation Loss",
        series_data=[
            ("E1: Spatial Baseline", "#2563eb", [h["val_loss"] for h in e1_h]),
            ("E2: Frequency Baseline", "#059669", [h["val_loss"] for h in e2_h]),
            ("E3: Dual-Domain Model", "#d97706", [h["val_loss"] for h in e3_h]),
        ],
        y_range=(0.15, 0.65),
        filename="val_loss_curves.svg",
    )

    # 3. Validation ROC-AUC Curves
    create_line_chart_svg(
        title="Validation ROC-AUC vs Epoch (Checkpoint Selection Metric)",
        y_label="Validation ROC-AUC",
        series_data=[
            ("E1: Spatial Baseline", "#2563eb", [h["val_roc_auc"] for h in e1_h]),
            ("E2: Frequency Baseline", "#059669", [h["val_roc_auc"] for h in e2_h]),
            ("E3: Dual-Domain Model", "#d97706", [h["val_roc_auc"] for h in e3_h]),
        ],
        y_range=(0.70, 1.00),
        filename="val_roc_auc_curves.svg",
    )

    # 4. Validation Accuracy Curves
    create_line_chart_svg(
        title="Validation Accuracy vs Epoch (Threshold = 0.50)",
        y_label="Validation Accuracy",
        series_data=[
            ("E1: Spatial Baseline", "#2563eb", [h["val_accuracy"] for h in e1_h]),
            ("E2: Frequency Baseline", "#059669", [h["val_accuracy"] for h in e2_h]),
            ("E3: Dual-Domain Model", "#d97706", [h["val_accuracy"] for h in e3_h]),
        ],
        y_range=(0.60, 0.98),
        filename="val_accuracy_curves.svg",
    )

    # 5. Generalization & Sub-Split Comparison
    e1_m = eval_r["models"]["E1"]
    e2_m = eval_r["models"]["E2"]
    e3_m = eval_r["models"]["E3"]

    create_bar_chart_svg(
        title="Unseen Test ROC-AUC by Generator Paradigm (Frozen Final Evaluation)",
        categories=["Aggregate Unseen", "BigGAN (GAN)", "Midjourney (Diffusion)"],
        models=[
            ("E1: Spatial", "#2563eb", [e1_m["aggregate"]["roc_auc"], e1_m["by_generator"]["BigGAN"]["roc_auc"], e1_m["by_generator"]["Midjourney"]["roc_auc"]]),
            ("E2: Frequency", "#059669", [e2_m["aggregate"]["roc_auc"], e2_m["by_generator"]["BigGAN"]["roc_auc"], e2_m["by_generator"]["Midjourney"]["roc_auc"]]),
            ("E3: Dual-Domain", "#d97706", [e3_m["aggregate"]["roc_auc"], e3_m["by_generator"]["BigGAN"]["roc_auc"], e3_m["by_generator"]["Midjourney"]["roc_auc"]]),
        ],
        y_range=(0.50, 1.00),
        filename="unseen_generator_roc_auc_comparison.svg",
    )

    # 6. Confusion Matrices Grid
    create_confusion_matrices_svg(eval_r, "confusion_matrices_grid.svg")

    print("[*] All SVG plots generated successfully in experiments/final_results/")


if __name__ == "__main__":
    main()
