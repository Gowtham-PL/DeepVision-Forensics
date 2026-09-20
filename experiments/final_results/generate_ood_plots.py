"""
Generates publication-quality standalone SVG visualization figures for the Extended OOD Generalization Study.
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def create_ood_generalization_svg():
    width, height = 750, 420
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    min_y, max_y = 0.85, 0.92
    models = [
        ("E3 Dual (MinMax)", 0.8851, "#94a3b8"),
        ("E3-Std Dual (Standardize)", 0.8959, "#0284c7"),
        ("E1 Spatial Baseline", 0.8991, "#10b981"),
    ]

    def y_px(val):
        norm = (val - min_y) / (max_y - min_y)
        return margin_t + plot_h - norm * plot_h

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 600; fill: #1e293b; }',
        '  .axis { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; }',
        '  .label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 600; fill: #334155; }',
        '  .bar-val { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 700; fill: #0f172a; }',
        '  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4 4; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8" />',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Overall Unseen-Generator ROC-AUC Comparison (Test Split, N=9,999)</text>',
    ]

    # Grid
    for i in range(8):
        val = min_y + i * 0.01
        y = y_px(val)
        svg_lines.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" y2="{y}" class="grid" />')
        svg_lines.append(f'<text x="{margin_l - 10}" y="{y + 4}" text-anchor="end" class="axis">{val:.2f}</text>')

    # Axis Lines
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<text x="24" y="{margin_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {margin_t + plot_h / 2})" class="label">ROC-AUC</text>')

    # Bars
    num_models = len(models)
    slot_w = plot_w / num_models
    bar_w = 90

    for idx, (m_name, val, color) in enumerate(models):
        bx = margin_l + idx * slot_w + (slot_w - bar_w) / 2
        by = y_px(val)
        bh = (margin_t + plot_h) - by

        svg_lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{color}" rx="4" />')
        svg_lines.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 8:.1f}" text-anchor="middle" class="bar-val">{val:.4f}</text>')
        svg_lines.append(f'<text x="{bx + bar_w/2:.1f}" y="{margin_t + plot_h + 24}" text-anchor="middle" class="label">{m_name}</text>')

    svg_lines.append('</svg>')
    out_file = OUT_DIR / "ood_generalization_comparison.svg"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"[*] Generated {out_file}")


def create_ood_per_generator_svg():
    width, height = 750, 420
    margin_l, margin_r, margin_t, margin_b = 80, 180, 60, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    min_y, max_y = 0.75, 1.00
    categories = ["BigGAN (GAN Paradigm)", "Midjourney (Commercial Diffusion)"]
    models = [
        ("E1 Spatial Baseline", "#10b981", [0.9732, 0.8224]),
        ("E3 Dual (MinMax)", "#94a3b8", [0.9465, 0.8228]),
        ("E3-Std Dual (Standardize)", "#0284c7", [0.9511, 0.8392]),
    ]

    def y_px(val):
        norm = (val - min_y) / (max_y - min_y)
        return margin_t + plot_h - norm * plot_h

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 600; fill: #1e293b; }',
        '  .axis { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; }',
        '  .cat { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 600; fill: #334155; }',
        '  .legend { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 600; }',
        '  .bar-val { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10px; font-weight: 700; fill: #0f172a; }',
        '  .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4 4; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#ffffff" rx="8" />',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Per-Generator Unseen ROC-AUC Comparison</text>',
    ]

    # Grid
    for i in range(6):
        val = min_y + i * 0.05
        y = y_px(val)
        svg_lines.append(f'<line x1="{margin_l}" y1="{y}" x2="{margin_l + plot_w}" y2="{y}" class="grid" />')
        svg_lines.append(f'<text x="{margin_l - 10}" y="{y + 4}" text-anchor="end" class="axis">{val:.2f}</text>')

    # Axis Lines
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" />')
    svg_lines.append(f'<text x="24" y="{margin_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {margin_t + plot_h / 2})" class="cat">ROC-AUC</text>')

    num_cats = len(categories)
    num_models = len(models)
    cat_width = plot_w / num_cats
    bar_width = (cat_width * 0.7) / num_models

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
    out_file = OUT_DIR / "ood_per_generator_comparison.svg"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"[*] Generated {out_file}")


if __name__ == "__main__":
    create_ood_generalization_svg()
    create_ood_per_generator_svg()
