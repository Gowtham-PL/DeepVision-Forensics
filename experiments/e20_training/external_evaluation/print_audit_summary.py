import json

d = json.load(open("experiments/e20_training/external_evaluation/e20_failure_audit_data.json", encoding="utf-8"))

print("=== ANALYSIS 1: Transitions Summary ===")
for r in d["analysis1_transitions"]:
    ds = r["dataset"]
    n = r["n"]
    t = r["transitions"]
    print(f"{ds} (N={n}):")
    print(f"  Transitions: {t}")
    print(f"  Real Transitions: {r['real_transitions']}")
    print(f"  AI Transitions: {r['ai_transitions']}")
    print(f"  Regressions by Gen: {r['regressions_by_generator']}")
    print(f"  Regressions by Dev: {r['regressions_by_device']}")
    print(f"  Regressions by Deg: {r['regressions_by_degradation']}")

print("\n=== ANALYSIS 2: Generator Failure Details ===")
for r in d["analysis2_generators"]:
    ds = r["dataset"]
    gen = r["generator"]
    n = r["n"]
    rec1, rec2 = r["e6c_recall"], r["e20_recall"]
    p1, p2 = r["e6c_mean_prob"], r["e20_mean_prob"]
    auc1, auc2 = r["e6c_auc_vs_real"], r["e20_auc_vs_real"]
    print(f"{ds:25s} | {gen:35s} (N={n:2d}) | Rec: {rec1:.2f}->{rec2:.2f} | Prob: {p1:.3f}->{p2:.3f} | AUC: {auc1:.3f}->{auc2:.3f}")

print("\n=== ANALYSIS 3: Real Image False Positives Details ===")
for r in d["analysis3_real_fpr"]:
    ds = r["dataset"]
    tot = r["total_real"]
    fp = r["e20_fp_count"]
    fpr = r["e20_fpr"] * 100
    fp_devs = r["fp_devices"]
    print(f"{ds:25s} | Real N={tot:3d} | FP={fp:2d} ({fpr:5.1f}%) | FP Devs: {fp_devs}")
    print(f"  Laplacian Var: All={r['mean_laplacian_var_all']:6.1f}, FPs={r['mean_laplacian_var_fps']:6.1f}, TNs={r['mean_laplacian_var_tns']:6.1f}")
    print(f"  Edge Density:  All={r['mean_edge_density_all']:6.3f}, FPs={r['mean_edge_density_fps']:6.3f}, TNs={r['mean_edge_density_tns']:6.3f}")
    print(f"  High Freq En:  All={r['mean_high_freq_all']:6.4f}, FPs={r['mean_high_freq_fps']:6.4f}, TNs={r['mean_high_freq_tns']:6.4f}")

print("\n=== ANALYSIS 5: Probability Distributions ===")
for r in d["analysis5_distributions"]:
    ds = r["dataset"]
    e6c_r = r["e6c_real"]
    e6c_a = r["e6c_ai"]
    e20_r = r["e20_real"]
    e20_a = r["e20_ai"]
    print(f"{ds}:")
    print(f"  Real Prob: E6-C Mean={e6c_r['mean']:.3f}, Med={e6c_r['median']:.3f}, Std={e6c_r['std']:.3f}, IQR=[{e6c_r['q25']:.3f}, {e6c_r['q75']:.3f}]")
    print(f"             E20  Mean={e20_r['mean']:.3f}, Med={e20_r['median']:.3f}, Std={e20_r['std']:.3f}, IQR=[{e20_r['q25']:.3f}, {e20_r['q75']:.3f}]")
    print(f"  AI Prob:   E6-C Mean={e6c_a['mean']:.3f}, Med={e6c_a['median']:.3f}, Std={e6c_a['std']:.3f}, IQR=[{e6c_a['q25']:.3f}, {e6c_a['q75']:.3f}]")
    print(f"             E20  Mean={e20_a['mean']:.3f}, Med={e20_a['median']:.3f}, Std={e20_a['std']:.3f}, IQR=[{e20_a['q25']:.3f}, {e20_a['q75']:.3f}]")
    print(f"  Margin:    E6-C = {r['e6c_margin']:+.3f}, E20 = {r['e20_margin']:+.3f}")
