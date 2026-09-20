"""
Dataset Builder and Integrity Verification for Experiment 24 (E24).

Creates:
- experiments/e24_joint_multidomain/manifests/e24_train_manifest.csv
- experiments/e24_joint_multidomain/manifests/e24_dev_manifest.csv

Rules:
- Source data exclusively from E20 and E21 training/dev pools.
- Split strictly by ORIGINAL image ID so all 7 compression variants stay grouped.
- Deterministic seed 42.
- 5 multi-label binary auxiliary attributes:
  1. is_legacy_ai
  2. is_modern_ai
  3. is_smartphone_real
  4. is_camera_real
  5. is_compressed
"""

import os
import sys
import csv
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_DIR = PROJECT_ROOT / "experiments/e24_joint_multidomain/manifests"
DATA_E20 = PROJECT_ROOT / "data/e20_training"
DATA_E21 = PROJECT_ROOT / "data/e21_targeted_data"

LEGACY_GENS = {
    "Stable Diffusion 1.3",
    "Stable Diffusion 1.4",
    "Stable Diffusion 2",
    "Adobe Firefly 1",
    "OpenAI DALL-E 2",
    "GLIDE",
    "OpenAI GLIDE"
}

def load_and_group(manifest_path: Path, is_e20: bool):
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    groups = {}
    for r in records:
        if is_e20:
            orig_id = r["image_id"] if r["derived_from"] == "self" else r["derived_from"]
            r["orig_id"] = orig_id
            r["full_filepath"] = str(Path("data/e20_training") / r["filepath"])
            r["source_pool"] = "E20"
            r["variant_type"] = r["compression_type"]
        else:
            orig_id = r["original_image_id"]
            r["orig_id"] = orig_id
            r["full_filepath"] = str(Path("data/e21_targeted_data") / r["filepath"])
            r["source_pool"] = "E21"
            r["variant_type"] = r.get("variant_type", r["image_id"].split("_")[-1])
        groups.setdefault(orig_id, []).append(r)
    return groups

def sample_origs(groups, filter_fn, n_max=None, rng=None):
    matching = [orig for orig, items in groups.items() if filter_fn(items[0])]
    matching.sort() # Ensure deterministic ordering before shuffle
    if rng:
        rng.shuffle(matching)
    if n_max is not None and len(matching) > n_max:
        return matching[:n_max]
    return matching

def annotate_attributes(records):
    for r in records:
        lbl = r["label"]
        gen = r.get("generator", "")
        dev = r.get("device_family", "")
        var = r.get("variant_type", "")

        is_legacy = 1 if (lbl == "AI" and gen in LEGACY_GENS) else 0
        is_modern = 1 if (lbl == "AI" and gen not in LEGACY_GENS) else 0
        is_smart = 1 if (lbl == "Real" and dev != "Nikon DSLR") else 0
        is_cam = 1 if (lbl == "Real" and dev == "Nikon DSLR") else 0
        is_comp = 1 if (var != "clean") else 0

        r["is_legacy_ai"] = is_legacy
        r["is_modern_ai"] = is_modern
        r["is_smartphone_real"] = is_smart
        r["is_camera_real"] = is_cam
        r["is_compressed"] = is_comp

def build_manifests():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    e20_tr_g = load_and_group(DATA_E20 / "manifests/e20_train_manifest.csv", is_e20=True)
    e20_dv_g = load_and_group(DATA_E20 / "manifests/e20_dev_manifest.csv", is_e20=True)
    e21_tr_g = load_and_group(DATA_E21 / "manifests/e21_train_manifest.csv", is_e20=False)
    e21_dv_g = load_and_group(DATA_E21 / "manifests/e21_dev_manifest.csv", is_e20=False)

    # 1. Build E24 Train
    # A. All E21 Train (468 originals = 181 legacy AI + 53 modern AI + 234 smartphone real)
    train_origs_e21 = list(e21_tr_g.keys())

    # B. Selected E20 Train:
    e20_flux = sample_origs(e20_tr_g, lambda r: r["label"] == "AI" and r["generator"] == "FLUX.1 [dev]", 60, rng)
    e20_sd = sample_origs(e20_tr_g, lambda r: r["label"] == "AI" and r["generator"] == "Stable Diffusion", 60, rng)
    e20_gem = sample_origs(e20_tr_g, lambda r: r["label"] == "AI" and r["generator"] == "Gemini / Imagen", None, rng)
    e20_dalle = sample_origs(e20_tr_g, lambda r: r["label"] == "AI" and r["generator"] == "DALL-E", None, rng)
    e20_mj = sample_origs(e20_tr_g, lambda r: r["label"] == "AI" and r["generator"] == "Midjourney", None, rng)

    e20_dslr = sample_origs(e20_tr_g, lambda r: r["label"] == "Real" and r["device_family"] == "Nikon DSLR", 100, rng)
    e20_pixel = sample_origs(e20_tr_g, lambda r: r["label"] == "Real" and r["device_family"] == "Google Pixel", None, rng)
    e20_vivo = sample_origs(e20_tr_g, lambda r: r["label"] == "Real" and r["device_family"] == "Vivo", 30, rng)
    e20_samsung = sample_origs(e20_tr_g, lambda r: r["label"] == "Real" and r["device_family"] == "Samsung Galaxy", 30, rng)
    e20_iphone = sample_origs(e20_tr_g, lambda r: r["label"] == "Real" and r["device_family"] == "Apple iPhone", 30, rng)

    train_origs_e20 = e20_flux + e20_sd + e20_gem + e20_dalle + e20_mj + e20_dslr + e20_pixel + e20_vivo + e20_samsung + e20_iphone

    train_rows = []
    for oid in train_origs_e21:
        train_rows.extend(e21_tr_g[oid])
    for oid in train_origs_e20:
        train_rows.extend(e20_tr_g[oid])

    annotate_attributes(train_rows)

    # 2. Build E24 Dev
    # A. All E21 Dev (116 originals = 58 AI + 58 Real)
    dev_origs_e21 = list(e21_dv_g.keys())

    # B. Selected E20 Dev:
    e20_dv_flux = sample_origs(e20_dv_g, lambda r: r["label"] == "AI" and r["generator"] == "FLUX.1 [dev]", 20, rng)
    e20_dv_sd = sample_origs(e20_dv_g, lambda r: r["label"] == "AI" and r["generator"] == "Stable Diffusion", 15, rng)
    e20_dv_gem = sample_origs(e20_dv_g, lambda r: r["label"] == "AI" and r["generator"] == "Gemini / Imagen", None, rng)
    e20_dv_dalle = sample_origs(e20_dv_g, lambda r: r["label"] == "AI" and r["generator"] == "DALL-E", None, rng)
    e20_dv_mj = sample_origs(e20_dv_g, lambda r: r["label"] == "AI" and r["generator"] == "Midjourney", None, rng)

    e20_dv_dslr = sample_origs(e20_dv_g, lambda r: r["label"] == "Real" and r["device_family"] == "Nikon DSLR", 15, rng)
    e20_dv_pixel = sample_origs(e20_dv_g, lambda r: r["label"] == "Real" and r["device_family"] == "Google Pixel", None, rng)
    e20_dv_vivo = sample_origs(e20_dv_g, lambda r: r["label"] == "Real" and r["device_family"] == "Vivo", 10, rng)
    e20_dv_samsung = sample_origs(e20_dv_g, lambda r: r["label"] == "Real" and r["device_family"] == "Samsung Galaxy", 10, rng)
    e20_dv_iphone = sample_origs(e20_dv_g, lambda r: r["label"] == "Real" and r["device_family"] == "Apple iPhone", 10, rng)

    dev_origs_e20 = e20_dv_flux + e20_dv_sd + e20_dv_gem + e20_dv_dalle + e20_dv_mj + e20_dv_dslr + e20_dv_pixel + e20_dv_vivo + e20_dv_samsung + e20_dv_iphone

    dev_rows = []
    for oid in dev_origs_e21:
        dev_rows.extend(e21_dv_g[oid])
    for oid in dev_origs_e20:
        dev_rows.extend(e20_dv_g[oid])

    annotate_attributes(dev_rows)

    # Save CSVs
    train_csv = MANIFEST_DIR / "e24_train_manifest.csv"
    dev_csv = MANIFEST_DIR / "e24_dev_manifest.csv"

    fieldnames = [
        "image_id", "orig_id", "full_filepath", "label", "source_pool",
        "generator", "device_family", "variant_type",
        "is_legacy_ai", "is_modern_ai", "is_smartphone_real", "is_camera_real", "is_compressed"
    ]

    for csv_path, rows in [(train_csv, train_rows), (dev_csv, dev_rows)]:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    print("==================================================")
    print("E24 DATASET CREATION SUMMARY & DISTRIBUTION TABLE")
    print("==================================================")
    print(f"Train Manifest: {train_csv} ({len(train_rows)} images, {len(train_origs_e21) + len(train_origs_e20)} originals)")
    print(f"Dev Manifest:   {dev_csv} ({len(dev_rows)} images, {len(dev_origs_e21) + len(dev_origs_e20)} originals)")

    for name, rows in [("TRAINING SET (E24 Train)", train_rows), ("DEVELOPMENT SET (E24 Dev)", dev_rows)]:
        print(f"\n--- {name} (N={len(rows)}) ---")
        n_real = sum(1 for r in rows if r["label"] == "Real")
        n_ai = sum(1 for r in rows if r["label"] == "AI")
        print(f"Binary Class Balance: Real = {n_real} ({n_real/len(rows)*100:.1f}%), AI = {n_ai} ({n_ai/len(rows)*100:.1f}%)")
        
        print("\nMulti-Label Auxiliary Attribute Distribution:")
        print(f"{'Attribute':<25} | {'Real':<8} | {'AI':<8} | {'Total':<8}")
        print("-" * 55)
        for attr in ["is_legacy_ai", "is_modern_ai", "is_smartphone_real", "is_camera_real", "is_compressed"]:
            r_c = sum(1 for r in rows if r["label"] == "Real" and r[attr] == 1)
            a_c = sum(1 for r in rows if r["label"] == "AI" and r[attr] == 1)
            print(f"{attr:<25} | {r_c:<8} | {a_c:<8} | {r_c + a_c:<8}")

        print("\nAI Generator Breakdown:")
        gen_counts = Counter(r["generator"] for r in rows if r["label"] == "AI")
        for g, c in sorted(gen_counts.items(), key=lambda x: -x[1]):
            print(f"  - {g:<28}: {c:>5} images ({c//7:>3} originals)")

        print("\nReal Device / Source Breakdown:")
        dev_counts = Counter(r["device_family"] for r in rows if r["label"] == "Real")
        for d, c in sorted(dev_counts.items(), key=lambda x: -x[1]):
            print(f"  - {d:<28}: {c:>5} images ({c//7:>3} originals)")

        print("\nCompression Variant Breakdown:")
        var_counts = Counter(r["variant_type"] for r in rows)
        for v, c in sorted(var_counts.items(), key=lambda x: -x[1]):
            print(f"  - {v:<28}: {c:>5} images")

        # Multi-attribute combinations count
        overlap_counts = Counter(
            (r["is_legacy_ai"], r["is_modern_ai"], r["is_smartphone_real"], r["is_camera_real"], r["is_compressed"])
            for r in rows
        )
        print(f"\nUnique Multi-Label Attribute Configurations: {len(overlap_counts)}")

if __name__ == "__main__":
    build_manifests()
