"""
audit.py — Summarise the manifest CSV produced by prepare.py.

Run:
    python -m ml.data.audit
"""

import csv
from collections import Counter

from ml.data import config


def run_audit() -> None:
    if not config.MANIFEST_PATH.exists():
        print(f"Manifest not found at {config.MANIFEST_PATH}. Run prepare.py first.")
        return

    total    = 0
    nature   = 0
    ai       = 0
    dups     = 0

    splits     = Counter()
    formats    = Counter()
    generators = Counter()
    orig_splits = Counter()
    # split × label breakdown for balance check
    split_label: Counter = Counter()

    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            label = int(row["label"])

            if label == config.LABEL_NATURE:
                nature += 1
            else:
                ai += 1

            splits[row["split"]]           += 1
            formats[row["format"]]         += 1
            generators[row["generator"]]   += 1
            orig_splits[row["orig_split"]] += 1
            split_label[(row["split"], row["class_name"])] += 1

            if row.get("is_cross_gen_dup", "False") == "True":
                dups += 1

    print("=" * 60)
    print("DATASET AUDIT")
    print(f"Manifest : {config.MANIFEST_PATH}")
    print("=" * 60)

    print(f"\nTotal images   : {total}")
    print(f"  nature (0)   : {nature}")
    print(f"  ai     (1)   : {ai}")
    print(f"  Balanced     : {'YES' if nature == ai else 'NO — IMBALANCE DETECTED'}")

    print("\nLogical splits:")
    for k in ("train", "val", "test"):
        print(f"  {k:<6}: {splits[k]}")

    print("\nSplit × label breakdown:")
    for split in ("train", "val", "test"):
        n_nat = split_label[(split, "nature")]
        n_ai  = split_label[(split, "ai")]
        bal   = "balanced" if n_nat == n_ai else "IMBALANCED"
        print(f"  {split:<6}: nature={n_nat}  ai={n_ai}  [{bal}]")

    print("\nOriginal dataset splits (provenance):")
    for k in ("train", "val"):
        print(f"  {k:<6}: {orig_splits[k]}")

    print("\nGenerators:")
    for gen in sorted(generators):
        print(f"  {gen:<14}: {generators[gen]}")

    print("\nFormats:")
    for k, v in sorted(formats.items()):
        print(f"  {k}: {v}")

    print(f"\nCross-generator duplicates flagged: {dups}")

    # Leakage check — no train/val generator should appear in test and vice versa
    print("\nLeakage check:")
    train_val_gens: set[str] = set()
    test_gens:      set[str] = set()
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] in ("train", "val"):
                train_val_gens.add(row["generator"])
            else:
                test_gens.add(row["generator"])
    overlap = train_val_gens & test_gens
    if overlap:
        print(f"  [FAIL] Generators in both train/val and test: {overlap}")
    else:
        print("  [PASS] No generator appears in both train/val and test.")


if __name__ == "__main__":
    run_audit()
