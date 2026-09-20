"""
Stratified split and cryptographic freezing of E14 External Pool.
Splits data/final_external_pool into:
- FINAL_TRAIN_POOL: exactly 200 images (100 Real, 100 AI)
- FINAL_TEST_POOL:  exactly 200 images (100 Real, 100 AI)
using a fixed random seed (seed=42) and stratified across all 8 diffusion generators.
Creates cryptographic manifest and JSON seal for the frozen final test set.
"""

import os
import csv
import random
import hashlib
from collections import defaultdict

def compute_file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest().lower()

def main():
    pool_csv = 'experiments/final_external_test/manifests/external_pool_manifest.csv'
    if not os.path.exists(pool_csv):
        print(f"Error: {pool_csv} does not exist.")
        return

    records = []
    with open(pool_csv, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))

    print(f"Loaded {len(records)} external pool records.")

    # Fixed seed for perfect reproducibility
    SEED = 42
    random.seed(SEED)

    # Separate Real and AI
    real_records = [r for r in records if r['label'] == 'Real']
    ai_records = [r for r in records if r['label'] == 'AI']

    print(f"Real records: {len(real_records)} | AI records: {len(ai_records)}")

    # 1. Stratify Real (RAISE-1k): shuffle and select 100 train, 100 test
    random.shuffle(real_records)
    train_real = real_records[:100]
    test_real = real_records[100:200]
    reservoir_real = real_records[200:]

    # 2. Stratify AI across generators:
    # Target exactly 100 train and 100 test
    ai_by_gen = defaultdict(list)
    for r in ai_records:
        ai_by_gen[r['generator']].append(r)

    # Sort keys for deterministic behavior
    gen_keys = sorted(ai_by_gen.keys())
    for g in gen_keys:
        random.shuffle(ai_by_gen[g])

    # Generator allocation to total 100:
    # dalle2 (30): 14 train, 14 test
    # dalle3 (13): 6 train, 6 test
    # firefly (30): 14 train, 14 test
    # glide (28): 13 train, 13 test
    # midjourney-v5 (26): 12 train, 12 test
    # stable-diffusion-1-3 (31): 14 train, 14 test
    # stable-diffusion-1-4 (26): 12 train, 12 test
    # stable-diffusion-2 (33): 15 train, 15 test
    # Total = 14 + 6 + 14 + 13 + 12 + 14 + 12 + 15 = 100 train, 100 test!
    allocations = {
        'dalle2': 14,
        'dalle3': 6,
        'firefly': 14,
        'glide': 13,
        'midjourney-v5': 12,
        'stable-diffusion-1-3': 14,
        'stable-diffusion-1-4': 12,
        'stable-diffusion-2': 15
    }

    train_ai = []
    test_ai = []
    for g in gen_keys:
        n = allocations[g]
        train_ai.extend(ai_by_gen[g][:n])
        test_ai.extend(ai_by_gen[g][n:2*n])

    assert len(train_ai) == 100, f"Expected 100 AI in train, got {len(train_ai)}"
    assert len(test_ai) == 100, f"Expected 100 AI in test, got {len(test_ai)}"

    train_records = train_real + train_ai
    test_records = test_real + test_ai

    # Final shuffle within splits using seed
    random.shuffle(train_records)
    random.shuffle(test_records)

    print("\n--- Split Results ---")
    print(f"FINAL_TRAIN_POOL: {len(train_records)} total (Real: 100, AI: 100)")
    print(f"FINAL_TEST_POOL:  {len(test_records)} total (Real: 100, AI: 100)")

    # Integrity check: zero overlap
    train_ids = set(r['image_id'] for r in train_records)
    test_ids = set(r['image_id'] for r in test_records)
    assert len(train_ids.intersection(test_ids)) == 0, "FATAL ERROR: Overlap between train and test pools!"
    print("Zero overlap between train and test verified successfully.")

    # Write manifests
    fieldnames = list(records[0].keys()) + ['split']
    for r in train_records: r['split'] = 'FINAL_TRAIN_POOL'
    for r in test_records: r['split'] = 'FINAL_TEST_POOL'

    manifest_train_path = 'experiments/final_external_test/manifests/final_train_manifest.csv'
    manifest_test_path = 'experiments/final_external_test/manifests/final_test_manifest.csv'

    with open(manifest_train_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in train_records: writer.writerow(r)
    print(f"Saved train manifest to {manifest_train_path}")

    with open(manifest_test_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in test_records: writer.writerow(r)
    print(f"Saved test manifest to {manifest_test_path}")

    # Cryptographic sealing of final test set
    with open(manifest_test_path, 'rb') as f:
        manifest_hash = hashlib.sha256(f.read()).hexdigest()

    img_hasher = hashlib.sha256()
    for r in sorted(test_records, key=lambda x: x['image_id']):
        img_hasher.update(r['sha256'].encode('ascii'))
    test_set_seal = img_hasher.hexdigest()

    import json
    seal_file = 'experiments/final_external_test/manifests/FINAL_TEST_SET_SEAL.json'
    seal_data = {
        "status": "SEALED_FROZEN",
        "random_seed": SEED,
        "n_total": len(test_records),
        "n_real": sum(1 for r in test_records if r['label'] == 'Real'),
        "n_ai": sum(1 for r in test_records if r['label'] == 'AI'),
        "generator_distribution": {
            gen: sum(1 for r in test_records if r['generator'] == gen)
            for gen in sorted(set(r['generator'] for r in test_records))
        },
        "test_manifest_sha256": manifest_hash,
        "aggregate_images_sha256": test_set_seal,
        "isolation_protocol": "FINAL_TEST_POOL is strictly quarantined for zero-shot final evaluation only. No model training, tuning, or threshold selection permitted."
    }
    with open(seal_file, 'w', encoding='utf-8') as f:
        json.dump(seal_data, f, indent=2)

    print(f"\nFINAL TEST SET SEAL GENERATED:")
    print(f"  Test Manifest SHA256: {manifest_hash}")
    print(f"  Aggregate Images SHA256: {test_set_seal}")
    print(f"  Seal saved to {seal_file}")

if __name__ == '__main__':
    main()
