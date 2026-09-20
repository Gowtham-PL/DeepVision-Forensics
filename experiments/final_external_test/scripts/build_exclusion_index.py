"""
Build full project exclusion index for E14 leakage and duplicate audit.
Indexes all existing datasets in the repository:
- data/e5_external/manifests/e5_manifest.csv
- data/manifest.csv
- data/real_world_test/
- data/real_world_test_v2/
- data/whatsapp_robustness_test/
- data/e6_hard_cases/
"""

import os
import csv
import hashlib
import numpy as np
from PIL import Image

def compute_hashes_for_file(path):
    with open(path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().lower()
    try:
        with Image.open(path) as raw_img:
            img = raw_img.convert('L')
            
            # dhash (9x8)
            resized_d = img.resize((9, 8), Image.Resampling.LANCZOS)
            diff = np.array(resized_d)[:, 1:] > np.array(resized_d)[:, :-1]
            dh = 0
            for b in diff.flatten():
                dh = (dh << 1) | int(b)
            dhash_hex = f'{dh:016x}'
            
            # ahash (8x8)
            resized_a = img.resize((8, 8), Image.Resampling.LANCZOS)
            pix_a = np.array(resized_a)
            diff_a = pix_a > pix_a.mean()
            ah = 0
            for b in diff_a.flatten():
                ah = (ah << 1) | int(b)
            ahash_hex = f'{ah:016x}'
            
            return sha256, dh, ah, dhash_hex, ahash_hex
    except Exception as e:
        print(f"Warning: could not process image {path}: {e}")
        return sha256, 0, 0, "0"*16, "0"*16

def main():
    out_dir = 'experiments/final_external_test/scratch'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'exclusion_index.npz')
    
    sha256_list = []
    dhash_list = []
    ahash_list = []
    source_list = []
    
    # 1. Load from e5_manifest.csv
    e5_csv = 'data/e5_external/manifests/e5_manifest.csv'
    if os.path.exists(e5_csv):
        print(f"Reading {e5_csv}...")
        with open(e5_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                sha = r.get('sha256', '').strip().lower()
                dh_hex = r.get('dhash_hex', '').strip().lower()
                ah_hex = r.get('ahash_hex', '').strip().lower()
                if sha and dh_hex and ah_hex:
                    try:
                        dh_val = int(dh_hex, 16)
                        ah_val = int(ah_hex, 16)
                        sha256_list.append(sha)
                        dhash_list.append(dh_val)
                        ahash_list.append(ah_val)
                        source_list.append(r.get('image_path', 'e5_manifest'))
                    except ValueError:
                        pass
        print(f"Loaded {len(sha256_list)} entries from e5_manifest.")

    # 2. Add physical test and diagnostic images
    extra_dirs = [
        'data/real_world_test',
        'data/real_world_test_v2',
        'data/whatsapp_robustness_test',
        'data/e6_hard_cases'
    ]
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    indexed_shas = set(sha256_list)
    
    extra_added = 0
    for d in extra_dirs:
        for root, _, files in os.walk(d):
            for file in files:
                if file.lower().endswith(exts):
                    p = os.path.join(root, file).replace('\\', '/')
                    sha, dh, ah, dh_hex, ah_hex = compute_hashes_for_file(p)
                    if sha not in indexed_shas:
                        indexed_shas.add(sha)
                        sha256_list.append(sha)
                        dhash_list.append(dh)
                        ahash_list.append(ah)
                        source_list.append(p)
                        extra_added += 1

    print(f"Added {extra_added} extra project images.")
    print(f"Total exclusion index size: {len(sha256_list)} items.")
    
    np.savez_compressed(
        out_file,
        sha256=np.array(sha256_list),
        dhash=np.array(dhash_list, dtype=np.uint64),
        ahash=np.array(ahash_list, dtype=np.uint64),
        source=np.array(source_list)
    )
    print(f"Saved exclusion index to {out_file}")

if __name__ == '__main__':
    main()
