"""
Generate realistic compression derivatives for E14:
1. Training Derivatives (from FINAL_TRAIN_POOL ONLY) -> data/final_external_pool/compression_train/
   - Clean
   - WhatsApp-tier JPEG (long edge 1280, Q75, 4:2:0 chroma)
   - Double JPEG (long edge 1400, Q80 -> decompress -> Q70, 4:2:0 chroma)
   - Moderate resize/downsample (long edge 960) + JPEG Q75
2. Robustness Test Derivatives (from FINAL_TEST_POOL ONLY) -> data/final_external_pool/final_test_degraded/
   - Controlled WhatsApp-tier JPEG (long edge 1280, Q75, 4:2:0)
   - Resize (long edge 1080) + JPEG Q80
   - Double JPEG (long edge 1280, Q80 -> Q70)
   Keeps original FINAL_TEST_POOL completely untouched.
"""

import os
import io
import csv
import hashlib
from PIL import Image

def compute_file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest().lower()

def resize_long_edge(img, max_long_edge):
    w, h = img.size
    if max(w, h) <= max_long_edge:
        return img.copy()
    if w >= h:
        new_w = max_long_edge
        new_h = int(round(h * (max_long_edge / w)))
    else:
        new_h = max_long_edge
        new_w = int(round(w * (max_long_edge / h)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def apply_whatsapp_jpeg(img, long_edge=1280, quality=75):
    resized = resize_long_edge(img, long_edge)
    buf = io.BytesIO()
    # In PIL, subsampling="4:2:0" or subsampling=2 produces standard 4:2:0 chroma
    resized.convert('RGB').save(buf, format='JPEG', quality=quality, subsampling=2)
    buf.seek(0)
    return Image.open(buf), buf.getvalue()

def apply_double_jpeg(img, long_edge=1400, q1=80, q2=70):
    resized = resize_long_edge(img, long_edge)
    buf1 = io.BytesIO()
    resized.convert('RGB').save(buf1, format='JPEG', quality=q1, subsampling=2)
    buf1.seek(0)
    intermediate = Image.open(buf1)
    
    buf2 = io.BytesIO()
    intermediate.save(buf2, format='JPEG', quality=q2, subsampling=2)
    buf2.seek(0)
    return Image.open(buf2), buf2.getvalue()

def apply_resize_jpeg(img, long_edge=960, quality=75):
    resized = resize_long_edge(img, long_edge)
    buf = io.BytesIO()
    resized.convert('RGB').save(buf, format='JPEG', quality=quality, subsampling=2)
    buf.seek(0)
    return Image.open(buf), buf.getvalue()

def main():
    train_manifest_csv = 'experiments/final_external_test/manifests/final_train_manifest.csv'
    test_manifest_csv = 'experiments/final_external_test/manifests/final_test_manifest.csv'
    
    if not os.path.exists(train_manifest_csv) or not os.path.exists(test_manifest_csv):
        print("Error: Train/test manifests not found. Run split_and_freeze_pool.py first.")
        return

    comp_train_dir = 'data/final_external_pool/compression_train'
    test_degraded_dir = 'data/final_external_pool/final_test_degraded'
    os.makedirs(comp_train_dir, exist_ok=True)
    os.makedirs(test_degraded_dir, exist_ok=True)

    # 1. GENERATE TRAINING COMPRESSION DERIVATIVES (FROM FINAL_TRAIN_POOL ONLY)
    print("\n--- Generating Training Compression Derivatives (Train Pool ONLY) ---")
    train_records = []
    with open(train_manifest_csv, 'r', encoding='utf-8') as f:
        train_records = list(csv.DictReader(f))

    comp_train_manifest_records = []
    for i, r in enumerate(train_records):
        img_id = r['image_id']
        src_path = r['filepath']
        lbl = r['label']
        gen = r['generator']
        
        with Image.open(src_path) as orig_img:
            # Variant A: Clean (copy as PNG or reference)
            clean_fn = f"{img_id}_clean.png"
            clean_out = os.path.join(comp_train_dir, clean_fn).replace('\\', '/')
            orig_img.save(clean_out, format="PNG")
            comp_train_manifest_records.append({
                'variant_id': clean_fn[:-4],
                'original_image_id': img_id,
                'filepath': clean_out,
                'label': lbl,
                'generator': gen,
                'transformation': 'Clean (uncompressed PNG)',
                'jpeg_quality': 'None',
                'width': orig_img.width,
                'height': orig_img.height,
                'sha256': compute_file_sha256(clean_out)
            })

            # Variant B: WhatsApp-tier JPEG (long edge 1280, Q75, 4:2:0)
            wa_img, wa_bytes = apply_whatsapp_jpeg(orig_img, long_edge=1280, quality=75)
            wa_fn = f"{img_id}_whatsapp_q75.jpg"
            wa_out = os.path.join(comp_train_dir, wa_fn).replace('\\', '/')
            with open(wa_out, 'wb') as f_out: f_out.write(wa_bytes)
            comp_train_manifest_records.append({
                'variant_id': wa_fn[:-4],
                'original_image_id': img_id,
                'filepath': wa_out,
                'label': lbl,
                'generator': gen,
                'transformation': 'WhatsApp-tier JPEG (L1280, Q75, 4:2:0)',
                'jpeg_quality': '75',
                'width': wa_img.width,
                'height': wa_img.height,
                'sha256': compute_file_sha256(wa_out)
            })

            # Variant C: Double JPEG (long edge 1400, Q80 -> Q70, 4:2:0)
            dj_img, dj_bytes = apply_double_jpeg(orig_img, long_edge=1400, q1=80, q2=70)
            dj_fn = f"{img_id}_double_jpeg.jpg"
            dj_out = os.path.join(comp_train_dir, dj_fn).replace('\\', '/')
            with open(dj_out, 'wb') as f_out: f_out.write(dj_bytes)
            comp_train_manifest_records.append({
                'variant_id': dj_fn[:-4],
                'original_image_id': img_id,
                'filepath': dj_out,
                'label': lbl,
                'generator': gen,
                'transformation': 'Double JPEG (L1400, Q80->Q70, 4:2:0)',
                'jpeg_quality': '80_70',
                'width': dj_img.width,
                'height': dj_img.height,
                'sha256': compute_file_sha256(dj_out)
            })

            # Variant D: Moderate Downsample + JPEG (long edge 960, Q75, 4:2:0)
            md_img, md_bytes = apply_resize_jpeg(orig_img, long_edge=960, quality=75)
            md_fn = f"{img_id}_downsample_q75.jpg"
            md_out = os.path.join(comp_train_dir, md_fn).replace('\\', '/')
            with open(md_out, 'wb') as f_out: f_out.write(md_bytes)
            comp_train_manifest_records.append({
                'variant_id': md_fn[:-4],
                'original_image_id': img_id,
                'filepath': md_out,
                'label': lbl,
                'generator': gen,
                'transformation': 'Downsample (L960) + JPEG Q75 (4:2:0)',
                'jpeg_quality': '75',
                'width': md_img.width,
                'height': md_img.height,
                'sha256': compute_file_sha256(md_out)
            })

        if (i + 1) % 50 == 0:
            print(f"Train derivatives: {i+1}/{len(train_records)} images generated (total variants: {len(comp_train_manifest_records)})")

    # Save training compression manifest
    train_comp_csv = 'experiments/final_external_test/manifests/final_train_compression_manifest.csv'
    with open(train_comp_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_train_manifest_records[0].keys()))
        writer.writeheader()
        for r in comp_train_manifest_records: writer.writerow(r)
    print(f"Saved training compression manifest ({len(comp_train_manifest_records)} variants) to {train_comp_csv}")

    # 2. GENERATE ROBUSTNESS TEST DERIVATIVES (FROM FINAL_TEST_POOL ONLY)
    print("\n--- Generating Robustness Evaluation Derivatives (Test Pool ONLY) ---")
    test_records = []
    with open(test_manifest_csv, 'r', encoding='utf-8') as f:
        test_records = list(csv.DictReader(f))

    test_deg_manifest_records = []
    for i, r in enumerate(test_records):
        img_id = r['image_id']
        src_path = r['filepath']
        lbl = r['label']
        gen = r['generator']
        
        with Image.open(src_path) as orig_img:
            # Controlled Variant 1: WhatsApp-tier JPEG (L1280, Q75, 4:2:0)
            wa_img, wa_bytes = apply_whatsapp_jpeg(orig_img, long_edge=1280, quality=75)
            wa_fn = f"{img_id}_test_whatsapp_q75.jpg"
            wa_out = os.path.join(test_degraded_dir, wa_fn).replace('\\', '/')
            with open(wa_out, 'wb') as f_out: f_out.write(wa_bytes)
            test_deg_manifest_records.append({
                'variant_id': wa_fn[:-4],
                'original_image_id': img_id,
                'filepath': wa_out,
                'label': lbl,
                'generator': gen,
                'degradation_type': 'WhatsApp-tier JPEG (L1280, Q75, 4:2:0)',
                'jpeg_quality': '75',
                'width': wa_img.width,
                'height': wa_img.height,
                'sha256': compute_file_sha256(wa_out)
            })

            # Controlled Variant 2: Resize (L1080) + JPEG Q80
            md_img, md_bytes = apply_resize_jpeg(orig_img, long_edge=1080, quality=80)
            md_fn = f"{img_id}_test_resize_q80.jpg"
            md_out = os.path.join(test_degraded_dir, md_fn).replace('\\', '/')
            with open(md_out, 'wb') as f_out: f_out.write(md_bytes)
            test_deg_manifest_records.append({
                'variant_id': md_fn[:-4],
                'original_image_id': img_id,
                'filepath': md_out,
                'label': lbl,
                'generator': gen,
                'degradation_type': 'Resize (L1080) + JPEG Q80 (4:2:0)',
                'jpeg_quality': '80',
                'width': md_img.width,
                'height': md_img.height,
                'sha256': compute_file_sha256(md_out)
            })

            # Controlled Variant 3: Double JPEG (L1280, Q80 -> Q70)
            dj_img, dj_bytes = apply_double_jpeg(orig_img, long_edge=1280, q1=80, q2=70)
            dj_fn = f"{img_id}_test_double_jpeg.jpg"
            dj_out = os.path.join(test_degraded_dir, dj_fn).replace('\\', '/')
            with open(dj_out, 'wb') as f_out: f_out.write(dj_bytes)
            test_deg_manifest_records.append({
                'variant_id': dj_fn[:-4],
                'original_image_id': img_id,
                'filepath': dj_out,
                'label': lbl,
                'generator': gen,
                'degradation_type': 'Double JPEG (L1280, Q80->Q70, 4:2:0)',
                'jpeg_quality': '80_70',
                'width': dj_img.width,
                'height': dj_img.height,
                'sha256': compute_file_sha256(dj_out)
            })

        if (i + 1) % 50 == 0:
            print(f"Test degraded: {i+1}/{len(test_records)} images generated (total variants: {len(test_deg_manifest_records)})")

    # Save test degraded manifest
    test_deg_csv = 'experiments/final_external_test/manifests/final_test_degraded_manifest.csv'
    with open(test_deg_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(test_deg_manifest_records[0].keys()))
        writer.writeheader()
        for r in test_deg_manifest_records: writer.writerow(r)
    print(f"Saved test degraded manifest ({len(test_deg_manifest_records)} variants) to {test_deg_csv}")

    print("\nAll derivatives generated successfully with strict isolation preserved!")

if __name__ == '__main__':
    main()
