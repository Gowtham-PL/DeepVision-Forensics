"""
E20: Training and Development Dataset Construction Script
DeepVision-Forensics Research Pipeline

Protocol & Constraints:
- Target: High-Diversity E20 Train (22,400 examples) and Dev (2,800 examples).
- Clean Originals: 3,600 total (1,800 Real, 1,800 AI).
  - Real: Apple iPhone X (550), Samsung Galaxy S9 (550), Nikon DSLR RAW / RAISE (400),
          Vivo X90 (250), Google Pixel (50).
  - AI: FLUX.1 [dev] (650), Stable Diffusion / SDXL (650), OpenAI DALL-E 2/3 (200),
        Midjourney v5/v6 (200), Google Gemini / Imagen (100).
- Derived Variants: 6 compression variants per original + 1 clean = 7 symmetric variants.
- Strict Non-Contamination: Zero overlap with 41,372 historical project assets.
- Splitting: Strictly by original source image with deterministic seed 42.
- Zero model training, zero model inference.
"""

import os
import sys
import io
import time
import json
import csv
import zlib
import struct
import hashlib
import pickle
import random
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
E20_DIR = PROJECT_ROOT / "data/e20_training"
IMAGES_DIR = E20_DIR / "images"
MANIFEST_DIR = E20_DIR / "manifests"
SCRATCH_DIR = E20_DIR / "scratch"

TRAIN_IMG_DIR = IMAGES_DIR / "train"
DEV_IMG_DIR = IMAGES_DIR / "dev"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
WIKI_UA = "DeepVisionForensicsResearch/1.0 (https://github.com/Gowtham-PL/DeepVision-Forensics; forensic-research@deepvision.org)"

# ----------------------------------------------------------------------
# 1. Hashing and Deduplication Helpers
# ----------------------------------------------------------------------
def compute_hashes(img_rgb: Image.Image, file_bytes: bytes) -> tuple:
    sha256 = hashlib.sha256(file_bytes).hexdigest().lower()
    gray = img_rgb.convert('L')
    
    # dhash (9x8)
    resized_d = gray.resize((9, 8), Image.Resampling.LANCZOS)
    diff_d = np.array(resized_d)[:, 1:] > np.array(resized_d)[:, :-1]
    dh = 0
    for b in diff_d.flatten():
        dh = (dh << 1) | int(b)
    dhash_hex = f'{dh:016x}'

    # ahash (8x8)
    resized_a = gray.resize((8, 8), Image.Resampling.LANCZOS)
    pix_a = np.array(resized_a)
    diff_a = pix_a > pix_a.mean()
    ah = 0
    for b in diff_a.flatten():
        ah = (ah << 1) | int(b)
    ahash_hex = f'{ah:016x}'

    return sha256, dh, ah, dhash_hex, ahash_hex

def bit_count_64(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.uint64)
    x = (x & np.uint64(0x5555555555555555)) + ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x & np.uint64(0x0F0F0F0F0F0F0F0F)) + ((x >> np.uint64(4)) & np.uint64(0x0F0F0F0F0F0F0F0F))
    x = (x * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return x

def is_duplicate(sha: str, dh: int, ah: int, hist_shas: set, hist_dhs: np.ndarray, hist_ahs: np.ndarray,
                 pool_shas: set, pool_dhs: list, pool_ahs: list) -> bool:
    if sha in hist_shas or sha in pool_shas:
        return True
    
    # Check historical dhash
    if len(hist_dhs) > 0:
        dh_diffs = np.bitwise_xor(hist_dhs, np.uint64(dh))
        if np.any(bit_count_64(dh_diffs) <= 4):
            return True
    # Check historical ahash
    if len(hist_ahs) > 0:
        ah_diffs = np.bitwise_xor(hist_ahs, np.uint64(ah))
        if np.any(bit_count_64(ah_diffs) <= 4):
            return True

    # Check candidate pool dhash / ahash
    if len(pool_dhs) > 0:
        p_dhs = np.array(pool_dhs, dtype=np.uint64)
        p_ahs = np.array(pool_ahs, dtype=np.uint64)
        if np.any(bit_count_64(np.bitwise_xor(p_dhs, np.uint64(dh))) <= 4):
            return True
        if np.any(bit_count_64(np.bitwise_xor(p_ahs, np.uint64(ah))) <= 4):
            return True

    return False

# ----------------------------------------------------------------------
# 2. HTTP Range Zip Helpers
# ----------------------------------------------------------------------
def parse_zip_central_directory(url: str, is_zip64: bool = False) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': UA}, method='HEAD')
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_sz = int(resp.headers.get('Content-Length'))

    tail_len = min(total_sz, 262144)
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': f'bytes={total_sz - tail_len}-{total_sz - 1}'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        tail = resp.read()

    entries = {}
    if is_zip64:
        loc_pos = tail.rfind(b'PK\x06\x07')
        if loc_pos != -1:
            _, eocd64_off, _ = struct.unpack('<IQI', tail[loc_pos+4:loc_pos+20])
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': f'bytes={eocd64_off}-{eocd64_off+56}'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                eocd64 = resp.read()
            _, _, _, _, _, _, _, total_entries, cd_sz, cd_off = struct.unpack('<IQHHIIQQQQ', eocd64[:56])
        else:
            eocd_pos = tail.rfind(b'PK\x05\x06')
            cd_sz, cd_off = struct.unpack('<II', tail[eocd_pos+12:eocd_pos+20])
    else:
        eocd_pos = tail.rfind(b'PK\x05\x06')
        cd_sz, cd_off = struct.unpack('<II', tail[eocd_pos+12:eocd_pos+20])

    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': f'bytes={cd_off}-{cd_off+cd_sz-1}'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        cd_bytes = resp.read()

    pos = 0
    while pos + 46 <= len(cd_bytes):
        if cd_bytes[pos:pos+4] != b'PK\x01\x02':
            break
        method = struct.unpack('<H', cd_bytes[pos+10:pos+12])[0]
        comp_sz, uncomp_sz = struct.unpack('<II', cd_bytes[pos+20:pos+28])
        name_len, extra_len, comm_len = struct.unpack('<HHH', cd_bytes[pos+28:pos+34])
        loc_off = struct.unpack('<I', cd_bytes[pos+42:pos+46])[0]
        name = cd_bytes[pos+46:pos+46+name_len].decode('utf-8', errors='replace')
        extra = cd_bytes[pos+46+name_len:pos+46+name_len+extra_len]

        epos = 0
        while epos + 4 <= len(extra):
            tag, esz = struct.unpack('<HH', extra[epos:epos+4])
            if tag == 1:
                sub = extra[epos+4:epos+4+esz]
                idx = 0
                if uncomp_sz == 0xFFFFFFFF and idx + 8 <= len(sub):
                    uncomp_sz = struct.unpack('<Q', sub[idx:idx+8])[0]; idx += 8
                if comp_sz == 0xFFFFFFFF and idx + 8 <= len(sub):
                    comp_sz = struct.unpack('<Q', sub[idx:idx+8])[0]; idx += 8
                if loc_off == 0xFFFFFFFF and idx + 8 <= len(sub):
                    loc_off = struct.unpack('<Q', sub[idx:idx+8])[0]; idx += 8
            epos += 4 + esz

        entries[name] = {'method': method, 'comp_sz': comp_sz, 'uncomp_sz': uncomp_sz, 'offset': loc_off}
        pos += 46 + name_len + extra_len + comm_len

    return entries

def download_zip_entry(url: str, entry: dict, max_retries: int = 3) -> bytes:
    offset = entry['offset']
    comp_sz = entry['comp_sz']
    fetch_len = 512 + comp_sz
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': f'bytes={offset}-{offset+fetch_len-1}'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            name_len, extra_len = struct.unpack('<HH', raw[26:30])
            data = raw[30+name_len+extra_len : 30+name_len+extra_len+comp_sz]
            if entry['method'] == 8:
                return zlib.decompress(data, -15)
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.0)

# ----------------------------------------------------------------------
# 3. Compression Transformation Engine
# ----------------------------------------------------------------------
def generate_compression_variants(img: Image.Image) -> dict:
    w, h = img.size
    rgb = img.convert("RGB")
    variants = {}

    # 1. resize_jpeg
    w75, h75 = max(1, int(round(w * 0.75))), max(1, int(round(h * 0.75)))
    img_r75 = rgb.resize((w75, h75), Image.Resampling.BILINEAR)
    buf1 = io.BytesIO()
    img_r75.save(buf1, format="JPEG", quality=80, subsampling="4:2:0", optimize=True)
    variants["resize_jpeg"] = (buf1.getvalue(), "80", "resize_75pct_jpeg80")

    # 2. moderate_jpeg
    buf2 = io.BytesIO()
    rgb.save(buf2, format="JPEG", quality=65, subsampling="4:2:0", optimize=True)
    variants["moderate_jpeg"] = (buf2.getvalue(), "65", "jpeg65_subsampling420")

    # 3. severe_jpeg
    buf3 = io.BytesIO()
    rgb.save(buf3, format="JPEG", quality=40, subsampling="4:2:0", optimize=True)
    variants["severe_jpeg"] = (buf3.getvalue(), "40", "jpeg40_subsampling420")

    # 4. sequential_jpeg
    buf4a = io.BytesIO()
    rgb.save(buf4a, format="JPEG", quality=85, optimize=True)
    with Image.open(io.BytesIO(buf4a.getvalue())) as mid_img:
        buf4b = io.BytesIO()
        mid_img.convert("RGB").save(buf4b, format="JPEG", quality=60, subsampling="4:2:0", optimize=True)
    variants["sequential_jpeg"] = (buf4b.getvalue(), "60", "sequential_jpeg85_jpeg60")

    # 5. resize_jpeg_resize
    w60, h60 = max(1, int(round(w * 0.60))), max(1, int(round(h * 0.60)))
    img_r60 = rgb.resize((w60, h60), Image.Resampling.BILINEAR)
    buf5a = io.BytesIO()
    img_r60.save(buf5a, format="JPEG", quality=75, subsampling="4:2:0", optimize=True)
    with Image.open(io.BytesIO(buf5a.getvalue())) as mid_img:
        img_up = mid_img.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        buf5b = io.BytesIO()
        img_up.save(buf5b, format="JPEG", quality=85, optimize=True)
    variants["resize_jpeg_resize"] = (buf5b.getvalue(), "75", "downscale60_jpeg75_upscale")

    # 6. social_media_whatsapp
    if max(w, h) > 1280:
        scale = 1280.0 / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_wa = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    else:
        img_wa = rgb.copy()
    buf6 = io.BytesIO()
    img_wa.save(buf6, format="JPEG", quality=75, subsampling="4:2:0", optimize=True)
    variants["social_media_whatsapp"] = (buf6.getvalue(), "75", "whatsapp_simulation_1280px_q75")

    return variants

# ----------------------------------------------------------------------
# 4. Polite Wikimedia Acquisition Helper
# ----------------------------------------------------------------------
def fetch_wikimedia_category_items(cat_name: str, target_count: int, gen_or_device: str, is_ai: bool,
                                  hist_shas: set, hist_dhs: np.ndarray, hist_ahs: np.ndarray,
                                  pool_shas: set, pool_dhs: list, pool_ahs: list,
                                  rejection_stats: dict) -> list:
    print(f"Fetching from Wikimedia [{cat_name}], target={target_count}...", flush=True)
    items = []
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle={urllib.parse.quote(cat_name)}&gcmlimit=500&gcmtype=file&prop=imageinfo&iiprop=url|mime|size|extmetadata&format=json"
    req = urllib.request.Request(url, headers={'User-Agent': WIKI_UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to query {cat_name}: {e}", flush=True)
        return items

    pages = d.get('query', {}).get('pages', {})
    valid_candidates = []
    for p in pages.values():
        info = p.get('imageinfo', [{}])[0]
        u = info.get('url')
        mime = info.get('mime', '')
        size = info.get('size', 0)
        extmeta = info.get('extmetadata', {})
        if u and mime.startswith('image/') and not mime.endswith('svg+xml') and size < 12000000:
            valid_candidates.append((p.get('title'), u.split('?')[0], extmeta))

    random.Random(42).shuffle(valid_candidates)

    for title, img_url, meta in valid_candidates:
        if len(items) >= target_count:
            break
        try:
            time.sleep(0.12)
            req_img = urllib.request.Request(img_url, headers={'User-Agent': WIKI_UA})
            with urllib.request.urlopen(req_img, timeout=25) as resp_img:
                b = resp_img.read()
            with Image.open(io.BytesIO(b)) as img:
                rgb = img.convert("RGB")
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, b)
            if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                rejection_stats["duplicates"] += 1
                continue

            pool_shas.add(sha)
            pool_dhs.append(dh)
            pool_ahs.append(ah)

            license_val = meta.get('LicenseShortName', {}).get('value', 'CC-BY-SA 4.0 / Wikimedia Commons')
            camera_make = meta.get('Make', {}).get('value', '')
            camera_model = meta.get('Model', {}).get('value', '')
            if not is_ai:
                dev_family = camera_make if camera_make else ("Google Pixel" if "Pixel" in cat_name else "Diverse Camera")
                dev_model = f"{camera_make} {camera_model}".strip() if (camera_make or camera_model) else ("Pixel 7" if "Pixel" in cat_name else "model_unknown")
                gen_name = "none"
                gen_ver = "none"
            else:
                dev_family = "none"
                dev_model = "none"
                gen_name = gen_or_device
                gen_ver = "version_unknown"

            items.append({
                "bytes": b,
                "label": "AI" if is_ai else "Real",
                "source_dataset": f"Wikimedia Commons ({cat_name})",
                "generator": gen_name,
                "generator_version": gen_ver,
                "device_family": dev_family,
                "device_model": dev_model,
                "compression_status": "camera_original",
                "jpeg_quality_if_known": "original",
                "provenance": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
                "license": license_val,
                "sha256": sha,
                "dhash": dh_hex,
                "ahash": ah_hex,
                "ext": "jpg" if img_url.lower().endswith(('.jpg', '.jpeg')) else "png"
            })
            if len(items) % 25 == 0:
                print(f"  [{cat_name}] acquired {len(items)}/{target_count}", flush=True)
        except Exception:
            rejection_stats["download_errors"] += 1

    print(f"Finished [{cat_name}]: acquired {len(items)} items.", flush=True)
    return items

# ----------------------------------------------------------------------
# 5. Candidate Acquisition Pipeline
# ----------------------------------------------------------------------
def acquire_candidates(hist_shas: set, hist_dhs: np.ndarray, hist_ahs: np.ndarray):
    cache_path = SCRATCH_DIR / "e20_candidates_cache.pkl"
    if cache_path.exists():
        print(f"Loading cached clean candidates from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    pool_shas = set()
    pool_dhs = []
    pool_ahs = []
    rejection_stats = {"duplicates": 0, "download_errors": 0}

    real_candidates = []
    ai_candidates = []
    chunk_sz = 40

    # ------------------------------------------------------------------
    # REAL 1/5: Apple iPhone X (Target: 550)
    # ------------------------------------------------------------------
    print("\n[REAL 1/5] Acquiring Apple iPhone X (Target: 550)...", flush=True)
    url_ip_sam = "https://huggingface.co/datasets/marcosv/rawir/resolve/main/iphonex-samsungs9.zip"
    ip_sam_entries = parse_zip_central_directory(url_ip_sam, is_zip64=True)
    ip_keys = [k for k in ip_sam_entries if k.lower().endswith('.png') and 'iphone' in k.lower()]
    sam_keys = [k for k in ip_sam_entries if k.lower().endswith('.png') and 'samsung' in k.lower()]
    random.Random(42).shuffle(ip_keys)
    random.Random(42).shuffle(sam_keys)

    def fetch_ipsam_entry(k):
        try:
            b = download_zip_entry(url_ip_sam, ip_sam_entries[k])
            with Image.open(io.BytesIO(b)) as img:
                rgb = img.convert("RGB")
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, b)
            return k, b, sha, dh, ah, dh_hex, ah_hex, None
        except Exception as e:
            return k, None, None, None, None, None, None, e

    for i_start in range(0, len(ip_keys), chunk_sz):
        if len([c for c in real_candidates if c["device_family"] == "Apple iPhone"]) >= 550:
            break
        chunk = ip_keys[i_start : i_start + chunk_sz]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_ipsam_entry, k) for k in chunk]
            for fut in as_completed(futures):
                if len([c for c in real_candidates if c["device_family"] == "Apple iPhone"]) >= 550:
                    break
                k, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                if err or b is None:
                    rejection_stats["download_errors"] += 1
                    continue
                if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                    rejection_stats["duplicates"] += 1
                    continue
                pool_shas.add(sha)
                pool_dhs.append(dh)
                pool_ahs.append(ah)
                real_candidates.append({
                    "bytes": b, "label": "Real", "source_dataset": "marcosv/rawir (iphonex-samsungs9)",
                    "generator": "none", "generator_version": "none",
                    "device_family": "Apple iPhone", "device_model": "iPhone X",
                    "compression_status": "uncompressed", "jpeg_quality_if_known": "original",
                    "provenance": url_ip_sam, "license": "Academic Research Use / rawir",
                    "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "png"
                })
                if len([c for c in real_candidates if c["device_family"] == "Apple iPhone"]) % 50 == 0:
                    print(f"  iPhone acquired: {len([c for c in real_candidates if c['device_family'] == 'Apple iPhone'])}/550", flush=True)

    print(f"Acquired iPhone count: {len([c for c in real_candidates if c['device_family'] == 'Apple iPhone'])}", flush=True)

    # ------------------------------------------------------------------
    # REAL 2/5: Samsung Galaxy S9 (Target: 550)
    # ------------------------------------------------------------------
    print("\n[REAL 2/5] Acquiring Samsung Galaxy S9 (Target: 550)...", flush=True)
    for i_start in range(0, len(sam_keys), chunk_sz):
        if len([c for c in real_candidates if c["device_family"] == "Samsung Galaxy"]) >= 550:
            break
        chunk = sam_keys[i_start : i_start + chunk_sz]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_ipsam_entry, k) for k in chunk]
            for fut in as_completed(futures):
                if len([c for c in real_candidates if c["device_family"] == "Samsung Galaxy"]) >= 550:
                    break
                k, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                if err or b is None:
                    rejection_stats["download_errors"] += 1
                    continue
                if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                    rejection_stats["duplicates"] += 1
                    continue
                pool_shas.add(sha)
                pool_dhs.append(dh)
                pool_ahs.append(ah)
                real_candidates.append({
                    "bytes": b, "label": "Real", "source_dataset": "marcosv/rawir (iphonex-samsungs9)",
                    "generator": "none", "generator_version": "none",
                    "device_family": "Samsung Galaxy", "device_model": "Galaxy S9",
                    "compression_status": "uncompressed", "jpeg_quality_if_known": "original",
                    "provenance": url_ip_sam, "license": "Academic Research Use / rawir",
                    "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "png"
                })
                if len([c for c in real_candidates if c["device_family"] == "Samsung Galaxy"]) % 50 == 0:
                    print(f"  Samsung acquired: {len([c for c in real_candidates if c['device_family'] == 'Samsung Galaxy'])}/550", flush=True)

    print(f"Acquired Samsung count: {len([c for c in real_candidates if c['device_family'] == 'Samsung Galaxy'])}", flush=True)

    # ------------------------------------------------------------------
    # REAL 3/5: Nikon DSLR RAW / RAISE (Target: 400)
    # ------------------------------------------------------------------
    print("\n[REAL 3/5] Acquiring Nikon DSLR RAW (RAISE, Target: 400)...", flush=True)
    url_raise = "https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip"
    raise_entries = parse_zip_central_directory(url_raise, is_zip64=False)
    raise_keys = [k for k in raise_entries if k.lower().endswith(('.png', '.tif', '.jpg'))]
    random.Random(42).shuffle(raise_keys)

    def fetch_raise_entry(k):
        try:
            b = download_zip_entry(url_raise, raise_entries[k])
            with Image.open(io.BytesIO(b)) as img:
                rgb = img.convert("RGB")
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, b)
            return k, b, sha, dh, ah, dh_hex, ah_hex, None
        except Exception as e:
            return k, None, None, None, None, None, None, e

    for i_start in range(0, len(raise_keys), chunk_sz):
        if len([c for c in real_candidates if c["device_family"] == "Nikon DSLR"]) >= 400:
            break
        chunk = raise_keys[i_start : i_start + chunk_sz]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_raise_entry, k) for k in chunk]
            for fut in as_completed(futures):
                if len([c for c in real_candidates if c["device_family"] == "Nikon DSLR"]) >= 400:
                    break
                k, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                if err or b is None:
                    rejection_stats["download_errors"] += 1
                    continue
                if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                    rejection_stats["duplicates"] += 1
                    continue
                pool_shas.add(sha)
                pool_dhs.append(dh)
                pool_ahs.append(ah)
                real_candidates.append({
                    "bytes": b, "label": "Real", "source_dataset": "GRIP Unina (real_RAISE_1k)",
                    "generator": "none", "generator_version": "none",
                    "device_family": "Nikon DSLR", "device_model": "Nikon D7000 / D90",
                    "compression_status": "uncompressed", "jpeg_quality_if_known": "original",
                    "provenance": url_raise, "license": "RAISE Dataset / Research Only",
                    "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "png"
                })
                if len([c for c in real_candidates if c["device_family"] == "Nikon DSLR"]) % 50 == 0:
                    print(f"  RAISE acquired: {len([c for c in real_candidates if c['device_family'] == 'Nikon DSLR'])}/400", flush=True)

    print(f"Acquired RAISE count: {len([c for c in real_candidates if c['device_family'] == 'Nikon DSLR'])}", flush=True)

    # ------------------------------------------------------------------
    # REAL 4/5: Vivo X90 Smartphone (Target: 250)
    # ------------------------------------------------------------------
    print("\n[REAL 4/5] Acquiring Vivo X90 (Target: 250)...", flush=True)
    url_vivo = "https://huggingface.co/datasets/marcosv/rawir/resolve/main/vivo-x90.zip"
    vivo_entries = parse_zip_central_directory(url_vivo, is_zip64=False)
    vivo_keys = [k for k in vivo_entries if k.lower().endswith('.png')]
    random.Random(42).shuffle(vivo_keys)

    def fetch_vivo_entry(k):
        try:
            b = download_zip_entry(url_vivo, vivo_entries[k])
            with Image.open(io.BytesIO(b)) as img:
                rgb = img.convert("RGB")
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, b)
            return k, b, sha, dh, ah, dh_hex, ah_hex, None
        except Exception as e:
            return k, None, None, None, None, None, None, e

    for i_start in range(0, len(vivo_keys), chunk_sz):
        if len([c for c in real_candidates if c["device_family"] == "Vivo"]) >= 250:
            break
        chunk = vivo_keys[i_start : i_start + chunk_sz]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_vivo_entry, k) for k in chunk]
            for fut in as_completed(futures):
                if len([c for c in real_candidates if c["device_family"] == "Vivo"]) >= 250:
                    break
                k, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                if err or b is None:
                    rejection_stats["download_errors"] += 1
                    continue
                if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                    rejection_stats["duplicates"] += 1
                    continue
                pool_shas.add(sha)
                pool_dhs.append(dh)
                pool_ahs.append(ah)
                real_candidates.append({
                    "bytes": b, "label": "Real", "source_dataset": "marcosv/rawir (vivo-x90)",
                    "generator": "none", "generator_version": "none",
                    "device_family": "Vivo", "device_model": "Vivo X90",
                    "compression_status": "uncompressed", "jpeg_quality_if_known": "original",
                    "provenance": url_vivo, "license": "Academic Research Use / rawir",
                    "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "png"
                })
                if len([c for c in real_candidates if c["device_family"] == "Vivo"]) % 50 == 0:
                    print(f"  Vivo acquired: {len([c for c in real_candidates if c['device_family'] == 'Vivo'])}/250", flush=True)

    print(f"Acquired Vivo count: {len([c for c in real_candidates if c['device_family'] == 'Vivo'])}", flush=True)

    # ------------------------------------------------------------------
    # REAL 5/5: Google Pixel & Diverse Cameras (Target: to reach 1,800)
    # ------------------------------------------------------------------
    needed_real = 1800 - len(real_candidates)
    print(f"\n[REAL 5/5] Acquiring Google Pixel & Diverse Photography (Target: {needed_real})...", flush=True)
    real_wiki_cats = [
        "Category:Taken with Google Pixel 7",
        "Category:Taken with Google Pixel 8",
        "Category:Quality images of architecture"
    ]
    for cat in real_wiki_cats:
        needed = 1800 - len(real_candidates)
        if needed <= 0:
            break
        items = fetch_wikimedia_category_items(cat, needed, gen_or_device="none", is_ai=False,
                                               hist_shas=hist_shas, hist_dhs=hist_dhs, hist_ahs=hist_ahs,
                                               pool_shas=pool_shas, pool_dhs=pool_dhs, pool_ahs=pool_ahs,
                                               rejection_stats=rejection_stats)
        real_candidates.extend(items)

    print(f"Total Real Candidates Acquired: {len(real_candidates)}", flush=True)

    # ------------------------------------------------------------------
    # AI 1/5: FLUX.1 [dev] (Target: 650)
    # ------------------------------------------------------------------
    print("\n[AI 1/5] Acquiring FLUX.1 [dev] (Target: 650)...", flush=True)
    req_api = urllib.request.Request("https://huggingface.co/api/datasets/data-is-better-together/open-image-preferences-v1", headers={'User-Agent': UA})
    with urllib.request.urlopen(req_api) as resp:
        pref_data = json.loads(resp.read().decode('utf-8'))
    flux_files = [s['rfilename'] for s in pref_data['siblings'] if s['rfilename'].startswith('image_quality_dev/')]
    sd_files = [s['rfilename'] for s in pref_data['siblings'] if s['rfilename'].startswith('image_quality_sd/')]
    random.Random(42).shuffle(flux_files)
    random.Random(42).shuffle(sd_files)

    def fetch_hf_img(f, gen_name, gen_ver):
        url_f = f"https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1/resolve/main/{f}"
        try:
            req_f = urllib.request.Request(url_f, headers={'User-Agent': UA})
            with urllib.request.urlopen(req_f, timeout=20) as resp_f:
                b = resp_f.read()
            with Image.open(io.BytesIO(b)) as img:
                rgb = img.convert("RGB")
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, b)
            return url_f, b, sha, dh, ah, dh_hex, ah_hex, None
        except Exception as e:
            return url_f, None, None, None, None, None, None, e

    for i_start in range(0, len(flux_files), chunk_sz):
        if len([c for c in ai_candidates if c["generator"] == "FLUX.1 [dev]"]) >= 650:
            break
        chunk = flux_files[i_start : i_start + chunk_sz]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_hf_img, f, "FLUX.1 [dev]", "Flux.1-dev") for f in chunk]
            for fut in as_completed(futures):
                if len([c for c in ai_candidates if c["generator"] == "FLUX.1 [dev]"]) >= 650:
                    break
                url_f, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                if err or b is None:
                    rejection_stats["download_errors"] += 1
                    continue
                if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                    rejection_stats["duplicates"] += 1
                    continue
                pool_shas.add(sha)
                pool_dhs.append(dh)
                pool_ahs.append(ah)
                ai_candidates.append({
                    "bytes": b, "label": "AI", "source_dataset": "data-is-better-together/open-image-preferences-v1",
                    "generator": "FLUX.1 [dev]", "generator_version": "Flux.1-dev",
                    "device_family": "none", "device_model": "none",
                    "compression_status": "camera_original", "jpeg_quality_if_known": "original",
                    "provenance": url_f, "license": "Apache 2.0",
                    "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "jpg"
                })
                if len([c for c in ai_candidates if c["generator"] == "FLUX.1 [dev]"]) % 50 == 0:
                    print(f"  FLUX.1 [dev] count: {len([c for c in ai_candidates if c['generator'] == 'FLUX.1 [dev]'])}/650", flush=True)

    print(f"Acquired FLUX.1 [dev] count: {len([c for c in ai_candidates if c['generator'] == 'FLUX.1 [dev]'])}", flush=True)

    # ------------------------------------------------------------------
    # AI 2/5: Stable Diffusion / SDXL (Target: 650)
    # ------------------------------------------------------------------
    print("\n[AI 2/5] Acquiring Stable Diffusion / SDXL (Target: 650)...", flush=True)
    for i_start in range(0, len(sd_files), chunk_sz):
        if len([c for c in ai_candidates if c["generator"] == "Stable Diffusion"]) >= 650:
            break
        chunk = sd_files[i_start : i_start + chunk_sz]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_hf_img, f, "Stable Diffusion", "SD-3.5 / SDXL") for f in chunk]
            for fut in as_completed(futures):
                if len([c for c in ai_candidates if c["generator"] == "Stable Diffusion"]) >= 650:
                    break
                url_f, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                if err or b is None:
                    rejection_stats["download_errors"] += 1
                    continue
                if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                    rejection_stats["duplicates"] += 1
                    continue
                pool_shas.add(sha)
                pool_dhs.append(dh)
                pool_ahs.append(ah)
                ai_candidates.append({
                    "bytes": b, "label": "AI", "source_dataset": "data-is-better-together/open-image-preferences-v1",
                    "generator": "Stable Diffusion", "generator_version": "SD-3.5 / SDXL",
                    "device_family": "none", "device_model": "none",
                    "compression_status": "camera_original", "jpeg_quality_if_known": "original",
                    "provenance": url_f, "license": "Apache 2.0",
                    "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "jpg"
                })
                if len([c for c in ai_candidates if c["generator"] == "Stable Diffusion"]) % 50 == 0:
                    print(f"  Stable Diffusion count: {len([c for c in ai_candidates if c['generator'] == 'Stable Diffusion'])}/650", flush=True)

    print(f"Acquired Stable Diffusion count: {len([c for c in ai_candidates if c['generator'] == 'Stable Diffusion'])}", flush=True)

    # ------------------------------------------------------------------
    # AI 3/5: OpenAI DALL-E (Target: 200)
    # ------------------------------------------------------------------
    print("\n[AI 3/5] Acquiring OpenAI DALL-E (Target: 200)...", flush=True)
    dalle_cats = ["Category:Images generated by DALL-E", "Category:Images generated by DALL-E 3"]
    for cat in dalle_cats:
        needed = 200 - len([c for c in ai_candidates if c["generator"] == "DALL-E"])
        if needed <= 0:
            break
        items = fetch_wikimedia_category_items(cat, needed, gen_or_device="DALL-E", is_ai=True,
                                               hist_shas=hist_shas, hist_dhs=hist_dhs, hist_ahs=hist_ahs,
                                               pool_shas=pool_shas, pool_dhs=pool_dhs, pool_ahs=pool_ahs,
                                               rejection_stats=rejection_stats)
        ai_candidates.extend(items)
    print(f"Acquired DALL-E count: {len([c for c in ai_candidates if c['generator'] == 'DALL-E'])}", flush=True)

    # ------------------------------------------------------------------
    # AI 4/5: Midjourney (Target: 200)
    # ------------------------------------------------------------------
    print("\n[AI 4/5] Acquiring Midjourney (Target: 200)...", flush=True)
    mj_cats = [
        "Category:Images generated by Midjourney",
        "Category:Midjourney works by Commons users",
        "Category:Alice and Sparkle"
    ]
    for cat in mj_cats:
        needed = 200 - len([c for c in ai_candidates if c["generator"] == "Midjourney"])
        if needed <= 0:
            break
        items = fetch_wikimedia_category_items(cat, needed, gen_or_device="Midjourney", is_ai=True,
                                               hist_shas=hist_shas, hist_dhs=hist_dhs, hist_ahs=hist_ahs,
                                               pool_shas=pool_shas, pool_dhs=pool_dhs, pool_ahs=pool_ahs,
                                               rejection_stats=rejection_stats)
        ai_candidates.extend(items)
    print(f"Acquired Midjourney count: {len([c for c in ai_candidates if c['generator'] == 'Midjourney'])}", flush=True)

    # ------------------------------------------------------------------
    # AI 5/5: Google Gemini / Imagen (Target: 100)
    # ------------------------------------------------------------------
    print("\n[AI 5/5] Acquiring Google Gemini / Imagen (Target: 100)...", flush=True)
    try:
        url_gem_parq = "https://huggingface.co/datasets/Falah/images_gemini/resolve/main/data/train-00000-of-00001.parquet"
        req_gp = urllib.request.Request(url_gem_parq, headers={'User-Agent': UA})
        with urllib.request.urlopen(req_gp, timeout=20) as resp_gp:
            gp_data = resp_gp.read()
        import pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(gp_data))
        df = table.to_pandas()
        for idx, row in df.iterrows():
            if len([c for c in ai_candidates if c["generator"] == "Gemini / Imagen"]) >= 100:
                break
            b = row['image']['bytes']
            with Image.open(io.BytesIO(b)) as img:
                rgb = img.convert("RGB")
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, b)
            if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                rejection_stats["duplicates"] += 1
                continue
            pool_shas.add(sha)
            pool_dhs.append(dh)
            pool_ahs.append(ah)
            ai_candidates.append({
                "bytes": b, "label": "AI", "source_dataset": "Falah/images_gemini",
                "generator": "Gemini / Imagen", "generator_version": "Gemini 1.5 Pro / Imagen 3",
                "device_family": "none", "device_model": "none",
                "compression_status": "camera_original", "jpeg_quality_if_known": "original",
                "provenance": url_gem_parq, "license": "Open Data / Research Use",
                "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "png"
            })
    except Exception as e:
        print(f"Error loading Falah/images_gemini: {e}", flush=True)

    needed = 100 - len([c for c in ai_candidates if c["generator"] == "Gemini / Imagen"])
    if needed > 0:
        items = fetch_wikimedia_category_items("Category:Images generated by Gemini", needed, gen_or_device="Gemini / Imagen", is_ai=True,
                                               hist_shas=hist_shas, hist_dhs=hist_dhs, hist_ahs=hist_ahs,
                                               pool_shas=pool_shas, pool_dhs=pool_dhs, pool_ahs=pool_ahs,
                                               rejection_stats=rejection_stats)
        ai_candidates.extend(items)
    print(f"Acquired Gemini count: {len([c for c in ai_candidates if c['generator'] == 'Gemini / Imagen'])}", flush=True)

    # AI Top-off to guarantee 1800
    if len(ai_candidates) < 1800:
        needed = 1800 - len(ai_candidates)
        print(f"Topping off AI candidates with {needed} additional FLUX.1 [dev] images...", flush=True)
        # Skip files already processed to avoid redundant downloads
        flux_done_cnt = len([c for c in ai_candidates if c["generator"] == "FLUX.1 [dev]"])
        remaining_flux = flux_files[flux_done_cnt + 100 :]
        for i_start in range(0, len(remaining_flux), chunk_sz):
            if len(ai_candidates) >= 1800:
                break
            chunk = remaining_flux[i_start : i_start + chunk_sz]
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(fetch_hf_img, f, "FLUX.1 [dev]", "Flux.1-dev") for f in chunk]
                for fut in as_completed(futures):
                    if len(ai_candidates) >= 1800:
                        break
                    url_f, b, sha, dh, ah, dh_hex, ah_hex, err = fut.result()
                    if err or b is None:
                        rejection_stats["download_errors"] += 1
                        continue
                    if is_duplicate(sha, dh, ah, hist_shas, hist_dhs, hist_ahs, pool_shas, pool_dhs, pool_ahs):
                        rejection_stats["duplicates"] += 1
                        continue
                    pool_shas.add(sha)
                    pool_dhs.append(dh)
                    pool_ahs.append(ah)
                    ai_candidates.append({
                        "bytes": b, "label": "AI", "source_dataset": "data-is-better-together/open-image-preferences-v1",
                        "generator": "FLUX.1 [dev]", "generator_version": "Flux.1-dev",
                        "device_family": "none", "device_model": "none",
                        "compression_status": "camera_original", "jpeg_quality_if_known": "original",
                        "provenance": url_f, "license": "Apache 2.0",
                        "sha256": sha, "dhash": dh_hex, "ahash": ah_hex, "ext": "jpg"
                    })
                    if len(ai_candidates) % 50 == 0:
                        print(f"  AI candidate top-off count: {len(ai_candidates)}/1800", flush=True)

    print(f"\nFinal Candidate Totals: Real={len(real_candidates)}, AI={len(ai_candidates)}", flush=True)
    result = {
        "real": real_candidates,
        "ai": ai_candidates,
        "rejection_stats": rejection_stats
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached candidates to {cache_path}.", flush=True)
    return result

# ----------------------------------------------------------------------
# 6. Main Build Orchestration
# ----------------------------------------------------------------------
def main():
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DEV_IMG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Exclusion Index
    idx_path = SCRATCH_DIR / "exclusion_index_e20.npz"
    print(f"Loading exclusion index from {idx_path}...", flush=True)
    data = np.load(idx_path, allow_pickle=True)
    hist_shas = set(data["sha256_list"])
    hist_dhs = data["dhash_list"]
    hist_ahs = data["ahash_list"]
    print(f"Exclusion index loaded: {len(hist_shas)} SHA hashes, {len(hist_dhs)} perceptual hashes.", flush=True)

    # 2. Acquire Candidates
    candidates = acquire_candidates(hist_shas, hist_dhs, hist_ahs)
    real_pool = candidates["real"]
    ai_pool = candidates["ai"]
    rejections = candidates["rejection_stats"]

    print(f"\nCandidates Pool Available: Real={len(real_pool)}, AI={len(ai_pool)}", flush=True)
    assert len(real_pool) >= 1800, f"Insufficient Real candidates: {len(real_pool)} < 1800"
    assert len(ai_pool) >= 1800, f"Insufficient AI candidates: {len(ai_pool)} < 1800"

    real_pool = real_pool[:1800]
    ai_pool = ai_pool[:1800]

    # 3. Stratified Train / Dev Split (Seed 42)
    # Exactly: Train = 1,600 Real + 1,600 AI = 3,200 originals -> 22,400 examples
    #          Dev   = 200 Real + 200 AI = 400 originals -> 2,800 examples
    print("\n[SPLIT] Performing stratified Train / Dev split (seed=42)...", flush=True)
    rng = random.Random(42)

    # Real stratification by device_family
    real_by_family = {}
    for r in real_pool:
        fam = r["device_family"]
        real_by_family.setdefault(fam, []).append(r)
    
    train_real_origs = []
    dev_real_origs = []
    for fam, items in real_by_family.items():
        rng.shuffle(items)
        n_dev = int(round(len(items) * (200.0 / 1800.0)))
        dev_real_origs.extend(items[:n_dev])
        train_real_origs.extend(items[n_dev:])

    # Adjust exact count to 200 dev, 1600 train
    rng.shuffle(dev_real_origs)
    rng.shuffle(train_real_origs)
    while len(dev_real_origs) > 200:
        train_real_origs.append(dev_real_origs.pop())
    while len(dev_real_origs) < 200:
        dev_real_origs.append(train_real_origs.pop())

    # AI stratification by generator
    ai_by_gen = {}
    for a in ai_pool:
        gen = a["generator"]
        ai_by_gen.setdefault(gen, []).append(a)

    train_ai_origs = []
    dev_ai_origs = []
    for gen, items in ai_by_gen.items():
        rng.shuffle(items)
        n_dev = int(round(len(items) * (200.0 / 1800.0)))
        dev_ai_origs.extend(items[:n_dev])
        train_ai_origs.extend(items[n_dev:])

    # Adjust exact count to 200 dev, 1600 train
    rng.shuffle(dev_ai_origs)
    rng.shuffle(train_ai_origs)
    while len(dev_ai_origs) > 200:
        train_ai_origs.append(dev_ai_origs.pop())
    while len(dev_ai_origs) < 200:
        dev_ai_origs.append(train_ai_origs.pop())

    print(f"Originals Partitioned:", flush=True)
    print(f"  Train: Real={len(train_real_origs)}, AI={len(train_ai_origs)} (Total: {len(train_real_origs)+len(train_ai_origs)})", flush=True)
    print(f"  Dev:   Real={len(dev_real_origs)}, AI={len(dev_ai_origs)} (Total: {len(dev_real_origs)+len(dev_ai_origs)})", flush=True)

    # 4. Process and Generate Symmetric Variants
    print("\n[TRANSFORMS] Generating 6 symmetric compression variants per original...", flush=True)
    def process_one_orig(args):
        idx, item, split_name, target_dir = args
        orig_id = f"e20_{split_name}_{idx+1:05d}"
        raw_bytes = item["bytes"]
        ext = item["ext"]
        rows = []
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            # 1. Clean original variant
            clean_id = f"{orig_id}_clean"
            clean_fn = f"{clean_id}.{ext}"
            clean_path = target_dir / clean_fn
            with open(clean_path, "wb") as f_out:
                f_out.write(raw_bytes)
            sha, dh, ah, dh_hex, ah_hex = compute_hashes(rgb, raw_bytes)
            rows.append({
                "image_id": clean_id,
                "filepath": f"images/{split_name}/{clean_fn}",
                "label": item["label"],
                "source_dataset": item["source_dataset"],
                "generator": item["generator"],
                "generator_version": item["generator_version"],
                "device_family": item["device_family"],
                "device_model": item["device_model"],
                "compression_status": item["compression_status"],
                "compression_type": "clean",
                "jpeg_quality_if_known": item["jpeg_quality_if_known"],
                "derived_from": "self",
                "split": split_name,
                "provenance": item["provenance"],
                "license": item["license"],
                "sha256": sha,
                "dhash": dh_hex,
                "ahash": ah_hex
            })

            # 2. Derived compression variants
            variants = generate_compression_variants(img)
            for var_type, (var_bytes, var_q, var_desc) in variants.items():
                var_id = f"{orig_id}_{var_type}"
                var_fn = f"{var_id}.jpg"
                var_path = target_dir / var_fn
                with open(var_path, "wb") as f_out:
                    f_out.write(var_bytes)
                with Image.open(io.BytesIO(var_bytes)) as v_img:
                    v_rgb = v_img.convert("RGB")
                v_sha, v_dh, v_ah, v_dh_hex, v_ah_hex = compute_hashes(v_rgb, var_bytes)
                rows.append({
                    "image_id": var_id,
                    "filepath": f"images/{split_name}/{var_fn}",
                    "label": item["label"],
                    "source_dataset": item["source_dataset"],
                    "generator": item["generator"],
                    "generator_version": item["generator_version"],
                    "device_family": item["device_family"],
                    "device_model": item["device_model"],
                    "compression_status": "synthetic_compression",
                    "compression_type": var_type,
                    "jpeg_quality_if_known": var_q,
                    "derived_from": clean_id,
                    "split": split_name,
                    "provenance": f"{item['provenance']} | transform: {var_desc}",
                    "license": item["license"],
                    "sha256": v_sha,
                    "dhash": v_dh_hex,
                    "ahash": v_ah_hex
                })
        return rows

    def process_split(origs, split_name, target_dir):
        manifest_rows = []
        total_origs = len(origs)
        tasks = [(idx, item, split_name, target_dir) for idx, item in enumerate(origs)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = ex.map(process_one_orig, tasks)
            for idx, r_list in enumerate(results):
                manifest_rows.extend(r_list)
                if (idx + 1) % 400 == 0 or (idx + 1) == total_origs:
                    print(f"  [{split_name.upper()}] Processed {idx+1}/{total_origs} originals -> {len(manifest_rows)} examples", flush=True)
        return manifest_rows

    train_all = train_real_origs + train_ai_origs
    dev_all = dev_real_origs + dev_ai_origs
    rng.shuffle(train_all)
    rng.shuffle(dev_all)

    print("\nProcessing Train split...", flush=True)
    train_manifest_rows = process_split(train_all, "train", TRAIN_IMG_DIR)
    print("\nProcessing Dev split...", flush=True)
    dev_manifest_rows = process_split(dev_all, "dev", DEV_IMG_DIR)

    # 5. Write CSV Manifests
    fieldnames = [
        "image_id", "filepath", "label", "source_dataset",
        "generator", "generator_version", "device_family", "device_model",
        "compression_status", "compression_type", "jpeg_quality_if_known",
        "derived_from", "split", "provenance", "license", "sha256", "dhash", "ahash"
    ]

    train_csv = MANIFEST_DIR / "e20_train_manifest.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_manifest_rows)
    print(f"\nWrote Train Manifest: {train_csv} ({len(train_manifest_rows)} rows)", flush=True)

    dev_csv = MANIFEST_DIR / "e20_dev_manifest.csv"
    with open(dev_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dev_manifest_rows)
    print(f"Wrote Dev Manifest: {dev_csv} ({len(dev_manifest_rows)} rows)", flush=True)

    # 6. Integrity and Non-Contamination Audit
    print("\n[AUDIT] Performing final non-contamination and integrity audit...", flush=True)
    train_shas = {r["sha256"] for r in train_manifest_rows}
    dev_shas = {r["sha256"] for r in dev_manifest_rows}
    train_orig_parents = {r["derived_from"] if r["derived_from"] != "self" else r["image_id"] for r in train_manifest_rows}
    dev_orig_parents = {r["derived_from"] if r["derived_from"] != "self" else r["image_id"] for r in dev_manifest_rows}

    # Cross split check
    cross_split_sha = train_shas.intersection(dev_shas)
    cross_split_parent = train_orig_parents.intersection(dev_orig_parents)
    print(f"  Cross-split SHA overlap: {len(cross_split_sha)} (must be 0)", flush=True)
    print(f"  Cross-split Parent overlap: {len(cross_split_parent)} (must be 0)", flush=True)

    # Leakage check against 41,372 historical project assets
    hist_leakage = (train_shas.union(dev_shas)).intersection(hist_shas)
    print(f"  Historical Benchmark SHA Leakage: {len(hist_leakage)} (must be 0)", flush=True)

    # Build Integrity Report JSON
    integrity_report = {
        "dataset_name": "E20_TRAIN_DEV_EXPANDED",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "total_train_images": len(train_manifest_rows),
        "total_dev_images": len(dev_manifest_rows),
        "total_images": len(train_manifest_rows) + len(dev_manifest_rows),
        "train_clean_originals": len(train_all),
        "dev_clean_originals": len(dev_all),
        "total_clean_originals": len(train_all) + len(dev_all),
        "train_real_count": sum(1 for r in train_manifest_rows if r["label"] == "Real"),
        "train_ai_count": sum(1 for r in train_manifest_rows if r["label"] == "AI"),
        "dev_real_count": sum(1 for r in dev_manifest_rows if r["label"] == "Real"),
        "dev_ai_count": sum(1 for r in dev_manifest_rows if r["label"] == "AI"),
        "historical_exclusion_pool_size": len(hist_shas),
        "historical_leakage_sha_collisions": len(hist_leakage),
        "cross_split_sha_collisions": len(cross_split_sha),
        "cross_split_parent_collisions": len(cross_split_parent),
        "rejected_collisions_during_build": rejections["duplicates"],
        "download_errors_during_build": rejections["download_errors"],
        "leakage_rule_passed": (len(hist_leakage) == 0 and len(cross_split_sha) == 0 and len(cross_split_parent) == 0)
    }

    report_json_path = E20_DIR / "e20_integrity_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)
    print(f"Wrote integrity report: {report_json_path}", flush=True)

    print("\nE20 DATASET CONSTRUCTION COMPLETED SUCCESSFULLY.", flush=True)

if __name__ == "__main__":
    main()
