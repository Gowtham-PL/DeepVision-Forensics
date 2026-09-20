"""
In-Memory Realistic Social-Media / WhatsApp Compression Augmentation Pipeline.

Simulates the degradation pipeline typical of WhatsApp and social messaging platforms:
1. Downscaling / resolution reduction
2. Lossy JPEG compression with 4:2:0 chroma subsampling
3. Multi-pass JPEG recompression
4. Subtle anti-aliasing / lens blur

Operates entirely in-memory using io.BytesIO and PIL, creating 0 permanent disk files.
"""

import io
import random
from PIL import Image, ImageFilter


def jpeg_compress_in_memory(img: Image.Image, quality: int, subsampling: int = 2) -> Image.Image:
    """
    Compresses an RGB PIL image using standard JPEG encoding and decodes it back.
    subsampling=2 corresponds to 4:2:0 chroma subsampling (standard in JPEG and WhatsApp).
    subsampling=0 corresponds to 4:4:4 (no subsampling).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # Force decode pixel buffer into memory
    return out


def apply_compression_augmentation(
    img: Image.Image,
    p_clean: float = 0.35,
    p_mild: float = 0.25,
    p_moderate: float = 0.20,
    p_severe: float = 0.20,
) -> Image.Image:
    """
    Applies probabilistic on-the-fly compression augmentation to a PIL RGB image.
    
    Regimes:
    - Clean (35%): Image untouched.
    - Mild (25%): Random single-pass JPEG (Q in [75, 95]).
    - Moderate (20%): Downscale (0.5x - 0.75x), resize back + JPEG (Q in [55, 75]).
    - Severe / WhatsApp Recompression (20%):
        Downscale (0.4x - 0.6x), optional mild Gaussian blur,
        1st JPEG pass (Q in [40, 65], 4:2:0 subsampling),
        decode, 2nd JPEG pass (Q in [45, 70]), resize back.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    orig_w, orig_h = img.size
    r = random.random()

    # 1. Clean regime (35%)
    if r < p_clean:
        return img

    # 2. Mild compression regime (25%)
    elif r < p_clean + p_mild:
        q = random.randint(75, 95)
        subsampling = random.choice([0, 2])
        return jpeg_compress_in_memory(img, quality=q, subsampling=subsampling)

    # 3. Moderate compression regime (20%)
    elif r < p_clean + p_mild + p_moderate:
        scale = random.uniform(0.50, 0.75)
        new_w = max(16, int(round(orig_w * scale)))
        new_h = max(16, int(round(orig_h * scale)))
        downscaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        q = random.randint(55, 75)
        compressed = jpeg_compress_in_memory(downscaled, quality=q, subsampling=2)
        return compressed.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    # 4. Severe / WhatsApp-like Recompression regime (20%)
    else:
        scale = random.uniform(0.40, 0.60)
        new_w = max(16, int(round(orig_w * scale)))
        new_h = max(16, int(round(orig_h * scale)))
        downscaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Mild blur simulating pre-compression filter or lens softening (50% chance)
        if random.random() < 0.5:
            radius = random.uniform(0.4, 0.9)
            downscaled = downscaled.filter(ImageFilter.GaussianBlur(radius=radius))

        # First pass JPEG
        q1 = random.randint(40, 65)
        pass1 = jpeg_compress_in_memory(downscaled, quality=q1, subsampling=2)

        # Second pass JPEG (recompression with slight quality variation)
        q2 = random.randint(45, 70)
        pass2 = jpeg_compress_in_memory(pass1, quality=q2, subsampling=2)

        # Upscale back to original geometry
        return pass2.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
