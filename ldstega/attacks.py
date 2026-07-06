"""Simple image transforms for robustness evaluation."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return small.resize((w, h))


def crop_roundtrip(image: Image.Image, pct: float) -> Image.Image:
    w, h = image.size
    dx, dy = int(w * pct), int(h * pct)
    return image.crop((dx, dy, w - dx, h - dy)).resize((w, h))


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def gaussian_noise(image: Image.Image, sigma: float = 2.0) -> Image.Image:
    arr = np.array(image).astype("float32")
    noisy = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype("uint8")
    return Image.fromarray(noisy)


def blur(image: Image.Image, radius: float = 1.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))

