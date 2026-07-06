"""Lightweight robustness and image metrics."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def bit_accuracy(expected: list[int], actual: list[int]) -> float:
    if len(expected) != len(actual):
        raise ValueError("bit lists must have the same length")
    if not expected:
        return 1.0
    return sum(1 for a, b in zip(expected, actual) if a == b) / len(expected)


def bit_error_rate(expected: list[int], actual: list[int]) -> float:
    return 1.0 - bit_accuracy(expected, actual)


def message_success(expected: bytes, actual: bytes) -> bool:
    return expected == actual


def psnr(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.array(a.convert("RGB")).astype("float32")
    arr_b = np.array(b.convert("RGB").resize(a.size)).astype("float32")
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0:
        return math.inf
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(a: Image.Image, b: Image.Image) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    arr_a = np.array(a.convert("RGB"))
    arr_b = np.array(b.convert("RGB").resize(a.size))
    return float(structural_similarity(arr_a, arr_b, channel_axis=2))

