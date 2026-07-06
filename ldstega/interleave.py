"""Bit repetition and deterministic interleaving for PLDStega."""

from __future__ import annotations

import hashlib
import random


def repeat_bits(bits: list[int], repeat: int) -> list[int]:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    return [bit for bit in bits for _ in range(repeat)]


def majority_vote(bits: list[int], repeat: int) -> list[int]:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if len(bits) % repeat:
        raise ValueError("bit count must be divisible by repeat")
    decoded: list[int] = []
    for start in range(0, len(bits), repeat):
        chunk = bits[start : start + repeat]
        decoded.append(1 if sum(chunk) >= (repeat / 2) else 0)
    return decoded


def interleave_bits(bits: list[int], key: bytes) -> list[int]:
    order = _permutation(len(bits), key)
    out = [0] * len(bits)
    for source_index, target_index in enumerate(order):
        out[target_index] = bits[source_index]
    return out


def deinterleave_bits(bits: list[int], key: bytes) -> list[int]:
    order = _permutation(len(bits), key)
    out = [0] * len(bits)
    for source_index, target_index in enumerate(order):
        out[source_index] = bits[target_index]
    return out


def _permutation(length: int, key: bytes) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(key + length.to_bytes(8, "big")).digest()[:8], "big")
    rng = random.Random(seed)
    order = list(range(length))
    rng.shuffle(order)
    return order

