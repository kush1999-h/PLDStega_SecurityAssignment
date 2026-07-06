"""Group-based sign and QIM latent embedding."""

from __future__ import annotations

from collections.abc import Sequence
import math

Coordinate = tuple[int, int, int]


def embed_bits_in_latent_groups(
    latent,
    bits: list[int],
    position_groups: Sequence[Sequence[Coordinate]],
    method: str = "sign",
    strength: float = 0.03,
    qim_step: float = 0.1,
):
    if len(bits) > len(position_groups):
        raise ValueError("not enough position groups for bits")
    if method not in {"sign", "qim"}:
        raise ValueError("method must be 'sign' or 'qim'")
    out = latent.clone() if _is_torch(latent) else latent.copy()
    for bit, group in zip(bits, position_groups):
        if method == "sign":
            _embed_sign(out, int(bit), group, strength)
        else:
            _embed_qim(out, int(bit), group, qim_step)
    return out


def extract_bits_from_latent_groups(
    latent,
    position_groups: Sequence[Sequence[Coordinate]],
    method: str = "sign",
    qim_step: float = 0.1,
    threshold: float = 0.0,
) -> list[int]:
    if method not in {"sign", "qim"}:
        raise ValueError("method must be 'sign' or 'qim'")
    bits: list[int] = []
    for group in position_groups:
        if method == "sign":
            score = sum(float(_get(latent, coord)) for coord in group) / len(group)
            bits.append(1 if score > threshold else 0)
        else:
            votes = []
            for coord in group:
                q = int(round(float(_get(latent, coord)) / qim_step))
                votes.append(q & 1)
            bits.append(1 if sum(votes) >= (len(votes) / 2) else 0)
    return bits


def _embed_sign(latent, bit: int, group: Sequence[Coordinate], strength: float) -> None:
    if strength < 0:
        raise ValueError("strength must be non-negative")
    values = [float(_get(latent, coord)) for coord in group]
    mean = sum(values) / len(values)
    if bit == 1:
        delta = strength - mean if mean < strength else 0.0
    else:
        delta = -strength - mean if mean > -strength else 0.0
    for coord in group:
        _set(latent, coord, _get(latent, coord) + delta)


def _embed_qim(latent, bit: int, group: Sequence[Coordinate], qim_step: float) -> None:
    if qim_step <= 0:
        raise ValueError("qim_step must be positive")
    for coord in group:
        value = float(_get(latent, coord))
        q = int(round(value / qim_step))
        if (q & 1) != bit:
            lower = q - 1
            upper = q + 1
            q = lower if abs(value - lower * qim_step) <= abs(value - upper * qim_step) else upper
        _set(latent, coord, q * qim_step)


def _is_torch(value) -> bool:
    return value.__class__.__module__.startswith("torch")


def _get(latent, coord: Coordinate):
    c, y, x = coord
    if getattr(latent, "ndim", None) == 4:
        return latent[0, c, y, x]
    return latent[c, y, x]


def _set(latent, coord: Coordinate, value) -> None:
    c, y, x = coord
    if getattr(latent, "ndim", None) == 4:
        latent[0, c, y, x] = value
    else:
        latent[c, y, x] = value
