"""Key-derived latent carrier position selection."""

from __future__ import annotations

import hashlib
import random


def select_position_groups(
    key: bytes,
    latent_shape: tuple[int, ...],
    num_bits: int,
    group_size: int,
    reserved_margin: int = 0,
) -> list[list[tuple[int, int, int]]]:
    """Return deterministic non-duplicated groups of latent coordinates.

    Coordinates are normalized to ``(channel, height, width)``. A batch dimension
    is accepted but must be one.
    """

    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    channels, height, width = _chw(latent_shape)
    candidates = [
        (c, y, x)
        for c in range(channels)
        for y in range(reserved_margin, height - reserved_margin)
        for x in range(reserved_margin, width - reserved_margin)
    ]
    needed = num_bits * group_size
    if needed > len(candidates):
        raise ValueError(f"not enough latent positions: need {needed}, got {len(candidates)}")
    seed_material = key + repr((latent_shape, num_bits, group_size, reserved_margin)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)
    selected = rng.sample(candidates, needed)
    return [selected[i : i + group_size] for i in range(0, needed, group_size)]


def validate_latent_capacity(
    latent_shape: tuple[int, ...],
    capacity_bytes: int,
    repeat: int,
    group_size: int,
) -> None:
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    channels, height, width = _chw(latent_shape)
    required_positions = capacity_bytes * 8 * repeat * group_size
    available_positions = channels * height * width
    if required_positions > available_positions:
        raise ValueError(
            "latent capacity is too small: "
            f"requires {required_positions} positions "
            f"({capacity_bytes} bytes * 8 * repeat {repeat} * group_size {group_size}), "
            f"but latent shape {latent_shape} has {available_positions}. "
            "Reduce --capacity-bytes, --repeat, or --group-size, or use a larger image size."
        )


def _chw(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 3:
        return shape
    if len(shape) == 4:
        batch, channels, height, width = shape
        if batch != 1:
            raise ValueError("only batch size 1 is supported")
        return channels, height, width
    raise ValueError("latent_shape must be (C,H,W) or (1,C,H,W)")
