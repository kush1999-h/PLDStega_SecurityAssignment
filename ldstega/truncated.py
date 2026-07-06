"""Selection logic for LDStega latent positions."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_MS_INTERVALS: tuple[tuple[float, float | None], ...] = (
    (0.0, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, None),
)


def bucket_index(value: float, intervals: Iterable[tuple[float, float | None]]) -> int:
    for idx, (lower, upper) in enumerate(intervals):
        if idx == 0:
            lower_ok = value >= lower
        else:
            lower_ok = value > lower
        upper_ok = True if upper is None else value <= upper
        if lower_ok and upper_ok:
            return idx
    raise ValueError(f"value {value!r} does not fit any interval")


def select_positions(
    discrepancies,
    count: int,
    intervals: tuple[tuple[float, float | None], ...] = DEFAULT_MS_INTERVALS,
) -> list[int]:
    """Select flattened latent indexes from low-loss MS buckets first."""

    if count < 0:
        raise ValueError("count must be non-negative")
    flat = list(_flatten_abs(discrepancies))
    buckets: list[list[int]] = [[] for _ in intervals]
    for index, value in enumerate(flat):
        buckets[bucket_index(value, intervals)].append(index)

    positions: list[int] = []
    for bucket in buckets:
        needed = count - len(positions)
        if needed <= 0:
            break
        positions.extend(bucket[:needed])
    if len(positions) < count:
        raise ValueError(f"not enough latent positions: need {count}, got {len(positions)}")
    return positions


def _flatten_abs(values):
    try:
        import numpy as np

        if isinstance(values, np.ndarray):
            return [float(x) for x in np.abs(values).reshape(-1)]
    except Exception:
        pass

    try:
        import torch

        if isinstance(values, torch.Tensor):
            return [float(x) for x in values.detach().abs().flatten().cpu().tolist()]
    except Exception:
        pass

    def walk(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                yield from walk(child)
        else:
            yield abs(float(item))

    return list(walk(values))

