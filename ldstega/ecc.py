"""Reed-Solomon error-correction wrapper."""

from __future__ import annotations


class ECCError(ValueError):
    """Raised when Reed-Solomon decoding fails."""


class ECCDependencyError(RuntimeError):
    """Raised when reedsolo is unavailable."""


def _reedsolo():
    try:
        import reedsolo
    except ImportError as exc:
        raise ECCDependencyError("reedsolo is required for PLDStega ECC") from exc
    return reedsolo


def rs_encode(data: bytes, nsym: int) -> bytes:
    if nsym < 0:
        raise ValueError("nsym must be non-negative")
    if nsym == 0:
        return data
    reedsolo = _reedsolo()
    return bytes(reedsolo.RSCodec(nsym).encode(data))


def rs_decode(data: bytes, nsym: int) -> bytes:
    if nsym < 0:
        raise ValueError("nsym must be non-negative")
    if nsym == 0:
        return data
    reedsolo = _reedsolo()
    try:
        decoded = reedsolo.RSCodec(nsym).decode(data)
    except reedsolo.ReedSolomonError as exc:
        raise ECCError("uncorrectable Reed-Solomon payload") from exc
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    return bytes(decoded)

