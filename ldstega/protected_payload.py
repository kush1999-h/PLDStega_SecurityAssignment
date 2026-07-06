"""Fixed-size protected payload codewords for PLDStega."""

from __future__ import annotations

from .crypto import decrypt_payload, encrypt_payload
from .ecc import rs_decode, rs_encode


def build_protected_codeword(
    message: bytes,
    key: str,
    capacity_bytes: int,
    ecc_symbols: int,
) -> bytes:
    data_len = _validate_config(capacity_bytes, ecc_symbols)
    encrypted_packet = encrypt_payload(message, key)
    if len(encrypted_packet) > data_len:
        raise ValueError(
            "protected payload exceeds capacity; increase capacity_bytes or reduce message length"
        )
    padded_data = encrypted_packet + bytes(data_len - len(encrypted_packet))
    codeword = rs_encode(padded_data, ecc_symbols)
    if len(codeword) != capacity_bytes:
        raise ValueError(
            f"internal ECC size mismatch: expected {capacity_bytes} bytes, got {len(codeword)}"
        )
    return codeword


def recover_protected_codeword(codeword: bytes, key: str, ecc_symbols: int) -> bytes:
    if ecc_symbols < 0:
        raise ValueError("ecc_symbols must be non-negative")
    if len(codeword) <= ecc_symbols:
        raise ValueError("capacity_bytes must be greater than ecc_symbols")
    data = rs_decode(codeword, ecc_symbols)
    return decrypt_payload(data, key)


def _validate_config(capacity_bytes: int, ecc_symbols: int) -> int:
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")
    if ecc_symbols < 0:
        raise ValueError("ecc_symbols must be non-negative")
    if capacity_bytes <= ecc_symbols:
        raise ValueError("capacity_bytes must be greater than ecc_symbols")
    return capacity_bytes - ecc_symbols
