"""Bit packing and key-stream helpers for LDStega payloads."""

from __future__ import annotations

import hashlib
from itertools import count


def bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise ValueError("bit length must be a multiple of 8")
    out = bytearray()
    for start in range(0, len(bits), 8):
        value = 0
        for bit in bits[start : start + 8]:
            if bit not in (0, 1):
                raise ValueError("bits must contain only 0 or 1")
            value = (value << 1) | bit
        out.append(value)
    return bytes(out)


def _keystream_bits(key: str, length: int) -> list[int]:
    if length < 0:
        raise ValueError("length must be non-negative")
    key_bytes = key.encode("utf-8")
    bits: list[int] = []
    for block_index in count():
        digest = hashlib.sha256(key_bytes + block_index.to_bytes(8, "big")).digest()
        bits.extend(bytes_to_bits(digest))
        if len(bits) >= length:
            return bits[:length]
    raise AssertionError("unreachable")


def xor_bits(bits: list[int], key: str) -> list[int]:
    stream = _keystream_bits(key, len(bits))
    return [bit ^ stream_bit for bit, stream_bit in zip(bits, stream)]


def pack_payload(message: bytes, key: str) -> list[int]:
    """Return encrypted bits with a 32-bit big-endian message length header."""

    if len(message) >= 2**32:
        raise ValueError("message is too large for the 32-bit length header")
    header = len(message).to_bytes(4, "big")
    plain_bits = bytes_to_bits(header + message)
    return xor_bits(plain_bits, key)


def unpack_payload(encrypted_bits: list[int], key: str) -> bytes:
    plain_bits = xor_bits(encrypted_bits, key)
    if len(plain_bits) < 32:
        raise ValueError("not enough bits to contain the payload header")
    message_len = int.from_bytes(bits_to_bytes(plain_bits[:32]), "big")
    required_bits = 32 + message_len * 8
    if len(plain_bits) < required_bits:
        raise ValueError(
            f"payload is incomplete: need {required_bits} bits, got {len(plain_bits)}"
        )
    return bits_to_bytes(plain_bits[32:required_bits])

