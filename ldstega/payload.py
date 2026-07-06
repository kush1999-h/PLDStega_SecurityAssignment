"""Binary packet format for PLDStega payloads."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

MAGIC = b"PLDS"
VERSION = 1
HEADER = struct.Struct(">4sBBI12sI")


class PayloadError(ValueError):
    """Raised when a PLDStega packet is malformed or fails validation."""


@dataclass(frozen=True)
class PayloadPacket:
    flags: int
    nonce: bytes
    body: bytes


def build_packet(body: bytes, nonce: bytes, flags: int = 0) -> bytes:
    if len(nonce) != 12:
        raise PayloadError("nonce must be 12 bytes")
    if not 0 <= flags <= 255:
        raise PayloadError("flags must fit in one byte")
    header_without_crc = HEADER.pack(MAGIC, VERSION, flags, len(body), nonce, 0)
    crc = zlib.crc32(header_without_crc[:-4] + body) & 0xFFFFFFFF
    return HEADER.pack(MAGIC, VERSION, flags, len(body), nonce, crc) + body


def parse_packet(packet: bytes) -> PayloadPacket:
    parsed, consumed = parse_packet_prefix(packet)
    if consumed != len(packet):
        raise PayloadError(f"invalid packet length: expected {consumed}, got {len(packet)}")
    return parsed


def parse_packet_prefix(packet: bytes) -> tuple[PayloadPacket, int]:
    if len(packet) < HEADER.size:
        raise PayloadError("packet is too short")
    magic, version, flags, body_len, nonce, crc = HEADER.unpack(packet[: HEADER.size])
    if magic != MAGIC:
        raise PayloadError("invalid packet magic")
    if version != VERSION:
        raise PayloadError(f"unsupported packet version: {version}")
    expected_len = HEADER.size + body_len
    if len(packet) < expected_len:
        raise PayloadError(f"invalid packet length: expected at least {expected_len}, got {len(packet)}")
    body = packet[HEADER.size :]
    body = body[:body_len]
    header_without_crc = HEADER.pack(magic, version, flags, body_len, nonce, 0)
    expected_crc = zlib.crc32(header_without_crc[:-4] + body) & 0xFFFFFFFF
    if crc != expected_crc:
        raise PayloadError("invalid packet CRC")
    return PayloadPacket(flags=flags, nonce=nonce, body=body), expected_len
