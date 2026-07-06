"""Authenticated encryption helpers for PLDStega."""

from __future__ import annotations

import os

from .payload import build_packet, parse_packet_prefix

SALT = b"pldstega-v1"
ENC_CONTEXT = b"pldstega/encryption"
POSITION_CONTEXT = b"pldstega/positions"
INTERLEAVE_CONTEXT = b"pldstega/interleave"


class CryptoDependencyError(RuntimeError):
    """Raised when cryptography is unavailable."""


def _crypto_imports():
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError as exc:
        raise CryptoDependencyError(
            "cryptography is required for PLDStega crypto; install requirements.txt"
        ) from exc
    return hashes, HKDF, ChaCha20Poly1305


def derive_master_key(passphrase: str, salt: bytes = SALT) -> bytes:
    hashes, HKDF, _ = _crypto_imports()
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"pldstega/master")
    return hkdf.derive(passphrase.encode("utf-8"))


def derive_subkey(master_key: bytes, context: bytes, length: int = 32) -> bytes:
    hashes, HKDF, _ = _crypto_imports()
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=context)
    return hkdf.derive(master_key)


def encrypt_payload(plaintext: bytes, passphrase: str, aad: bytes = b"") -> bytes:
    _, _, ChaCha20Poly1305 = _crypto_imports()
    master = derive_master_key(passphrase)
    key = derive_subkey(master, ENC_CONTEXT)
    nonce = os.urandom(12)
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
    return build_packet(ciphertext, nonce)


def decrypt_payload(packet: bytes, passphrase: str, aad: bytes = b"") -> bytes:
    _, _, ChaCha20Poly1305 = _crypto_imports()
    parsed, _ = parse_packet_prefix(packet)
    master = derive_master_key(passphrase)
    key = derive_subkey(master, ENC_CONTEXT)
    return ChaCha20Poly1305(key).decrypt(parsed.nonce, parsed.body, aad)


def derive_context_key(passphrase: str, context: bytes, length: int = 32) -> bytes:
    return derive_subkey(derive_master_key(passphrase), context, length)
