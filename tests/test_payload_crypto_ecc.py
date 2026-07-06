import importlib.util
import unittest

from ldstega.payload import PayloadError, build_packet, parse_packet


class PayloadTests(unittest.TestCase):
    def test_packet_round_trip(self):
        packet = build_packet(b"ciphertext", b"123456789012", flags=3)
        parsed = parse_packet(packet)
        self.assertEqual(parsed.flags, 3)
        self.assertEqual(parsed.nonce, b"123456789012")
        self.assertEqual(parsed.body, b"ciphertext")

    def test_packet_rejects_bad_crc(self):
        packet = bytearray(build_packet(b"ciphertext", b"123456789012"))
        packet[-1] ^= 1
        with self.assertRaises(PayloadError):
            parse_packet(bytes(packet))


@unittest.skipUnless(importlib.util.find_spec("cryptography"), "cryptography not installed")
class CryptoTests(unittest.TestCase):
    def test_encrypt_decrypt_round_trip(self):
        from ldstega.crypto import decrypt_payload, encrypt_payload

        packet = encrypt_payload(b"secret", "shared-key")
        self.assertEqual(decrypt_payload(packet, "shared-key"), b"secret")

    def test_wrong_key_fails(self):
        from ldstega.crypto import decrypt_payload, encrypt_payload

        packet = encrypt_payload(b"secret", "shared-key")
        with self.assertRaises(Exception):
            decrypt_payload(packet, "wrong-key")


@unittest.skipUnless(importlib.util.find_spec("reedsolo"), "reedsolo not installed")
class ECCTests(unittest.TestCase):
    def test_reed_solomon_corrects_error(self):
        from ldstega.ecc import rs_decode, rs_encode

        encoded = bytearray(rs_encode(b"hello", 8))
        encoded[0] ^= 3
        self.assertEqual(rs_decode(bytes(encoded), 8), b"hello")

    def test_reed_solomon_uncorrectable_error(self):
        from ldstega.ecc import ECCError, rs_decode, rs_encode

        encoded = bytearray(rs_encode(b"hello", 4))
        for i in range(5):
            encoded[i] ^= i + 1
        with self.assertRaises(ECCError):
            rs_decode(bytes(encoded), 4)


if __name__ == "__main__":
    unittest.main()

