import unittest

from ldstega.bits import bits_to_bytes, bytes_to_bits, pack_payload, unpack_payload, xor_bits


class BitTests(unittest.TestCase):
    def test_bytes_round_trip(self):
        payload = b"LDStega"
        self.assertEqual(bits_to_bytes(bytes_to_bits(payload)), payload)

    def test_xor_bits_is_reversible(self):
        bits = [0, 1, 1, 0, 1, 0, 0, 1]
        encrypted = xor_bits(bits, "secret")
        self.assertNotEqual(encrypted, bits)
        self.assertEqual(xor_bits(encrypted, "secret"), bits)

    def test_pack_payload_round_trip(self):
        message = "hello latent world".encode()
        encrypted = pack_payload(message, "k1")
        self.assertEqual(unpack_payload(encrypted, "k1"), message)


if __name__ == "__main__":
    unittest.main()
