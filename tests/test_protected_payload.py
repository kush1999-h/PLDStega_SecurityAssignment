import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("cryptography") and importlib.util.find_spec("reedsolo"),
    "cryptography and reedsolo are required",
)
class ProtectedPayloadTests(unittest.TestCase):
    def test_round_trip_fixed_size_codeword(self):
        from ldstega.protected_payload import build_protected_codeword, recover_protected_codeword

        codeword = build_protected_codeword(b"KOUSHIK", "shared-key", 128, 32)
        self.assertEqual(len(codeword), 128)
        self.assertEqual(recover_protected_codeword(codeword, "shared-key", 32), b"KOUSHIK")

    def test_wrong_key_fails(self):
        from ldstega.protected_payload import build_protected_codeword, recover_protected_codeword

        codeword = build_protected_codeword(b"KOUSHIK", "shared-key", 128, 32)
        with self.assertRaises(Exception):
            recover_protected_codeword(codeword, "wrong-key", 32)

    def test_correctable_corruption_succeeds(self):
        from ldstega.protected_payload import build_protected_codeword, recover_protected_codeword

        codeword = bytearray(build_protected_codeword(b"KOUSHIK", "shared-key", 128, 32))
        for index in range(4):
            codeword[index] ^= index + 1
        self.assertEqual(recover_protected_codeword(bytes(codeword), "shared-key", 32), b"KOUSHIK")

    def test_too_much_corruption_fails(self):
        from ldstega.protected_payload import build_protected_codeword, recover_protected_codeword

        codeword = bytearray(build_protected_codeword(b"KOUSHIK", "shared-key", 128, 32))
        for index in range(24):
            codeword[index] ^= index + 1
        with self.assertRaises(Exception):
            recover_protected_codeword(bytes(codeword), "shared-key", 32)

    def test_payload_too_large_fails(self):
        from ldstega.protected_payload import build_protected_codeword

        with self.assertRaisesRegex(ValueError, "protected payload exceeds capacity"):
            build_protected_codeword(b"x" * 200, "shared-key", 128, 32)

    def test_invalid_config_fails(self):
        from ldstega.protected_payload import build_protected_codeword

        with self.assertRaises(ValueError):
            build_protected_codeword(b"x", "shared-key", 32, 32)
        with self.assertRaises(ValueError):
            build_protected_codeword(b"x", "shared-key", 128, -1)
        with self.assertRaises(ValueError):
            build_protected_codeword(b"x", "shared-key", 0, 0)


if __name__ == "__main__":
    unittest.main()

