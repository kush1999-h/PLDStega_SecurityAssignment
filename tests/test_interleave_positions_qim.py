import unittest

import numpy as np

from ldstega.interleave import deinterleave_bits, interleave_bits, majority_vote, repeat_bits
from ldstega.positions import select_position_groups, validate_latent_capacity
from ldstega.qim import embed_bits_in_latent_groups, extract_bits_from_latent_groups


class InterleaveTests(unittest.TestCase):
    def test_repeat_and_majority_vote(self):
        bits = [0, 1, 1]
        repeated = repeat_bits(bits, 5)
        repeated[1] = 1
        repeated[7] = 0
        self.assertEqual(majority_vote(repeated, 5), bits)

    def test_interleave_round_trip(self):
        bits = [0, 1, 1, 0, 1, 0, 0, 1]
        key = b"interleave-key"
        self.assertEqual(deinterleave_bits(interleave_bits(bits, key), key), bits)
        self.assertNotEqual(interleave_bits(bits, key), interleave_bits(bits, b"other-key"))


class PositionTests(unittest.TestCase):
    def test_position_selection_is_deterministic_and_unique(self):
        groups_a = select_position_groups(b"key", (4, 8, 8), 10, 3)
        groups_b = select_position_groups(b"key", (4, 8, 8), 10, 3)
        groups_c = select_position_groups(b"other", (4, 8, 8), 10, 3)
        self.assertEqual(groups_a, groups_b)
        self.assertNotEqual(groups_a, groups_c)
        flat = [coord for group in groups_a for coord in group]
        self.assertEqual(len(flat), len(set(flat)))

    def test_default_capacity_fits_sdxl_1024_and_768(self):
        validate_latent_capacity((1, 4, 128, 128), 128, 3, 5)
        validate_latent_capacity((1, 4, 96, 96), 128, 3, 5)

    def test_invalid_capacity_exceeding_positions_fails(self):
        with self.assertRaises(ValueError):
            validate_latent_capacity((1, 4, 96, 96), 512, 5, 15)


class QIMTests(unittest.TestCase):
    def test_sign_embed_extract(self):
        latent = np.zeros((4, 8, 8), dtype=np.float32)
        bits = [0, 1, 1, 0]
        groups = select_position_groups(b"sign", latent.shape, len(bits), 5)
        stego = embed_bits_in_latent_groups(latent, bits, groups, method="sign", strength=0.1)
        self.assertEqual(extract_bits_from_latent_groups(stego, groups, method="sign"), bits)

    def test_sign_embedding_leaves_already_valid_group_unchanged(self):
        latent = np.ones((1, 2, 2), dtype=np.float32) * 0.2
        groups = [[(0, 0, 0), (0, 0, 1)]]
        stego = embed_bits_in_latent_groups(latent, [1], groups, method="sign", strength=0.1)
        np.testing.assert_allclose(stego, latent)

    def test_sign_embedding_corrects_wrong_side_group(self):
        latent = np.ones((1, 2, 2), dtype=np.float32) * -0.2
        groups = [[(0, 0, 0), (0, 0, 1)]]
        stego = embed_bits_in_latent_groups(latent, [1], groups, method="sign", strength=0.1)
        self.assertEqual(extract_bits_from_latent_groups(stego, groups, method="sign"), [1])

    def test_qim_embed_extract(self):
        latent = np.zeros((4, 8, 8), dtype=np.float32)
        bits = [0, 1, 1, 0]
        groups = select_position_groups(b"qim", latent.shape, len(bits), 5)
        stego = embed_bits_in_latent_groups(latent, bits, groups, method="qim", qim_step=0.2)
        self.assertEqual(extract_bits_from_latent_groups(stego, groups, method="qim", qim_step=0.2), bits)


if __name__ == "__main__":
    unittest.main()
