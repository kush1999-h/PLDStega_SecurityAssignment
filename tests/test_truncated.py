import unittest

from ldstega.truncated import bucket_index, select_positions


class TruncatedIntervalTests(unittest.TestCase):
    def test_bucket_index_matches_paper_intervals(self):
        self.assertEqual(bucket_index(0.0, ((0.0, 0.05), (0.05, 0.1))), 0)
        self.assertEqual(bucket_index(0.05, ((0.0, 0.05), (0.05, 0.1))), 0)
        self.assertEqual(bucket_index(0.051, ((0.0, 0.05), (0.05, 0.1))), 1)

    def test_select_positions_prefers_low_discrepancy(self):
        discrepancies = [0.20, 0.01, 0.08, 0.03]
        self.assertEqual(select_positions(discrepancies, 3), [1, 3, 2])

    def test_select_positions_raises_when_capacity_is_too_small(self):
        with self.assertRaises(ValueError):
            select_positions([0.01], 2)


if __name__ == "__main__":
    unittest.main()
