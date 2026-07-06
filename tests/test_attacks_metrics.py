import math
import unittest

from PIL import Image

from ldstega.attacks import adjust_brightness, crop_roundtrip, jpeg_roundtrip, resize_roundtrip
from ldstega.metrics import bit_accuracy, bit_error_rate, message_success, psnr


class AttackMetricTests(unittest.TestCase):
    def test_basic_attacks_preserve_size(self):
        image = Image.new("RGB", (32, 32), "gray")
        self.assertEqual(jpeg_roundtrip(image, 90).size, image.size)
        self.assertEqual(resize_roundtrip(image, 0.75).size, image.size)
        self.assertEqual(crop_roundtrip(image, 0.05).size, image.size)
        self.assertEqual(adjust_brightness(image, 1.1).size, image.size)

    def test_metrics(self):
        self.assertEqual(bit_accuracy([0, 1, 1], [0, 0, 1]), 2 / 3)
        self.assertAlmostEqual(bit_error_rate([0, 1, 1], [0, 0, 1]), 1 / 3)
        self.assertTrue(message_success(b"a", b"a"))
        self.assertTrue(math.isinf(psnr(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)))))


if __name__ == "__main__":
    unittest.main()
