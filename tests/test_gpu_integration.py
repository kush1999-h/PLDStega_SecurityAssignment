import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_PLDSTEGA_GPU_TESTS") == "1", "GPU integration tests are opt-in")
class PLDStegaGPUIntegrationTests(unittest.TestCase):
    def test_sdxl_hide_extract_short_message(self):
        self.skipTest("Run manually on RTX 3070 after model download; not executed in CPU CI")


if __name__ == "__main__":
    unittest.main()

