import unittest
from argparse import Namespace

from ldstega.cli import _validate_args, main


class CLITests(unittest.TestCase):
    def test_pldstega_extract_accepts_promptless_arguments(self):
        args = Namespace(
            command="extract",
            mode="pldstega",
            prompt=None,
            seed=None,
            model=None,
            height=None,
            width=None,
            steps=None,
        )
        _validate_args(args, None)
        self.assertEqual(args.model, "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(args.height, 1024)
        self.assertEqual(args.width, 1024)
        self.assertEqual(args.steps, 30)

    def test_ldm_mean_extract_requires_prompt_and_seed(self):
        with self.assertRaises(SystemExit) as cm:
            main(["extract", "--mode", "ldm-mean", "--image", "x.png", "--key", "k"])
        self.assertEqual(cm.exception.code, 2)

    def test_pldstega_hide_requires_prompt(self):
        with self.assertRaises(SystemExit) as cm:
            main(["hide", "--mode", "pldstega", "--message", "x", "--key", "k", "--seed", "1"])
        self.assertEqual(cm.exception.code, 2)

    def test_posthoc_vae_qim_reserved_error(self):
        with self.assertRaises(SystemExit) as cm:
            main(["extract", "--mode", "posthoc-vae-qim", "--image", "x.png", "--key", "k"])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
