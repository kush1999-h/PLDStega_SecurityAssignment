"""Post-generation VAE-QIM baseline placeholder.

This baseline is intentionally separate from PLDStega because it hides after
image generation and is not the main generative method.
"""

from __future__ import annotations


class PosthocVAEQIMNotImplemented(NotImplementedError):
    """Raised when the optional posthoc baseline is requested."""


def unavailable() -> None:
    raise PosthocVAEQIMNotImplemented("posthoc-vae-qim baseline is not implemented yet")

