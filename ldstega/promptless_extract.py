"""Promptless extraction for PLDStega images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .bits import bits_to_bytes
from .crypto import INTERLEAVE_CONTEXT, POSITION_CONTEXT, derive_context_key
from .interleave import deinterleave_bits, majority_vote
from .positions import select_position_groups, validate_latent_capacity
from .protected_payload import recover_protected_codeword
from .qim import extract_bits_from_latent_groups
from .sdxl_pipeline import DEFAULT_SDXL_MODEL, SDXLConfig, SDXLPipelineRuntime
from .vae_runtime import VAERuntime, VAERuntimeConfig


@dataclass(frozen=True)
class PLDStegaExtractConfig:
    model_id: str = DEFAULT_SDXL_MODEL
    height: int = 1024
    width: int = 1024
    capacity_bytes: int = 128
    group_size: int = 5
    repeat: int = 3
    ecc_symbols: int = 32
    embed_method: str = "sign"
    qim_step: float = 0.1
    device: str = "auto"
    dtype: str = "auto"
    enable_cpu_offload: bool = False
    use_vae_only: bool = True


class PLDStegaExtractor:
    """Extracts PLDStega payloads from image + key only."""

    def __init__(self, config: PLDStegaExtractConfig):
        self.config = config
        validate_latent_capacity(
            _latent_shape(config.height, config.width),
            config.capacity_bytes,
            config.repeat,
            config.group_size,
        )
        if config.use_vae_only:
            self.runtime = VAERuntime(VAERuntimeConfig(config.model_id, config.device, config.dtype))
            self.vae = self.runtime.vae
        else:
            self.runtime = SDXLPipelineRuntime(
                SDXLConfig(
                    model_id=config.model_id,
                    height=config.height,
                    width=config.width,
                    steps=1,
                    guidance_scale=1.0,
                    device=config.device,
                    dtype=config.dtype,
                    enable_cpu_offload=config.enable_cpu_offload,
                )
            )
            self.vae = self.runtime.pipe.vae

    def extract(self, image_path: str | Path, key: str) -> bytes:
        image = Image.open(image_path).convert("RGB")
        errors: list[Exception] = []
        for candidate in self._candidates(image):
            try:
                latents = self._encode(candidate)
                raw_bits = self._extract_bits(latents, key)
                deinterleaved = deinterleave_bits(raw_bits, derive_context_key(key, INTERLEAVE_CONTEXT))
                voted = majority_vote(deinterleaved, self.config.repeat)
                codeword = bits_to_bytes(voted)[: self.config.capacity_bytes]
                return recover_protected_codeword(codeword, key, self.config.ecc_symbols)
            except Exception as exc:
                errors.append(exc)
        raise ValueError(f"no valid PLDStega payload found across {len(errors)} candidates")

    def _extract_bits(self, latents, key: str) -> list[int]:
        num_bits = self.config.capacity_bytes * 8 * self.config.repeat
        groups = select_position_groups(
            derive_context_key(key, POSITION_CONTEXT),
            tuple(latents.shape),
            num_bits,
            self.config.group_size,
        )
        return extract_bits_from_latent_groups(
            latents,
            groups,
            method=self.config.embed_method,
            qim_step=self.config.qim_step,
        )

    def _encode(self, image: Image.Image):
        torch = self.runtime.torch
        array = np.array(image.resize((self.config.width, self.config.height))).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor * 2 - 1).to(device=self.runtime.device, dtype=self.vae.dtype)
        scaling = getattr(self.vae.config, "scaling_factor", 0.13025)
        with torch.no_grad():
            return self.vae.encode(tensor).latent_dist.mean * scaling

    def _candidates(self, image: Image.Image):
        yield image
        yield image.resize((self.config.width, self.config.height))
        for pct in (0.02, 0.05):
            w, h = image.size
            dx, dy = int(w * pct), int(h * pct)
            if dx > 0 and dy > 0 and w > 2 * dx and h > 2 * dy:
                yield image.crop((dx, dy, w - dx, h - dy)).resize(image.size)


def _latent_shape(height: int, width: int) -> tuple[int, int, int, int]:
    return (1, 4, height // 8, width // 8)
