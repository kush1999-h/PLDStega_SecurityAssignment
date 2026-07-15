"""VAE-only runtime for promptless PLDStega extraction."""

from __future__ import annotations

from dataclasses import dataclass

from .sdxl_pipeline import DEFAULT_SDXL_MODEL, _from_pretrained_with_fp16_variant, _import_torch


@dataclass(frozen=True)
class VAERuntimeConfig:
    model_id: str = DEFAULT_SDXL_MODEL
    device: str = "auto"
    dtype: str = "auto"


class VAERuntime:
    def __init__(self, config: VAERuntimeConfig):
        self.config = config
        self.torch = _import_torch()
        self.device, self.dtype = self._resolve_runtime()
        self.vae = self._load_vae()

    def _resolve_runtime(self):
        torch = self.torch
        device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else self.config.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.config.dtype == "auto":
            dtype = torch.float16 if device == "cuda" else torch.float32
        elif self.config.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        return device, dtype

    def _load_vae(self):
        torch = self.torch
        from diffusers import AutoencoderKL

        vae = _from_pretrained_with_fp16_variant(
            AutoencoderKL,
            self.config.model_id,
            self.dtype,
            torch,
            subfolder="vae",
        )
        if getattr(vae.config, "force_upcast", False):
            vae.to(dtype=torch.float32)
        vae = vae.to(self.device)
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        return vae
