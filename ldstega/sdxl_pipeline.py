"""SDXL loading and generation helpers for PLDStega."""

from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


@dataclass(frozen=True)
class SDXLConfig:
    model_id: str = DEFAULT_SDXL_MODEL
    height: int = 1024
    width: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    device: str = "auto"
    dtype: str = "auto"
    enable_cpu_offload: bool = False


class SDXLPipelineRuntime:
    """Thin wrapper around Diffusers SDXL with RTX-friendly defaults."""

    def __init__(self, config: SDXLConfig):
        self.config = config
        self.torch = _import_torch()
        self.device, self.dtype = self._resolve_runtime()
        self.pipe = self._load_pipeline()

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

    def _load_pipeline(self):
        from diffusers import StableDiffusionXLPipeline

        pipe = _from_pretrained_with_fp16_variant(
            StableDiffusionXLPipeline,
            self.config.model_id,
            self.dtype,
            self.torch,
        )
        if getattr(pipe.vae.config, "force_upcast", False):
            pipe.vae.to(dtype=self.torch.float32)
        if self.config.enable_cpu_offload and self.device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=False)
        return pipe


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SDXL PLDStega") from exc
    return torch


def _from_pretrained_with_fp16_variant(loader, model_id: str, dtype, torch, **kwargs):
    """Load SDXL repos that store weights as either normal or fp16 variants."""
    if dtype is torch.float16:
        try:
            return loader.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16",
                use_safetensors=True,
                **kwargs,
            )
        except (OSError, ValueError):
            pass
    return loader.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
