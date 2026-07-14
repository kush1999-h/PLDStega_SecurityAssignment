"""Promptless generative latent diffusion steganography sender."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from .bits import bytes_to_bits
from .crypto import INTERLEAVE_CONTEXT, POSITION_CONTEXT, derive_context_key
from .interleave import interleave_bits, repeat_bits
from .positions import select_position_groups, validate_latent_capacity
from .protected_payload import build_protected_codeword
from .qim import embed_bits_in_latent_groups
from .sdxl_pipeline import DEFAULT_SDXL_MODEL, SDXLConfig, SDXLPipelineRuntime


@dataclass(frozen=True)
class PLDStegaConfig:
    model_id: str = DEFAULT_SDXL_MODEL
    height: int = 1024
    width: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    capacity_bytes: int = 128
    group_size: int = 5
    repeat: int = 3
    ecc_symbols: int = 32
    embed_method: str = "sign"
    embed_strength: float = 0.03
    qim_step: float = 0.1
    reinforce_fraction: float = 0.25
    device: str = "auto"
    dtype: str = "auto"
    enable_cpu_offload: bool = False
    allow_size_fallback: bool = False


class PLDStegaEmbedder:
    """Embeds a protected payload into SDXL latents during generation."""

    def __init__(self, config: PLDStegaConfig):
        self.config = config
        validate_latent_capacity(
            _latent_shape(config.height, config.width),
            config.capacity_bytes,
            config.repeat,
            config.group_size,
        )
        self.runtime = SDXLPipelineRuntime(
            SDXLConfig(
                model_id=config.model_id,
                height=config.height,
                width=config.width,
                steps=config.steps,
                guidance_scale=config.guidance_scale,
                device=config.device,
                dtype=config.dtype,
                enable_cpu_offload=config.enable_cpu_offload,
            )
        )

    def hide(
        self,
        prompt: str,
        message: bytes,
        key: str,
        seed: int,
        output: str | Path,
        negative_prompt: str | None = None,
    ) -> dict[str, int | str]:
        try:
            return self._hide_once(prompt, message, key, seed, output, negative_prompt)
        except RuntimeError as exc:
            if not self._should_retry_768(exc):
                raise
            self.runtime.torch.cuda.empty_cache()
            fallback = replace(self.config, height=768, width=768, allow_size_fallback=False)
            print("CUDA OOM at 1024x1024; retrying PLDStega hide at 768x768.")
            return PLDStegaEmbedder(fallback).hide(prompt, message, key, seed, output, negative_prompt)

    def _hide_once(
        self,
        prompt: str,
        message: bytes,
        key: str,
        seed: int,
        output: str | Path,
        negative_prompt: str | None = None,
    ) -> dict[str, int | str]:
        bits = self._protected_bits(message, key)
        pipe = self.runtime.pipe
        torch = self.runtime.torch
        device = self.runtime.device

        prompt_embeds, negative_embeds, pooled, negative_pooled = self._encode_prompt(prompt, negative_prompt)
        pipe.scheduler.set_timesteps(self.config.steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = self._initial_latents(generator, prompt_embeds.dtype)

        groups = self._groups(key, latents.shape, len(bits))
        latents = embed_bits_in_latent_groups(
            latents,
            bits,
            groups,
            method=self.config.embed_method,
            strength=self.config.embed_strength,
            qim_step=self.config.qim_step,
        )

        timesteps = list(pipe.scheduler.timesteps)
        reinforce_from = int(len(timesteps) * (1.0 - self.config.reinforce_fraction))
        with torch.no_grad():
            for index, timestep in enumerate(timesteps):
                latent_input = torch.cat([latents] * 2) if self.config.guidance_scale > 1.0 else latents
                latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)
                added = {
                    "text_embeds": torch.cat([negative_pooled, pooled])
                    if self.config.guidance_scale > 1.0
                    else pooled,
                    "time_ids": self._time_ids(prompt_embeds.dtype),
                }
                noise_pred = pipe.unet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=torch.cat([negative_embeds, prompt_embeds])
                    if self.config.guidance_scale > 1.0
                    else prompt_embeds,
                    added_cond_kwargs=added,
                ).sample
                if self.config.guidance_scale > 1.0:
                    uncond, text = noise_pred.chunk(2)
                    noise_pred = uncond + self.config.guidance_scale * (text - uncond)
                latents = pipe.scheduler.step(noise_pred, timestep, latents, generator=generator).prev_sample
                if index >= reinforce_from:
                    latents = embed_bits_in_latent_groups(
                        latents,
                        bits,
                        groups,
                        method=self.config.embed_method,
                        strength=self.config.embed_strength,
                        qim_step=self.config.qim_step,
                    )

        image = self._decode(latents)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return {"output": str(output), "embedded_bits": len(bits), "capacity_bits": int(latents.numel())}

    def _protected_bits(self, message: bytes, key: str) -> list[int]:
        codeword = build_protected_codeword(
            message,
            key,
            self.config.capacity_bytes,
            self.config.ecc_symbols,
        )
        bits = bytes_to_bits(codeword)
        bits = repeat_bits(bits, self.config.repeat)
        return interleave_bits(bits, derive_context_key(key, INTERLEAVE_CONTEXT))

    def _groups(self, key: str, latent_shape, num_bits: int):
        position_key = derive_context_key(key, POSITION_CONTEXT)
        return select_position_groups(position_key, tuple(latent_shape), num_bits, self.config.group_size)

    def _encode_prompt(self, prompt: str, negative_prompt: str | None):
        return self.runtime.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=self.runtime.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.config.guidance_scale > 1.0,
        )[:4]

    def _initial_latents(self, generator, dtype):
        torch = self.runtime.torch
        pipe = self.runtime.pipe
        shape = (
            1,
            pipe.unet.config.in_channels,
            self.config.height // pipe.vae_scale_factor,
            self.config.width // pipe.vae_scale_factor,
        )
        latents = torch.randn(shape, generator=generator, device=self.runtime.device, dtype=dtype)
        return latents * pipe.scheduler.init_noise_sigma

    def _time_ids(self, dtype):
        torch = self.runtime.torch
        pipe = self.runtime.pipe
        values = (self.config.height, self.config.width, 0, 0, self.config.height, self.config.width)
        time_ids = torch.tensor([values], device=self.runtime.device, dtype=dtype)
        if self.config.guidance_scale > 1.0:
            time_ids = torch.cat([time_ids, time_ids])
        return time_ids

    def _decode(self, latents):
        pipe = self.runtime.pipe
        torch = self.runtime.torch
        if not torch.isfinite(latents).all():
            raise RuntimeError("non-finite diffusion latents before VAE decode")
        scaling = getattr(pipe.vae.config, "scaling_factor", 0.13025)
        decode_latents = (latents / scaling).to(dtype=pipe.vae.dtype)
        image = pipe.vae.decode(decode_latents).sample
        if not torch.isfinite(image).all():
            raise RuntimeError(
                "VAE decode produced non-finite pixels; try --dtype float32, "
                "lower --guidance-scale, or lower --embed-strength"
            )
        image = (image / 2 + 0.5).clamp(0, 1)
        if (image.max() - image.min()).item() < 1e-4:
            raise RuntimeError(
                "VAE decode produced a nearly constant image; delete this output and retry "
                "with --dtype float32, lower --guidance-scale, or lower --embed-strength"
            )
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
        from PIL import Image

        return Image.fromarray((image * 255).round().astype("uint8"))

    def _should_retry_768(self, exc: RuntimeError) -> bool:
        if not self.config.allow_size_fallback:
            return False
        if self.config.height != 1024 or self.config.width != 1024:
            return False
        if not hasattr(self.runtime.torch, "cuda") or not self.runtime.torch.cuda.is_available():
            return False
        return "out of memory" in str(exc).lower()


def _latent_shape(height: int, width: int) -> tuple[int, int, int, int]:
    return (1, 4, height // 8, width // 8)
