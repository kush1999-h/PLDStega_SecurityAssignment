"""Diffusers-backed LDStega implementation.

This follows the LDStega paper's practical recipe:

1. Reproduce the sender/receiver LDM run from a shared prompt and seed.
2. Find latent positions whose VAE decode/encode discrepancy is low.
3. Replace selected final-latent values with samples from left/right
   truncated Gaussian intervals around the final DDIM transition mean.
4. Decode and save the stego image; recover by VAE-encoding the received
   image and thresholding selected latent values against the reproduced mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image

from .bits import pack_payload, unpack_payload
from .truncated import DEFAULT_MS_INTERVALS, select_positions


@dataclass(frozen=True)
class LDStegaConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    height: int = 256
    width: int = 256
    steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 1.0
    gamma: float = 0.3
    device: str = "auto"
    dtype: str = "auto"
    position_strategy: str = "low-loss"


class LDStegaDiffusers:
    def __init__(self, config: LDStegaConfig):
        self.config = config
        self.torch = _import_torch()
        self.device, self.torch_dtype = self._resolve_runtime()
        self.pipe = self._load_pipeline()

    def hide(
        self,
        prompt: str,
        message: bytes,
        key: str,
        seed: int,
        output: str | Path,
        negative_prompt: str | None = None,
    ) -> dict[str, int | str]:
        bits = pack_payload(message, key)
        state = self._prepare_state(
            prompt,
            seed,
            negative_prompt,
            need_discrepancy=self.config.position_strategy == "low-loss",
        )
        positions = self._positions(state, len(bits), seed)
        stego_latents = state["base_final"].clone()
        flat_latents = stego_latents.flatten()
        flat_mean = state["mean"].flatten()

        generator = self.torch.Generator(device=self.device).manual_seed(seed + 7919)
        sampled = self._sample_truncated(
            flat_mean[self.torch.tensor(positions, device=self.device)],
            state["sigma"],
            bits,
            generator,
        )
        flat_latents[self.torch.tensor(positions, device=self.device)] = sampled

        image = self._decode_latents(stego_latents)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return {
            "output": str(output),
            "embedded_bits": len(bits),
            "message_bytes": len(message),
            "capacity_bits": int(state["base_final"].numel()),
        }

    def extract(
        self,
        prompt: str,
        image_path: str | Path,
        key: str,
        seed: int,
        max_message_bytes: int,
        negative_prompt: str | None = None,
    ) -> bytes:
        bit_count = 32 + max_message_bytes * 8
        state = self._prepare_state(
            prompt,
            seed,
            negative_prompt,
            need_discrepancy=self.config.position_strategy == "low-loss",
        )
        positions = self._positions(state, bit_count, seed)
        received = Image.open(image_path).convert("RGB")
        recovered_latents = self._encode_image(received)

        flat_recovered = recovered_latents.flatten()
        flat_mean = state["mean"].flatten()
        index = self.torch.tensor(positions, device=self.device)
        encrypted_bits = (flat_recovered[index] > flat_mean[index]).int().cpu().tolist()
        return unpack_payload(encrypted_bits, key)

    def _resolve_runtime(self):
        torch = self.torch
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch install does not include CUDA. "
                "Use --device cpu or install a CUDA-enabled PyTorch wheel."
            )

        if self.config.dtype == "auto":
            dtype = torch.float16 if device == "cuda" else torch.float32
        elif self.config.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        if device == "cpu" and dtype == torch.float16:
            raise RuntimeError("float16 on CPU is not supported here; use --dtype float32 or --dtype auto.")

        print(f"Using device={device}, dtype={str(dtype).replace('torch.', '')}")
        return device, dtype

    def _load_pipeline(self):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=False)
        return pipe

    def _prepare_state(
        self,
        prompt: str,
        seed: int,
        negative_prompt: str | None,
        need_discrepancy: bool,
    ):
        mean, sigma, base_final = self._final_transition(prompt, seed, negative_prompt)
        if not need_discrepancy:
            return {
                "mean": mean,
                "sigma": sigma,
                "base_final": base_final,
                "discrepancy": None,
            }
        image = self._decode_latents(base_final)
        reconstructed = self._encode_image(image)
        discrepancy = (reconstructed - base_final).abs()
        return {
            "mean": mean,
            "sigma": sigma,
            "base_final": base_final,
            "discrepancy": discrepancy,
        }

    def _positions(self, state, count: int, seed: int) -> list[int]:
        total = int(state["base_final"].numel())
        if count > total:
            raise ValueError(f"payload needs {count} bits, but latent capacity is {total} bits")

        if self.config.position_strategy == "low-loss":
            return select_positions(state["discrepancy"], count, DEFAULT_MS_INTERVALS)
        if self.config.position_strategy == "sequential":
            return list(range(count))
        if self.config.position_strategy == "random":
            rng = random.Random(seed + 104729)
            return rng.sample(range(total), count)
        raise ValueError("position_strategy must be one of: low-loss, random, sequential")

    def _final_transition(self, prompt: str, seed: int, negative_prompt: str | None):
        torch = self.torch
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.config.steps, device=self.device)
        prompt_embeds = self._prompt_embeds(prompt, negative_prompt)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (
            1,
            self.pipe.unet.config.in_channels,
            self.config.height // self.pipe.vae_scale_factor,
            self.config.width // self.pipe.vae_scale_factor,
        )
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=prompt_embeds.dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        timesteps = list(scheduler.timesteps)
        for timestep in timesteps[:-1]:
            latents = self._scheduler_step(latents, timestep, prompt_embeds, generator)

        timestep = timesteps[-1]
        model_output = self._noise_pred(latents, timestep, prompt_embeds)
        zero_noise = torch.zeros_like(latents)
        mean = scheduler.step(
            model_output,
            timestep,
            latents,
            eta=self.config.eta,
            variance_noise=zero_noise,
        ).prev_sample
        sigma = self._ddim_sigma(timestep)
        base_final = scheduler.step(
            model_output,
            timestep,
            latents,
            eta=self.config.eta,
            generator=generator,
        ).prev_sample
        return mean, sigma, base_final

    def _scheduler_step(self, latents, timestep, prompt_embeds, generator):
        model_output = self._noise_pred(latents, timestep, prompt_embeds)
        return self.pipe.scheduler.step(
            model_output,
            timestep,
            latents,
            eta=self.config.eta,
            generator=generator,
        ).prev_sample

    def _noise_pred(self, latents, timestep, prompt_embeds):
        torch = self.torch
        do_cfg = self.config.guidance_scale > 1.0
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = self.pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        if not do_cfg:
            return noise_pred
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + self.config.guidance_scale * (noise_text - noise_uncond)

    def _prompt_embeds(self, prompt: str, negative_prompt: str | None):
        do_cfg = self.config.guidance_scale > 1.0
        if hasattr(self.pipe, "encode_prompt"):
            embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
            )
            prompt_embeds, negative_embeds = embeds[:2]
            if do_cfg:
                return self.torch.cat([negative_embeds, prompt_embeds])
            return prompt_embeds

        return self.pipe._encode_prompt(
            prompt,
            self.device,
            1,
            do_cfg,
            negative_prompt=negative_prompt,
        )

    def _ddim_sigma(self, timestep) -> float:
        scheduler = self.pipe.scheduler
        prev_timestep = int(timestep) - scheduler.config.num_train_timesteps // self.config.steps
        variance = scheduler._get_variance(int(timestep), prev_timestep)
        sigma = self.config.eta * float(variance**0.5)
        if sigma <= 0:
            raise ValueError("DDIM sigma is zero; set --eta above 0 for LDStega embedding")
        return sigma

    def _sample_truncated(self, mean, sigma: float, bits: list[int], generator):
        torch = self.torch
        bit_tensor = torch.tensor(bits, device=self.device, dtype=mean.dtype)
        noise = torch.randn(mean.shape, generator=generator, device=self.device, dtype=mean.dtype)
        offset = self.config.gamma + noise.abs() * sigma
        return torch.where(bit_tensor > 0, mean + offset, mean - offset)

    def _decode_latents(self, latents):
        torch = self.torch
        scaling = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            image = self.pipe.vae.decode(latents / scaling).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
        return Image.fromarray((image * 255).round().astype("uint8"))

    def _encode_image(self, image: Image.Image):
        torch = self.torch
        import numpy as np

        array = np.array(image.resize((self.config.width, self.config.height))).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor * 2 - 1).to(device=self.device, dtype=self.pipe.vae.dtype)
        scaling = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            encoded = self.pipe.vae.encode(tensor).latent_dist
            latents = encoded.mean * scaling
        return latents


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install requirements.txt before using LDStega."
        ) from exc
    return torch
