# PLDStega Research Prototype

## Status

This repository is an LDStega-inspired / PLDStega research prototype. It is not
a full paper-faithful LDStega reproduction, not production-ready, not
undetectable, and not benchmark-proven robust.

This pass stabilizes the scaffold for RTX 3070 validation. Do not treat the
robustness targets as claims until the GPU benchmark suite is run.

## Implemented Modes

- `ldm-mean`: legacy prompt-dependent LDStega-style baseline.
- `pldstega`: experimental promptless generative latent diffusion
  steganography mode.
- `posthoc-vae-qim`: reserved optional post-generation baseline; not
  implemented yet.

## What PLDStega Does

PLDStega injects a protected payload into selected latent groups during SDXL
diffusion generation. The prompt controls image content during hiding. The key
controls encryption, interleaving, carrier position selection, and latent
constraints.

For extraction, PLDStega uses only:

```text
stego image + secret key
```

It does not require the original prompt, seed, scheduler, guidance scale, or
denoising path.

## What PLDStega Does Not Claim

PLDStega does not claim:

- full LDStega reproduction
- production readiness
- guaranteed robustness
- superiority over LDStega
- undetectability

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install CPU-safe development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On the RTX 3070 machine, install CUDA PyTorch first:

```powershell
python -m pip install --upgrade pip
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-gpu.txt
```

The dependency versions are pinned because this prototype was stabilized
against a specific PyTorch/Diffusers/Transformers stack. Avoid installing the
latest unpinned Diffusers or Transformers for the smoke test.

## PLDStega Hide

Run this on the RTX 3070 machine:

```powershell
python -m ldstega.cli hide `
  --mode pldstega `
  --model stabilityai/stable-diffusion-xl-base-1.0 `
  --prompt "realistic photo of a boy sitting on a chair playing an acoustic guitar, natural indoor light, sharp focus" `
  --negative-prompt "blurry, distorted, deformed hands, extra fingers, low quality" `
  --message "KOUSHIK" `
  --key "shared-key" `
  --seed 1234 `
  --output outputs\pldstega_smoke.png `
  --height 768 `
  --width 768 `
  --steps 20 `
  --guidance-scale 7.5 `
  --capacity-bytes 128 `
  --group-size 5 `
  --repeat 3 `
  --ecc-symbols 32 `
  --embed-method sign `
  --embed-strength 0.05
```

If the command fails with a VAE decode warning or produces a black image, delete
the bad output and retry with `--dtype float32` and a lower embedding strength,
for example `--embed-strength 0.02`. Do not attempt extraction from a black or
nearly blank output image.

## PLDStega Promptless Extract

No prompt. No seed.

```powershell
python -m ldstega.cli extract `
  --mode pldstega `
  --image outputs\pldstega_smoke.png `
  --key "shared-key" `
  --height 768 `
  --width 768 `
  --capacity-bytes 128 `
  --group-size 5 `
  --repeat 3 `
  --ecc-symbols 32 `
  --embed-method sign
```

The decoding parameters must match the hide parameters.

## Legacy `ldm-mean` Baseline

The legacy baseline is prompt-dependent. Extraction requires the same prompt,
seed, model, dimensions, steps, guidance scale, and payload length bound.

```powershell
python -m ldstega.cli hide `
  --mode ldm-mean `
  --prompt "a boy is playing guitar, oil on canvas" `
  --message "secret hello" `
  --key "shared-key" `
  --seed 1234 `
  --output outputs\legacy_stego.png
```

```powershell
python -m ldstega.cli extract `
  --mode ldm-mean `
  --prompt "a boy is playing guitar, oil on canvas" `
  --image outputs\legacy_stego.png `
  --key "shared-key" `
  --seed 1234 `
  --max-message-bytes 64
```

## CPU Tests

Run locally:

```powershell
python -m unittest discover -s tests -v
python -m py_compile ldstega\*.py ldstega\baselines\*.py
```

These tests do not run SDXL generation.

## Optional RTX 3070 GPU Smoke Test

After installing CUDA PyTorch and dependencies, run the PLDStega hide/extract
commands above. If 1024x1024 default generation is used and CUDA OOM occurs,
the code can retry 768x768 only when dimensions were not explicitly passed.
When dimensions are explicit, reduce dimensions or reduce capacity/repeat/group
size.

## Robustness Benchmark Plan

The robustness target is social-sharing style degradation:

- PNG recovery
- JPEG Q95/Q90/Q80/Q70
- resize round trip
- mild crop compensation
- brightness +/-10%
- contrast +/-10%
- Gaussian noise
- blur

These are benchmark targets, not completed claims.

## Publication Positioning

Recommended wording:

> This project implements an LDStega-style prompt-dependent baseline and
> proposes PLDStega, a promptless generative latent diffusion steganography
> prototype. Unlike post-generation steganography, PLDStega injects the
> protected payload into selected latent groups during SDXL diffusion
> generation. Unlike prompt-dependent LDStega-style extraction, PLDStega
> attempts to recover the message from the final stego image and secret key only
> by re-encoding the image into VAE latent space and decoding robust
> key-derived latent features.

## Limitations

- SDXL generation has not been validated on this laptop.
- RTX 3070 validation is still required.
- Robustness is not proven until benchmark results are produced.
- `posthoc-vae-qim` is reserved but not implemented.
- PLDStega is experimental and may need tuning of capacity, group size,
  repetition, ECC symbols, and embedding strength.
