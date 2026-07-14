"""Command line interface for the local LDStega implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ldstega")
    sub = parser.add_subparsers(dest="command", required=True)

    hide = _add_common(sub.add_parser("hide", help="embed a message into a generated image"))
    hide.add_argument("--message", required=True)
    hide.add_argument("--output")

    extract = _add_common(sub.add_parser("extract", help="extract a message from a stego image"))
    extract.add_argument("--image", required=True)
    extract.add_argument("--max-message-bytes", type=int)

    args = parser.parse_args(argv)
    try:
        _validate_args(args, parser)
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode == "pldstega":
        return _run_pldstega(args)
    if args.mode == "posthoc-vae-qim":
        parser.error("posthoc-vae-qim is reserved but not implemented yet")

    return _run_legacy(args)


def _run_legacy(args) -> int:
    from .diffusers_ldstega import LDStegaConfig, LDStegaDiffusers

    config = LDStegaConfig(
        model_id=args.model,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        gamma=args.gamma,
        device=args.device,
        dtype=args.dtype,
        position_strategy=args.position_strategy,
    )
    runner = LDStegaDiffusers(config)
    if args.command == "hide":
        result = runner.hide(
            prompt=args.prompt,
            message=args.message.encode("utf-8"),
            key=args.key,
            seed=args.seed,
            output=args.output,
            negative_prompt=args.negative_prompt,
        )
        print(f"wrote {result['output']}")
        print(f"embedded_bits={result['embedded_bits']} capacity_bits={result['capacity_bits']}")
        return 0

    try:
        message = runner.extract(
            prompt=args.prompt,
            image_path=Path(args.image),
            key=args.key,
            seed=args.seed,
            max_message_bytes=args.max_message_bytes,
            negative_prompt=args.negative_prompt,
        )
    except ValueError as exc:
        print(f"extract failed: {exc}", file=sys.stderr)
        print(
            "Increase --max-message-bytes to at least the original message length "
            "and rerun with the same prompt, key, seed, and generation settings.",
            file=sys.stderr,
        )
        return 2
    print(message.decode("utf-8", errors="replace"))
    return 0


def _run_pldstega(args) -> int:
    if args.command == "hide":
        from .generative_embed import PLDStegaConfig, PLDStegaEmbedder

        config = PLDStegaConfig(
            model_id=args.model,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            capacity_bytes=args.capacity_bytes,
            group_size=args.group_size,
            repeat=args.repeat,
            ecc_symbols=args.ecc_symbols,
            embed_method=args.embed_method,
            embed_strength=args.embed_strength,
            qim_step=args.qim_step,
            device=args.device,
            dtype=args.dtype,
            enable_cpu_offload=args.cpu_offload,
            allow_size_fallback=args.allow_size_fallback,
            stabilize_rounds=args.stabilize_rounds,
            verify_after_hide=not args.allow_unverified,
        )
        result = PLDStegaEmbedder(config).hide(
            prompt=args.prompt,
            message=args.message.encode("utf-8"),
            key=args.key,
            seed=args.seed,
            output=args.output,
            negative_prompt=args.negative_prompt,
        )
        print(f"wrote {result['output']}")
        print(f"embedded_bits={result['embedded_bits']} capacity_bits={result['capacity_bits']}")
        print(f"verified={bool(result.get('verified', 0))}")
        return 0

    from .promptless_extract import PLDStegaExtractConfig, PLDStegaExtractor

    config = PLDStegaExtractConfig(
        model_id=args.model,
        height=args.height,
        width=args.width,
        capacity_bytes=args.capacity_bytes,
        group_size=args.group_size,
        repeat=args.repeat,
        ecc_symbols=args.ecc_symbols,
        embed_method=args.embed_method,
        qim_step=args.qim_step,
        device=args.device,
        dtype=args.dtype,
        enable_cpu_offload=args.cpu_offload,
        use_vae_only=not args.full_pipeline_extract,
    )
    try:
        message = PLDStegaExtractor(config).extract(args.image, args.key)
    except ValueError as exc:
        print(f"extract failed: {exc}", file=sys.stderr)
        return 2
    print(message.decode("utf-8", errors="replace"))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mode", choices=["ldm-mean", "pldstega", "posthoc-vae-qim"], default="ldm-mean")
    parser.add_argument("--prompt")
    parser.add_argument("--key", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--model")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument(
        "--position-strategy",
        choices=["low-loss", "random", "sequential"],
        default="low-loss",
        help="low-loss follows the paper but uses more memory; random is lighter",
    )
    parser.add_argument("--capacity-bytes", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--ecc-symbols", type=int, default=32)
    parser.add_argument("--embed-method", choices=["sign", "qim"], default="sign")
    parser.add_argument("--embed-strength", type=float, default=0.03)
    parser.add_argument("--qim-step", type=float, default=0.1)
    parser.add_argument("--stabilize-rounds", type=int, default=2)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--full-pipeline-extract", action="store_true")
    return parser


def _validate_args(args, parser: argparse.ArgumentParser) -> None:
    args.allow_size_fallback = False
    if args.mode == "posthoc-vae-qim":
        return
    if args.command == "hide" and not args.prompt:
        raise ValueError("hide requires --prompt")
    if args.command == "hide" and args.seed is None:
        raise ValueError("hide requires --seed")
    if args.mode == "ldm-mean":
        if args.model is None:
            args.model = "runwayml/stable-diffusion-v1-5"
        if args.height is None:
            args.height = 256
        if args.width is None:
            args.width = 256
        if args.steps is None:
            args.steps = 50
        if args.command == "hide" and not args.output:
            args.output = "outputs/stego.png"
        if not args.prompt:
            raise ValueError("ldm-mean mode requires --prompt")
        if args.seed is None:
            raise ValueError("ldm-mean mode requires --seed")
        if args.command == "extract" and args.max_message_bytes is None:
            raise ValueError("ldm-mean extract requires --max-message-bytes")
    if args.mode == "pldstega":
        explicit_size = args.height is not None or args.width is not None
        if args.model is None or args.model == "runwayml/stable-diffusion-v1-5":
            args.model = "stabilityai/stable-diffusion-xl-base-1.0"
        if args.height is None:
            args.height = 1024
        if args.width is None:
            args.width = 1024
        if args.steps is None:
            args.steps = 30
        args.allow_size_fallback = not explicit_size
        if args.command == "hide" and not args.output:
            raise ValueError("pldstega hide requires --output")


if __name__ == "__main__":
    raise SystemExit(main())
