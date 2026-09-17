from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DamageConfig
from .core import stupidify_model


def _config_from_args(args: argparse.Namespace) -> DamageConfig:
    return DamageConfig(
        method=args.method,
        severity=args.severity,
        seed=args.seed,
        bits=args.bits,
        include=args.include or [],
        exclude=args.exclude or [],
        include_embeddings=args.include_embeddings,
        include_norms=args.include_norms,
        include_biases=args.include_biases,
        chunk_size=args.chunk_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stupidify",
        description="Deliberately degrade safetensors model checkpoints without training.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mutate = sub.add_parser("mutate", help="Create a stupidified copy of a model")
    mutate.add_argument("model", help="Hugging Face repo id or local model directory")
    mutate.add_argument("output", help="Output model directory")
    mutate.add_argument(
        "--method",
        choices=["goblin", "zero", "noise", "signflip", "bitcrush", "shuffle"],
        default="goblin",
    )
    mutate.add_argument("--severity", type=float, default=0.10, help="0.0 to 1.0 (default: 0.10)")
    mutate.add_argument("--seed", type=int, default=42)
    mutate.add_argument("--bits", type=int, default=4, help="bit depth for bitcrush/goblin")
    mutate.add_argument("--include", action="append", help="only mutate tensor names containing this text")
    mutate.add_argument("--exclude", action="append", help="skip tensor names containing this text")
    mutate.add_argument("--include-embeddings", action="store_true")
    mutate.add_argument("--include-norms", action="store_true")
    mutate.add_argument("--include-biases", action="store_true")
    mutate.add_argument("--chunk-size", type=int, default=1_000_000)
    mutate.add_argument("--cache-dir")
    mutate.add_argument("--dry-run", action="store_true", help="scan and mutate in RAM but write nothing")
    mutate.add_argument("--json", action="store_true", help="print final stats as JSON")

    recipe = sub.add_parser("recipe", help="Print a starter recipe JSON")
    recipe.add_argument("--method", default="goblin")
    recipe.add_argument("--severity", type=float, default=0.10)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "recipe":
        cfg = DamageConfig(method=args.method, severity=args.severity)
        cfg.validate()
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    cfg = _config_from_args(args)
    stats = stupidify_model(
        args.model,
        args.output,
        cfg,
        cache_dir=args.cache_dir,
        progress=None if args.json else print,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))
    elif not args.dry_run:
        print(f"Done. Damaged model saved to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
