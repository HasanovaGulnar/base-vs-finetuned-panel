"""Run the full base vs fine-tuned benchmark over every test prompt."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import run_benchmark
from src.config import DEFAULT_MAX_NEW_TOKENS, configure_console
from src.engine import ComparisonEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "finetuned"],
        choices=["base", "finetuned", "merged"],
    )
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    engine = ComparisonEngine()
    info = engine.load_info
    print(
        f"Loaded on {info['device']} ({info['dtype']}), "
        f"base {info['base_load_s']:.1f}s + adapter {info['adapter_attach_s']:.2f}s"
    )
    if "merged" in args.variants:
        print("Merging adapter into a separate model copy...")
        build_s = engine.ensure_merged()
        print(f"Merged model ready in {build_s:.1f}s")

    run_benchmark(
        engine,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        variants=args.variants,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
