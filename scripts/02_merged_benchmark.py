"""Bonus: compare the merged model against the adapter-attached model.

The merged checkpoint is built with ``merge_and_unload()`` on a separate copy of
the base weights, then the same prompts are generated with all three variants.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import run_benchmark
from src.config import MERGED_JSON, configure_console
from src.engine import ComparisonEngine
from src.prompts import get_prompt


def main() -> int:
    configure_console()
    engine = ComparisonEngine()
    info = engine.load_info
    print(
        f"Loaded on {info['device']} ({info['dtype']}): "
        f"base {info['base_load_s']:.1f}s, adapter {info['adapter_attach_s']:.2f}s"
    )

    print("Building merged model (load + merge_and_unload)...")
    build_s = engine.ensure_merged()
    print(f"Merged model built in {build_s:.1f}s")

    prompts = [get_prompt("explain"), get_prompt("quiz")]
    run_benchmark(
        engine,
        prompts=prompts,
        max_new_tokens=128,
        temperature=0.0,
        variants=["base", "finetuned", "merged"],
        output_path=MERGED_JSON,
        csv_path=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
