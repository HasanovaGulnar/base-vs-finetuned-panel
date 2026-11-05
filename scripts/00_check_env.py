"""Check the environment, model cache and adapter before running anything heavy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import transformers
import peft
import streamlit

from src.config import ADAPTER_DIR, MODEL_ID, configure_console
from src.prompts import PROMPTS, domain_prompts, out_of_domain_prompts


def main() -> int:
    configure_console()
    print("=" * 64)
    print("Base vs Fine-Tuned Model Comparison Panel - environment check")
    print("=" * 64)
    print(f"Python       : {sys.version.split()[0]}")
    print(f"Interpreter  : {sys.executable}")
    print(f"Project root : {ROOT}")
    print(f"OpenMP flag  : KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    print()
    print("Packages")
    print(f"  torch        : {torch.__version__} (cuda={torch.cuda.is_available()})")
    print(f"  transformers : {transformers.__version__}")
    print(f"  peft         : {peft.__version__}")
    print(f"  streamlit    : {streamlit.__version__}")
    print(f"  threads      : {torch.get_num_threads()}")

    print()
    print("Adapter")
    required = ["adapter_config.json", "adapter_model.safetensors"]
    ok = True
    for name in required:
        path = ADAPTER_DIR / name
        exists = path.exists()
        ok = ok and exists
        size = f"{path.stat().st_size / 1_048_576:.2f} MB" if exists else "MISSING"
        print(f"  {name:<26}: {size}")
    tokenizer_present = (ADAPTER_DIR / "tokenizer.json").exists()
    print(f"  {'tokenizer.json':<26}: {'OK' if tokenizer_present else 'missing (falls back to base tokenizer)'}")

    print()
    print("Base model cache")
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
        print(f"  {MODEL_ID}: OK (architecture={config.architectures}, layers={config.num_hidden_layers})")
    except Exception as exc:  # noqa: BLE001 - diagnostic
        ok = False
        print(f"  {MODEL_ID}: NOT AVAILABLE ({type(exc).__name__}: {exc})")

    print()
    print("Prompts")
    print(f"  total          : {len(PROMPTS)}")
    print(f"  in-domain      : {len(domain_prompts())}")
    print(f"  out-of-domain  : {len(out_of_domain_prompts())}")
    for prompt in PROMPTS:
        print(f"    {prompt['id']:<8} {prompt['domain']:<13} {prompt['title']}")

    print()
    if not ok:
        print("NOT READY: fix the missing model/adapter files above.")
        return 1
    print("Ready. Next: python scripts/01_run_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
