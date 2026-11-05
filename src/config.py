"""Project paths and shared settings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ADAPTER_DIR = ROOT / "adapters" / "qwen25-python-tutor"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

RUNS_LOG = LOGS_DIR / "runs.jsonl"
VOTES_LOG = LOGS_DIR / "votes.jsonl"

RESULTS_JSON = REPORTS_DIR / "results.json"
RESULTS_CSV = REPORTS_DIR / "results.csv"
MERGED_JSON = REPORTS_DIR / "merged_benchmark.json"
RESULTS_MD = REPORTS_DIR / "RESULTS.md"
TIMINGS_PNG = REPORTS_DIR / "timings.png"

DEFAULT_MAX_NEW_TOKENS = 128

VARIANT_LABELS = {
    "base": "Base",
    "finetuned": "Fine-Tuned",
    "merged": "Merged (bonus)",
}

VARIANT_ORDER = ("base", "finetuned", "merged")


def ensure_directories() -> None:
    for directory in (REPORTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_console() -> None:
    """Force UTF-8 console output so Azerbaijani characters never crash a run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            continue
