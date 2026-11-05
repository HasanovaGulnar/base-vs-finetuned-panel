"""Small, dependency-free helpers used by the UI, scripts and tests."""

from __future__ import annotations

from statistics import median
from typing import Iterable, Sequence


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def char_count(text: str) -> int:
    return len(text or "")


def compare_timings(base_s: float, finetuned_s: float) -> dict:
    """Describe the timing difference in absolute and relative terms."""
    diff = finetuned_s - base_s
    pct = (diff / base_s * 100.0) if base_s else 0.0
    faster = "base" if base_s < finetuned_s else "finetuned" if finetuned_s < base_s else "tie"
    ratio = (finetuned_s / base_s) if base_s else 0.0
    return {
        "base_s": round(base_s, 3),
        "finetuned_s": round(finetuned_s, 3),
        "diff_s": round(diff, 3),
        "diff_pct": round(pct, 2),
        "faster": faster,
        "ratio": round(ratio, 3),
    }


def summarize_variant(
    results: Sequence[dict],
    variant: str,
) -> dict:
    rows = [row[variant] for row in results if variant in row and not row[variant].get("error")]
    if not rows:
        return {
            "variant": variant,
            "n": 0,
            "avg_seconds": 0.0,
            "median_seconds": 0.0,
            "avg_tokens_per_second": 0.0,
            "avg_output_tokens": 0.0,
            "avg_output_words": 0.0,
            "avg_ttft_s": 0.0,
        }
    seconds = [float(row["elapsed_s"]) for row in rows]
    ttfts = [float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None]
    return {
        "variant": variant,
        "n": len(rows),
        "avg_seconds": round(sum(seconds) / len(seconds), 3),
        "median_seconds": round(median(seconds), 3),
        "avg_tokens_per_second": round(
            sum(float(row["tokens_per_second"]) for row in rows) / len(rows), 3
        ),
        "avg_output_tokens": round(
            sum(int(row["output_tokens"]) for row in rows) / len(rows), 2
        ),
        "avg_output_words": round(
            sum(int(row["output_words"]) for row in rows) / len(rows), 2
        ),
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0.0,
    }


def summarize_runs(runs: Iterable[dict], variants: Sequence[str]) -> list[dict]:
    runs = list(runs)
    results = [run["results"] for run in runs]
    return [summarize_variant(results, variant) for variant in variants]


def flatten_runs(runs: Iterable[dict], variants: Sequence[str]) -> list[dict]:
    """One row per (prompt, variant) for CSV export."""
    rows: list[dict] = []
    for run in runs:
        for variant in variants:
            result = run["results"].get(variant)
            if not result:
                continue
            rows.append(
                {
                    "run_id": run.get("run_id", ""),
                    "prompt_id": run.get("prompt_id", ""),
                    "prompt_title": run.get("title", ""),
                    "domain": run.get("domain", ""),
                    "variant": variant,
                    "elapsed_s": result.get("elapsed_s"),
                    "ttft_s": result.get("ttft_s"),
                    "prompt_tokens": result.get("prompt_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "output_words": result.get("output_words"),
                    "output_chars": result.get("output_chars"),
                    "tokens_per_second": result.get("tokens_per_second"),
                    "max_new_tokens": result.get("max_new_tokens"),
                    "temperature": result.get("temperature"),
                    "error": result.get("error") or "",
                }
            )
    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)
