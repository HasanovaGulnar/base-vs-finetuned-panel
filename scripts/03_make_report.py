"""Generate reports/RESULTS.md and reports/timings.png from results.json."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import compare_timings, markdown_table, summarize_runs
from src.config import (
    MERGED_JSON,
    RESULTS_JSON,
    RESULTS_MD,
    TIMINGS_PNG,
    VARIANT_LABELS,
    VOTES_LOG,
    configure_console,
)
from src.logging_utils import read_jsonl

VARIANTS = ("base", "finetuned")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def _timing_table(payload: dict) -> str:
    rows = []
    for run in payload["runs"]:
        base = run["results"].get("base", {})
        tuned = run["results"].get("finetuned", {})
        if not base or not tuned:
            continue
        cmp = compare_timings(base["elapsed_s"], tuned["elapsed_s"])
        rows.append(
            [
                run["title"],
                run["domain"],
                f"{base['elapsed_s']:.1f}",
                f"{tuned['elapsed_s']:.1f}",
                f"{cmp['diff_s']:+.1f} s ({_pct(cmp['diff_pct'])})",
                base["output_tokens"],
                tuned["output_tokens"],
                f"{base['tokens_per_second']:.2f}",
                f"{tuned['tokens_per_second']:.2f}",
            ]
        )
    return markdown_table(
        [
            "Prompt",
            "Domain",
            "Base (s)",
            "Fine-Tuned (s)",
            "Fərq",
            "Base tokens",
            "FT tokens",
            "Base tok/s",
            "FT tok/s",
        ],
        rows,
    )


def _summary_table(payload: dict) -> str:
    summary = summarize_runs(payload["runs"], VARIANTS)
    rows = [
        [
            VARIANT_LABELS[row["variant"]],
            row["n"],
            f"{row['avg_seconds']:.1f}",
            f"{row['median_seconds']:.1f}",
            f"{row['avg_tokens_per_second']:.2f}",
            f"{row['avg_output_tokens']:.1f}",
            f"{row['avg_output_words']:.1f}",
        ]
        for row in summary
    ]
    return markdown_table(
        [
            "Variant",
            "N",
            "Ortalama (s)",
            "Median (s)",
            "Ort. tok/s",
            "Ort. token",
            "Ort. söz",
        ],
        rows,
    )


def _responses(payload: dict) -> str:
    blocks: list[str] = []
    for run in payload["runs"]:
        blocks.append(f"### {run['title']}  \n> {run['user']}\n")
        for variant in VARIANTS:
            result = run["results"].get(variant)
            if not result:
                continue
            label = VARIANT_LABELS[variant]
            blocks.append(
                f"<details open>\n<summary><b>{label}</b> — {result['elapsed_s']:.1f}s, "
                f"{result['output_tokens']} token, {result['tokens_per_second']:.2f} tok/s</summary>\n\n"
                f"~~~~text\n{result['text'] or '(empty)'}\n~~~~\n\n</details>\n"
            )
    return "\n".join(blocks)


def _chart(payload: dict) -> None:
    runs = [run for run in payload["runs"] if "base" in run["results"] and "finetuned" in run["results"]]
    if not runs:
        return
    labels = [run["title"].split(". ", 1)[-1] for run in runs]
    base = [run["results"]["base"]["elapsed_s"] for run in runs]
    tuned = [run["results"]["finetuned"]["elapsed_s"] for run in runs]

    x = np.arange(len(runs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars_base = ax.bar(x - width / 2, base, width, label="Base", color="#4C78A8")
    bars_ft = ax.bar(x + width / 2, tuned, width, label="Fine-Tuned (LoRA)", color="#F58518")
    for bars in (bars_base, bars_ft):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{bar.get_height():.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("Response time (s)")
    ax.set_title("Base vs Fine-Tuned response time per prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    TIMINGS_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TIMINGS_PNG, dpi=180)
    plt.close(fig)


def main() -> int:
    configure_console()
    payload = _load(RESULTS_JSON)
    if payload is None:
        print(f"Missing {RESULTS_JSON}. Run scripts/01_run_benchmark.py first.")
        return 1

    meta = payload["meta"]
    info = meta["load_info"]
    settings = meta["settings"]

    sections: list[str] = []
    sections.append("# Benchmark Results\n")
    sections.append(
        f"Model: **{info['model_id']}** · adapter: `{info['adapter_dir']}`  \n"
        f"Device: **{info['device']}** · dtype: **{info['dtype']}** · "
        f"torch {info['torch_version']} · {info['torch_threads']} CPU threads  \n"
        f"Settings: max_new_tokens={settings['max_new_tokens']}, "
        f"temperature={settings['temperature']}\n"
    )

    sections.append("## 1. Load cost\n")
    sections.append(
        markdown_table(
            ["Step", "Time"],
            [
                ["Tokenizer", f"{info['tokenizer_load_s']:.2f} s"],
                ["Base weights", f"{info['base_load_s']:.2f} s"],
                ["Adapter attach (`PeftModel.from_pretrained`)", f"{info['adapter_attach_s']:.2f} s"],
                ["Total startup", f"{info['total_load_s']:.2f} s"],
                ["Adapter size", f"{info['adapter_size_mb']:.2f} MB"],
                ["Trainable adapter parameters", f"{info['adapter_parameters']:,}"],
                ["Base parameters", f"{info['base_parameters']:,}"],
                ["Warm-up (2 variants, 2 tokens each)", f"{info.get('warmup_s') or 0:.2f} s"],
            ],
        )
    )

    sections.append("\n## 2. Response time per prompt\n")
    sections.append(_timing_table(payload))

    sections.append("\n## 3. Aggregate timing\n")
    sections.append(_summary_table(payload))

    sections.append("\n![Response time comparison](timings.png)\n")

    sections.append("\n## 4. Full responses\n")
    sections.append(_responses(payload))

    merged = _load(MERGED_JSON)
    if merged:
        sections.append("\n## 5. Bonus: merged model vs adapter-attached model\n")
        info_m = merged["meta"]["load_info"]
        sections.append(
            f"Merged build (`merge_and_unload` on a fresh base copy): "
            f"**{info_m.get('merged_build_s', 0):.1f} s**\n"
        )
        rows = []
        for run in merged["runs"]:
            row = [run["title"]]
            for variant in ("base", "finetuned", "merged"):
                result = run["results"].get(variant, {})
                row.append(f"{result.get('elapsed_s', 0):.1f}")
                row.append(f"{result.get('output_tokens', 0)}")
            rows.append(row)
        sections.append(
            markdown_table(
                [
                    "Prompt",
                    "Base s",
                    "Base tok",
                    "FT s",
                    "FT tok",
                    "Merged s",
                    "Merged tok",
                ],
                rows,
            )
        )

    votes = read_jsonl(VOTES_LOG)
    if votes:
        sections.append("\n## 6. Human votes\n")
        counts: dict[str, int] = {}
        for vote in votes:
            counts[vote["vote"]] = counts.get(vote["vote"], 0) + 1
        sections.append(
            markdown_table(
                ["Choice", "Votes"],
                [[key, value] for key, value in sorted(counts.items(), key=lambda item: -item[1])],
            )
        )

    RESULTS_MD.write_text("\n".join(sections), encoding="utf-8")
    _chart(payload)
    print(f"Saved: {RESULTS_MD}")
    print(f"Saved: {TIMINGS_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
