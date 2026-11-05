"""Run every test prompt through every requested variant and save the results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .analysis import flatten_runs
from .config import RESULTS_CSV, RESULTS_JSON
from .logging_utils import new_run_id
from .prompts import PROMPTS

Progress = Callable[[str], None]


def _default_progress(message: str) -> None:
    print(message, flush=True)


def run_benchmark(
    engine,
    prompts: Iterable[dict] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    variants: Iterable[str] = ("base", "finetuned"),
    output_path: str | Path = RESULTS_JSON,
    csv_path: str | Path | None = RESULTS_CSV,
    progress: Progress | None = None,
) -> dict:
    prompts = list(PROMPTS if prompts is None else prompts)
    variants = list(variants)
    say = progress or _default_progress
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "load_info": dict(engine.load_info),
            "settings": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "variants": variants,
            },
            "prompts": [{k: p[k] for k in ("id", "title", "domain", "system", "user")} for p in prompts],
        },
        "runs": [],
    }

    for prompt in prompts:
        run = {
            "run_id": new_run_id(),
            "prompt_id": prompt["id"],
            "title": prompt["title"],
            "domain": prompt["domain"],
            "system": prompt["system"],
            "user": prompt["user"],
            "results": {},
        }
        say(f"\n=== {prompt['title']} ({prompt['id']}) ===")
        payload["runs"].append(run)
        for variant in variants:
            say(f"  {variant:<9} generating...")
            result = engine.generate(
                variant,
                prompt["system"],
                prompt["user"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            run["results"][variant] = result.to_dict()
            if result.error:
                say(f"  {variant:<9} ERROR {result.error}")
            else:
                say(
                    f"  {variant:<9} {result.elapsed_s:6.1f}s  "
                    f"{result.output_tokens:3d} tokens  "
                    f"{result.tokens_per_second:5.2f} tok/s  "
                    f"{result.output_words:3d} words"
                )
            _write_payload(payload, output_path, csv_path)
        _write_payload(payload, output_path, csv_path)

    say(f"\nSaved: {output_path}")
    if csv_path:
        say(f"Saved: {csv_path}")
    return payload


def _write_payload(payload: dict, output_path: Path, csv_path: str | Path | None) -> None:
    import json

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_path:
        variants = payload["meta"]["settings"]["variants"]
        rows = flatten_runs(payload["runs"], variants)
        if rows:
            with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
