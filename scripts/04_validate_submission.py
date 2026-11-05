"""Verify that every must-have feature has evidence on disk."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    ADAPTER_DIR,
    MERGED_JSON,
    RESULTS_JSON,
    RESULTS_MD,
    RUNS_LOG,
    TIMINGS_PNG,
    VOTES_LOG,
    configure_console,
)
from src.prompts import PROMPTS, domain_prompts, out_of_domain_prompts

CHECKS: list[tuple[str, bool, str, bool]] = []


def check(name: str, condition: bool, evidence: str, optional: bool = False) -> None:
    CHECKS.append((name, bool(condition), evidence, optional))


def main() -> int:
    configure_console()
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    check("Streamlit app exists", (ROOT / "app.py").exists(), "app.py")
    check("Side-by-side layout uses st.columns", "st.columns" in app, "app.py -> st.columns")
    check("Both variants are labelled", 'VARIANT_LABELS' in app and "Fine-Tuned" in app, "src/config.py VARIANT_LABELS")
    check(
        "Timing is captured and displayed",
        "elapsed_s" in app and "compare_timings" in app,
        "app.py -> compare_timings / elapsed_s",
    )

    from src.engine import ComparisonEngine

    for method in ("generate_base", "generate_finetuned", "ensure_merged"):
        check(
            f"Engine exposes {method}()",
            hasattr(ComparisonEngine, method),
            f"src/engine.py -> {method}",
        )
    engine_source = (ROOT / "src" / "engine.py").read_text(encoding="utf-8")
    check(
        "Adapter can be disabled on the same base model",
        "disable_adapter" in engine_source,
        "src/engine.py -> PeftModel.disable_adapter()",
    )
    check(
        "Bonus merged model is implemented",
        "merge_and_unload" in engine_source,
        "src/engine.py -> merge_and_unload()",
    )

    adapter_ok = all(
        (ADAPTER_DIR / name).exists()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    )
    check("QLoRA adapter files are present", adapter_ok, str(ADAPTER_DIR))

    domain = domain_prompts()
    ood = out_of_domain_prompts()
    check("At least 3 in-domain test prompts", len(domain) >= 3, f"{len(domain)} in-domain prompts")
    check("One hard out-of-domain prompt", len(ood) >= 1, f"{len(ood)} out-of-domain prompt")
    check("Prompts are unique", len({p["id"] for p in PROMPTS}) == len(PROMPTS), "src/prompts.py")

    if RESULTS_JSON.exists():
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        runs = payload.get("runs", [])
        complete = [
            run
            for run in runs
            if "base" in run["results"]
            and "finetuned" in run["results"]
            and not run["results"]["base"].get("error")
            and not run["results"]["finetuned"].get("error")
            and run["results"]["base"].get("elapsed_s", 0) > 0
            and run["results"]["finetuned"].get("elapsed_s", 0) > 0
        ]
        check("Benchmark results exist", len(runs) >= 5, f"{len(runs)} runs in results.json")
        check(
            "Every prompt ran through both variants",
            len(complete) == len(runs) and len(runs) >= 5,
            f"{len(complete)}/{len(runs)} complete runs",
        )
    else:
        check("Benchmark results exist", False, f"missing {RESULTS_JSON}")

    check("Generated results tables exist", RESULTS_MD.exists(), str(RESULTS_MD))
    check("Timing chart exists", TIMINGS_PNG.exists(), str(TIMINGS_PNG))
    check(
        "Bonus merged benchmark exists",
        MERGED_JSON.exists(),
        str(MERGED_JSON),
        optional=True,
    )

    report = ROOT / "reports" / "REPORT.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        for keyword in ("Tone", "Format", "Accuracy", "Task-following", "Limitations"):
            check(f"Report covers {keyword}", keyword.lower() in text.lower(), "reports/REPORT.md")
    else:
        check("Written analysis exists", False, "reports/REPORT.md")

    votes_ok = VOTES_LOG.exists() and VOTES_LOG.stat().st_size > 0
    check("Vote logging wired up (bonus)", votes_ok, str(VOTES_LOG), optional=True)
    check("Run logging wired up", RUNS_LOG.exists(), str(RUNS_LOG), optional=True)

    print("=" * 74)
    print("SUBMISSION CHECK")
    print("=" * 74)
    failed = 0
    optional_failed = 0
    for name, passed, evidence, optional in CHECKS:
        status = "PASS" if passed else ("SKIP" if optional else "FAIL")
        if not passed:
            if optional:
                optional_failed += 1
            else:
                failed += 1
        print(f"[{status}] {name:<52} {evidence}")
    print("-" * 74)
    required = sum(1 for _, _, _, optional in CHECKS if not optional)
    print(
        f"{required - failed}/{required} required checks passed"
        + (f" · {optional_failed} optional bonus not recorded yet" if optional_failed else "")
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
