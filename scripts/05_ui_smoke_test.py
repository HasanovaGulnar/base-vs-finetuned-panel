"""Headless end-to-end test of the Streamlit panel using Streamlit's AppTest.

It loads the real model, runs one short comparison, checks the side-by-side
layout and the timing block, then clicks a vote button and verifies that the
vote reached logs/votes.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from src.config import VOTES_LOG, configure_console
from src.logging_utils import read_jsonl

CAST_VOTE = os.environ.get("UI_SMOKE_VOTE", "1") != "0"


def _button(at: AppTest, needle: str):
    matches = [button for button in at.button if needle in button.label]
    if not matches:
        labels = [button.label for button in at.button]
        raise AssertionError(f"No button containing {needle!r}. Buttons: {labels}")
    return matches[0]


def main() -> int:
    configure_console()
    before = len(read_jsonl(VOTES_LOG))

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    if at.exception:
        for exception in at.exception:
            print("APP EXCEPTION:", exception.value)
        return 1

    # Keep the smoke test short: 32 new tokens per variant.
    at.slider[0].set_value(32)
    at.run()

    _button(at, "göndər").click()
    at.run(timeout=600)
    if at.exception:
        for exception in at.exception:
            print("RUN EXCEPTION:", exception.value)
        return 1

    state = at.session_state["last_run"]
    results = state["results"]
    assert "base" in results and "finetuned" in results, results.keys()
    assert results["base"]["output_tokens"] > 0, results["base"]
    assert results["finetuned"]["output_tokens"] > 0, results["finetuned"]
    assert results["base"]["elapsed_s"] > 0 and results["finetuned"]["elapsed_s"] > 0

    markdown = "\n".join(element.value for element in at.markdown)
    for label in ("Base", "Fine-Tuned"):
        assert label in markdown, f"{label} heading missing from the rendered page"

    # Vote through the UI and confirm the JSONL log grew.
    if CAST_VOTE:
        _button(at, "Base daha yaxşı").click()
        at.run(timeout=600)
        if at.exception:
            for exception in at.exception:
                print("VOTE EXCEPTION:", exception.value)
            return 1

        votes = read_jsonl(VOTES_LOG)
        assert len(votes) == before + 1, f"vote log grew from {before} to {len(votes)}"
        assert votes[-1]["run_id"] == state["run_id"]
        assert at.session_state["last_run"].get("vote") == "base"
        vote_value = votes[-1]["vote"]
    else:
        vote_value = None

    history = {
        "base_s": results["base"]["elapsed_s"],
        "finetuned_s": results["finetuned"]["elapsed_s"],
        "base_tokens": results["base"]["output_tokens"],
        "finetuned_tokens": results["finetuned"]["output_tokens"],
        "vote": vote_value,
    }
    print("UI smoke test passed")
    print(json.dumps(history, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
