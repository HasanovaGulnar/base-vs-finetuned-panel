"""Fast unit tests that never load the 1.5B model."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import (
    char_count,
    compare_timings,
    flatten_runs,
    markdown_table,
    summarize_runs,
    summarize_variant,
    word_count,
)
from src.logging_utils import append_jsonl, new_run_id, read_jsonl
from src.prompts import PROMPTS, domain_prompts, get_prompt, out_of_domain_prompts


class PromptTests(unittest.TestCase):
    def test_at_least_three_domain_prompts(self) -> None:
        self.assertGreaterEqual(len(domain_prompts()), 3)

    def test_one_out_of_domain_prompt(self) -> None:
        self.assertEqual(len(out_of_domain_prompts()), 1)

    def test_prompt_fields(self) -> None:
        for prompt in PROMPTS:
            self.assertTrue(prompt["system"].strip())
            self.assertTrue(prompt["user"].strip())
            self.assertIn(prompt["domain"], {"in-domain", "out-of-domain"})

    def test_ids_unique(self) -> None:
        ids = [prompt["id"] for prompt in PROMPTS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_get_prompt(self) -> None:
        self.assertEqual(get_prompt("quiz")["id"], "quiz")
        with self.assertRaises(KeyError):
            get_prompt("missing")


class AnalysisTests(unittest.TestCase):
    def test_compare_timings(self) -> None:
        result = compare_timings(40.0, 42.0)
        self.assertEqual(result["diff_s"], 2.0)
        self.assertEqual(result["diff_pct"], 5.0)
        self.assertEqual(result["faster"], "base")
        self.assertEqual(compare_timings(10.0, 10.0)["faster"], "tie")

    def test_word_and_char_count(self) -> None:
        self.assertEqual(word_count("salam dünya"), 2)
        self.assertEqual(word_count(""), 0)
        self.assertEqual(char_count("abc"), 3)
        self.assertEqual(char_count(""), 0)

    def test_summarize_variant(self) -> None:
        results = [
            {"base": {"elapsed_s": 2.0, "tokens_per_second": 5.0, "output_tokens": 10, "output_words": 4, "ttft_s": 0.5}},
            {"base": {"elapsed_s": 4.0, "tokens_per_second": 3.0, "output_tokens": 12, "output_words": 5, "ttft_s": 0.7}},
        ]
        summary = summarize_variant(results, "base")
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["avg_seconds"], 3.0)
        self.assertEqual(summary["median_seconds"], 3.0)
        self.assertEqual(summary["avg_output_tokens"], 11.0)

    def test_summarize_skips_errors(self) -> None:
        results = [{"base": {"elapsed_s": 0.0, "tokens_per_second": 0.0, "output_tokens": 0, "output_words": 0, "ttft_s": None, "error": "boom"}}]
        self.assertEqual(summarize_variant(results, "base")["n"], 0)

    def test_flatten_runs(self) -> None:
        runs = [
            {
                "run_id": "r1",
                "prompt_id": "quiz",
                "title": "Quiz",
                "domain": "in-domain",
                "results": {
                    "base": {"elapsed_s": 1.0, "ttft_s": 0.2, "prompt_tokens": 5, "output_tokens": 7, "output_words": 3, "output_chars": 20, "tokens_per_second": 7.0, "max_new_tokens": 32, "temperature": 0.0},
                },
            }
        ]
        rows = flatten_runs(runs, ["base", "finetuned"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant"], "base")

    def test_summarize_runs(self) -> None:
        runs = [
            {
                "results": {
                    "base": {"elapsed_s": 1.0, "tokens_per_second": 1.0, "output_tokens": 1, "output_words": 1, "ttft_s": 0.1},
                    "finetuned": {"elapsed_s": 2.0, "tokens_per_second": 1.0, "output_tokens": 2, "output_words": 1, "ttft_s": 0.2},
                }
            }
        ]
        summary = summarize_runs(runs, ["base", "finetuned"])
        self.assertEqual([row["variant"] for row in summary], ["base", "finetuned"])

    def test_markdown_table(self) -> None:
        table = markdown_table(["A", "B"], [[1, 2]])
        self.assertIn("| A | B |", table)
        self.assertIn("| 1 | 2 |", table)


class LoggingTests(unittest.TestCase):
    def test_append_and_read_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "votes.jsonl"
            append_jsonl(path, {"vote": "base", "note": "Azərbaycan"})
            append_jsonl(path, {"vote": "tie"})
            records = read_jsonl(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["note"], "Azərbaycan")

    def test_read_missing_file(self) -> None:
        self.assertEqual(read_jsonl(Path("does") / "not" / "exist.jsonl"), [])

    def test_run_ids_unique(self) -> None:
        ids = {new_run_id() for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_jsonl_is_one_json_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            append_jsonl(path, {"a": 1})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0]), {"a": 1})


class EngineApiTests(unittest.TestCase):
    def test_required_methods_exist(self) -> None:
        from src.engine import ComparisonEngine

        for method in ("generate", "generate_base", "generate_finetuned", "ensure_merged"):
            self.assertTrue(hasattr(ComparisonEngine, method), method)

    def test_format_seconds(self) -> None:
        from src.engine import format_seconds

        self.assertEqual(format_seconds(None), "—")
        self.assertEqual(format_seconds(0.25), "250 ms")
        self.assertEqual(format_seconds(2.5), "2.5 s")


if __name__ == "__main__":
    unittest.main()
