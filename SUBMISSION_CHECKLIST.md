# Submission Checklist

## Must-have features

| # | Requirement | Evidence | Status |
|---|---|---|---|
| 1 | Base and fine-tuned variants running and independently callable | `src/engine.py`: `generate_base()` uses `model.disable_adapter()`, `generate_finetuned()` uses the adapter-active model | ✅ |
| 2 | Identical prompt dispatch | `app.py`: one button sends the same `system` + `user` messages to every selected variant | ✅ |
| 3 | Side-by-side display labelled Base / Fine-Tuned | `app.py`: `st.columns` + `VARIANT_LABELS` from `src/config.py` | ✅ |
| 4 | Response time measured and displayed per variant | `elapsed_s`, `ttft_s`, `tokens_per_second` in `src/engine.py`; delta metric in `app.py` | ✅ |
| 5 | 3+ domain test prompts documented | `src/prompts.py`: 4 in-domain prompts (explain, code, quiz, check) + 1 hard prompt; every prompt is benchmarked in `reports/results.json` | ✅ |
| 6 | Written analysis of behaviour, tone, format, accuracy | `reports/REPORT.md` with concrete quotes from the runs | ✅ |

## Bonus features

| Bonus | Evidence | Status |
|---|---|---|
| Vote which answer is better + log | `app.py` vote buttons → `logs/votes.jsonl`; summary in `reports/RESULTS.md` | ✅ |
| Token-count / response-length comparison | tokens, words, characters, tokens/s per variant in `reports/results.csv` and the UI | ✅ |
| Hard out-of-domain prompt | `hard` prompt in `src/prompts.py`; analysed in `reports/REPORT.md` | ✅ |
| Third column with merged (`merge_and_unload`) model | sidebar checkbox + `scripts/02_merged_benchmark.py` → `reports/merged_benchmark.json` | ✅ |

## Technical requirements

| Requirement | Evidence |
|---|---|
| Hugging Face `transformers` + `peft` | `src/engine.py` uses `AutoModelForCausalLM`, `AutoTokenizer`, `PeftModel`, `disable_adapter()`, `merge_and_unload()` |
| Streamlit columns layout | `app.py` → `st.columns(len(variants))` |
| Reuses the mandatory fine-tuning project | Same base model (`Qwen/Qwen2.5-1.5B-Instruct`), same adapter (`adapters/qwen25-python-tutor`), same 72-example dataset copied into `data/` |
| No unrelated new stack | Only `transformers`, `peft`, `accelerate`, `streamlit`, `matplotlib` |

## Reproducibility evidence

```powershell
.\run.ps1 scripts\00_check_env.py
.\run.ps1 scripts\01_run_benchmark.py --max-new-tokens 128
.\run.ps1 scripts\02_merged_benchmark.py
.\run.ps1 scripts\03_make_report.py
.\run.ps1 scripts\04_validate_submission.py
.\run.ps1 -m unittest discover -s tests -v
```

- `reports/results.json` records every prompt, both variants, timings and full text.
- `reports/results.csv` is the flat table used for the analysis.
- `reports/merged_benchmark.json` records the merged-model build time and speed.
- `reports/timings.png` visualises the per-prompt timing comparison.
- `logs/runs.jsonl` and `logs/votes.jsonl` are the runtime logs written by the app.

## Honest limitations

1. The adapter was trained on 65 examples; the behavioural differences are those
   of a small fine-tuning run.
2. One greedy generation per prompt — no variance estimate.
3. Timings are CPU-only (PyTorch 2.12.0+cpu); absolute seconds are specific to
   this laptop, the relative comparison is the useful part.
4. The merged model is mathematically equivalent to the adapter-attached model,
   so it is expected to produce identical text; the bonus compares load/speed.
