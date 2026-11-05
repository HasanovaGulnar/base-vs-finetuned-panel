# Base vs Fine-Tuned — Qwen2.5-1.5B Python Tutor

A Streamlit panel that sends **one prompt to two variants of the same model** and
shows the answers side by side with measured response times:

| Column | What runs |
|---|---|
| **Base** | `Qwen/Qwen2.5-1.5B-Instruct` with the LoRA adapter disabled (`PeftModel.disable_adapter()`) |
| **Fine-Tuned** | The *same* base weights with `adapters/qwen25-python-tutor` active |
| **Merged** (optional bonus) | A separate copy of the base with `merge_and_unload()` applied |

Both required variants share one copy of the base weights, the same tokenizer,
the same precision and the same decoding settings, so the only difference is the
adapter. That makes the timing and behaviour comparison a genuine A/B test
instead of a comparison of two differently configured pipelines.

---

## Headline results

Measured on 5 prompts (4 in-domain + 1 hard), greedy decoding, CPU bfloat16.
Full tables and all responses: [`reports/RESULTS.md`](reports/RESULTS.md).
Analysis: [`reports/REPORT.md`](reports/REPORT.md).

| Metric | Base | Fine-Tuned |
|---|---:|---:|
| Average response time (5 prompts) | 29.9 s | 30.9 s |
| Average response time (3 full-length prompts) | 39.0 s | 43.1 s |
| Average decode speed | 2.83 tok/s | 2.32 tok/s |
| Average output tokens | 91.0 | 84.4 |
| Average output words | 32.2 | 26.0 |
| Adapter load cost | — | 0.17 s / 2.09 MB |

**Short version:** the adapter costs almost nothing to attach (0.17 s, 2.09 MB,
544,768 parameters = 0.035% of the base model) and about 5–10% in decode speed
on the full-length answers. Behaviourally it is a mixed result: it produced a
complete, runnable `while` loop for the code prompt where the base ran out of
budget mid-explanation, but it also produced a nonsensical quiz and a
one-word "Düzdür." for the answer-checking prompt, and neither variant solved
the out-of-domain trick question.

---

## Must-have features

| Requirement | Where it is delivered |
|---|---|
| Base and fine-tuned variants independently callable | `ComparisonEngine.generate_base()` / `generate_finetuned()` in `src/engine.py`; one shared base model, adapter toggled per call |
| Identical prompt dispatch | One button sends the same `system` + `user` messages to both variants sequentially |
| Side-by-side display, labelled "Base" / "Fine-Tuned" | `app.py` → `st.columns`, `VARIANT_LABELS` |
| Response-time measurement per variant | Wall-clock around `generate()` plus TTFT; shown as seconds, tokens/s and a delta metric |
| 3+ domain test prompts | 4 in-domain prompts (explain, code, quiz, check) + 1 hard out-of-domain prompt in `src/prompts.py` |
| Written analysis | [`reports/REPORT.md`](reports/REPORT.md), generated tables in [`reports/RESULTS.md`](reports/RESULTS.md) |

## Bonus features

| Bonus | Status |
|---|---|
| Vote which answer is better + log | ✅ Three buttons (Base / Fine-Tuned / Tie) appended to `logs/votes.jsonl` |
| Token-count / response-length comparison | ✅ prompt tokens, output tokens, words, characters, tokens/s per variant |
| One hard out-of-domain prompt | ✅ The marathon prompt (`id=hard`), analysed separately in the report |
| Third column with `merge_and_unload()` model | ✅ Optional checkbox in the sidebar; `scripts/02_merged_benchmark.py` measures build time and inference speed |

---

## Quick start

The project ships with a local `.venv` that inherits the Anaconda scientific
stack (`torch`, `transformers`, `streamlit`, `matplotlib`). `peft` and
`accelerate` are installed inside the venv.

```powershell
cd C:\Users\User\base-vs-finetuned-panel
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 1. verify environment, adapter and cached base model
.\run.ps1 scripts\00_check_env.py

# 2. run the comparison UI
.\run.ps1 -m streamlit run app.py

# 3. offline benchmark (5 prompts × 2 variants)
.\run.ps1 scripts\01_run_benchmark.py --max-new-tokens 128

# 4. bonus: merged model benchmark
.\run.ps1 scripts\02_merged_benchmark.py

# 5. rebuild tables and the timing chart
.\run.ps1 scripts\03_make_report.py

# 6. verify every deliverable
.\run.ps1 scripts\04_validate_submission.py
.\run.ps1 -m unittest discover -s tests -v
```

`run.ps1` sets `KMP_DUPLICATE_LIB_OK=TRUE` (two OpenMP runtimes in this Anaconda
install), `HF_HUB_OFFLINE=1` and forces UTF-8 output.

### Rebuilding the environment

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The base model must be in the Hugging Face cache for offline use:

```
C:\Users\User\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct
```

---

## How the comparison works

1. The base model is loaded once in `bfloat16` (CPU) or `float16` (CUDA).
2. `PeftModel.from_pretrained()` attaches the 2.09 MB adapter to that instance.
3. `generate_base()` runs inside `model.disable_adapter()`, so the same weights
   answer with the adapter switched off.
4. `generate_finetuned()` runs with the adapter active.
5. The two calls are made **sequentially** on purpose: running them at the same
   time on a CPU would make each one slower and the timing comparison
   meaningless.
6. One warm-up generation per variant runs at startup so the reported timings
   exclude lazy kernel/allocator initialisation.
7. Decoding is greedy (`do_sample=False`) by default, so both variants are
   deterministic and differences come from the weights, not from sampling.

### Timing methodology

| Metric | Meaning |
|---|---|
| `elapsed_s` | Wall-clock from the start of `generate()` to the last token, including prefill |
| `ttft_s` | Time to the first streamed token (prefill latency) |
| `tokens_per_second` | `output_tokens / elapsed_s` |
| `base_load_s` | Loading the base weights from the local cache |
| `adapter_attach_s` | `PeftModel.from_pretrained()` on top of the loaded base |
| `merged_build_s` | Loading a second base copy **and** merging the adapter into it |

---

## Project layout

```
base-vs-finetuned-panel/
├─ app.py                       # Streamlit comparison panel
├─ src/
│  ├─ config.py                 # paths, labels, defaults
│  ├─ prompts.py                # 4 in-domain + 1 out-of-domain prompt
│  ├─ engine.py                 # base / adapter / merged loading + timing
│  ├─ benchmark.py              # offline prompt sweep
│  ├─ analysis.py               # timing summaries and tables
│  └─ logging_utils.py          # JSONL run and vote logs
├─ scripts/
│  ├─ 00_check_env.py
│  ├─ 01_run_benchmark.py
│  ├─ 02_merged_benchmark.py
│  ├─ 03_make_report.py
│  └─ 04_validate_submission.py
├─ adapters/qwen25-python-tutor/   # the mandatory QLoRA adapter (r=4, alpha=8)
├─ data/                           # 72-example Azerbaijani Python-tutor dataset
├─ reference/                      # original mandatory app files, for provenance
├─ reports/
│  ├─ REPORT.md                 # written analysis
│  ├─ RESULTS.md                # generated tables
│  ├─ results.json / results.csv
│  ├─ merged_benchmark.json
│  └─ timings.png
├─ logs/
│  ├─ runs.jsonl                # every comparison run
│  └─ votes.jsonl               # every human vote
├─ tests/test_project.py        # 18 model-free tests
├─ run.ps1
└─ requirements.txt
```

---

## Hardware used for the measurements

The figures in `reports/` were produced on the machine that also holds the
mandatory project:

| Component | Value |
|---|---|
| CPU | Intel Core i5-11400H, 6 cores / 12 threads |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3050 Laptop, 4 GB (not used: the installed PyTorch build is CPU-only) |
| PyTorch | 2.12.0+cpu |
| Precision | `bfloat16` |

CPU inference is roughly **2.5–3 tokens/second**, so a 128-token answer takes
about 45 seconds. The app streams tokens so the user sees progress. If a CUDA
build of PyTorch is installed, the engine switches to the GPU automatically and
the same code runs several times faster.

---

## Limitations

- The adapter was trained on only 65 examples, so the comparison describes a
  small fine-tuning run, not a production model.
- The benchmark records one generation per prompt (greedy decoding); it does not
  estimate run-to-run variance.
- CPU timings are hardware-specific. The *relative* difference between variants
  is the meaningful number, not the absolute seconds.
- The merged model is mathematically equivalent to the adapter-attached model,
  so its output should be identical; the bonus compares load cost and speed, not
  quality.

Full discussion: [`reports/REPORT.md`](reports/REPORT.md).
