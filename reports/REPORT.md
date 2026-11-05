# Base vs Fine-Tuned — Qwen2.5-1.5B Python Tutor

**Project:** a comparison panel that sends the same prompt to the base
`Qwen/Qwen2.5-1.5B-Instruct` model and to the same model with the mandatory
QLoRA adapter attached, then measures and displays both answers side by side.

**Question:** what did the fine-tuning actually change — tone, format, accuracy,
task-following — and is it worth the extra adapter load and inference cost?

---

## 1. Executive summary

- The adapter is **cheap to attach**: 2.09 MB, 544,768 parameters (0.035% of the
  1.54 B base weights), **0.17 s** on top of the base load.
- The adapter is **not free at inference**: on the three full-length prompts it
  was **10.4% slower** (43.1 s vs 39.0 s average) and produced
  **2.98 tok/s vs 3.29 tok/s**. The difference is small relative to the 40 s
  answer time, so the decision should be about quality, not speed.
- Fine-tuning produced **one clear win**: the `while`-loop task. The adapter
  answered with a complete, runnable program and short inline comments; the base
  model was still explaining the concept when the 128-token budget ran out.
- Fine-tuning produced **two clear regressions**: the quiz became a single
  question with meaningless options, and the answer-checking reply collapsed to
  the one word **"Düzdür."** — a confident false confirmation.
- Fine-tuning **did not improve reasoning**: both variants answered the
  out-of-domain marathon trick question with the classic wrong answer
  ("third place" instead of "second").
- Verdict: with only 65 training examples, the adapter changed **style more than
  competence**. It made the model terser and more code-block-oriented, but it
  did not make it a more reliable Python tutor, and on two prompts it made the
  answer worse. The extra load cost is negligible; the extra quality is not
  reliable enough to justify switching by default.

---

## 2. Models and adapter

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (Qwen2ForCausalLM, 28 layers, 1,544,259,072 parameters) |
| Adapter | `adapters/qwen25-python-tutor` (LoRA, `r=4`, `alpha=8`, `dropout=0.05`, targets `q_proj` + `v_proj`, PEFT 0.21.2) |
| Adapter size | 2.09 MB |
| Trainable parameters | 544,768 (0.035% of the base) |
| Training data | 65 Azerbaijani Python-tutor examples (`data/train.jsonl`, seed 42) |
| Training regime (mandatory task) | 4-bit NF4 QLoRA, 3 epochs, max sequence 256, gradient accumulation 4, FP16 |
| Inference dtype here | `bfloat16` (CPU); `float16` automatically on CUDA |

Both variants use the **same tokenizer**, the **same chat template** (loaded from
the adapter directory), the same precision and the same decoding parameters.
The only experimental variable is whether the LoRA weights are active.

---

## 3. Method

### 3.1 One base model, two callables

The base model is loaded once and wrapped with `PeftModel`:

```python
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, ...)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

# Fine-tuned answer
model.generate(...)

# Base answer — identical weights, adapter switched off
with model.disable_adapter():
    model.generate(...)
```

This is the `disable_adapter()` route recommended by the PEFT documentation. It
is preferable to loading two full copies because both variants then share exactly
the same weights, tokenizer state and memory layout. The panel exposes them as
two independent methods, `generate_base()` and `generate_finetuned()`.

### 3.2 Fair timing

- Both variants run **sequentially**, never concurrently: on a CPU, two
  simultaneous generations would each slow down and the comparison would be
  meaningless.
- One warm-up generation per variant runs at startup, so lazy kernel and
  allocator initialisation are excluded from the reported numbers.
- Decoding is **greedy** (`do_sample=False`) by default, so both variants are
  deterministic and differences come from the weights, not from sampling.
- `elapsed_s` is wall-clock from the start of `generate()` to the last token,
  including prompt prefill. `tokens_per_second` is output tokens divided by that
  wall-clock, so it includes prefill overhead.
- The Streamlit UI additionally records **time to first token (TTFT)** while
  streaming. The offline benchmark uses non-streaming generation, so its TTFT
  field is empty.

### 3.3 Hardware

| Component | Value |
|---|---|
| CPU | Intel Core i5-11400H, 6 cores / 12 threads |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3050 Laptop 4 GB — not used |
| PyTorch | 2.12.0+cpu |
| Threads | 12 |

The installed PyTorch build is CPU-only, so the measurements reflect CPU
inference at roughly 3 tokens/second. On a CUDA build the same code moves the
model to the GPU automatically. Absolute seconds are hardware-specific; the
**relative** base-vs-adapter comparison is the transferable result.

---

## 4. Test prompts

`src/prompts.py` defines five prompts. Four are in the fine-tuning domain
(Azerbaijani Python tutoring); one is deliberately outside it.

| ID | Type | Prompt |
|---|---|---|
| `explain` | in-domain | "Python-da list və tuple arasındakı fərqi yeni başlayan üçün izah et. Qısa kod nümunəsi ver." |
| `code` | in-domain | "while döngüsü ilə 1-dən 5-ə qədər ədədləri çap edən Python kodu yaz və hər sətri qısa izah et." |
| `quiz` | in-domain | "Python siyahıları mövzusunda 3 suallıq quiz yarat və cavab açarını sonda göstər." |
| `check` | in-domain | "Cavabımı yoxla: `[3, 7, 8][1]` nəticəsi 8-dir. Düzdür?" |
| `hard` | out-of-domain | "Bir marafonda ikinci yerdə gedən idmançını keçdin. İndi neçənci yerdəsən? Qısa cavab ver və səbəbini bir cümlə ilə izah et." |

All five were run through both variants with `max_new_tokens=128`,
`temperature=0`. Every answer in this report is the **first and only** sample —
there was no cherry-picking.

The `check` prompt has a definite correct answer: `[3, 7, 8][1]` is `7`, so the
student's claim that it is `8` is **wrong**.

The `hard` prompt has a definite correct answer: overtaking the runner in second
place puts you in **second** place, not third.

---

## 5. Timing results

### 5.1 Per prompt

| Prompt | Base | Fine-Tuned | Difference | Base tok/s | FT tok/s |
|---|---:|---:|---:|---:|---:|
| Explain | 40.8 s | 42.4 s | +1.6 s (+4.0%) | 3.16 | 3.02 |
| Code | 38.3 s | 47.3 s | **+9.0 s (+23.4%)** | 3.34 | 2.68 |
| Quiz | 38.0 s | 39.5 s | +1.5 s (+4.1%) | 3.37 | 3.24 |
| Check | 17.3 s | 9.4 s | −8.0 s (−46.0%) | 2.31 | 0.53 |
| Hard | 15.1 s | 15.9 s | +0.8 s (+5.1%) | 1.98 | 2.14 |

The `check` row is **not** evidence that the adapter is faster. The fine-tuned
model emitted only 5 tokens and hit its end-of-sequence token, while the base
emitted 40. A shorter answer is cheaper to produce; the per-token speed was
actually lower (0.53 tok/s, dominated by the fixed prefill cost).

### 5.2 Aggregate

| Metric | Base | Fine-Tuned | Change |
|---|---:|---:|---:|
| Average time, 3 full-length prompts | 39.0 s | 43.1 s | +10.4% |
| Average decode speed, 3 full-length prompts | 3.29 tok/s | 2.98 tok/s | −9.5% |
| Average output tokens, all 5 prompts | 91.0 | 84.4 | −7.3% |
| Average output words, all 5 prompts | 32.2 | 26.0 | −19.3% |

### 5.3 Reading the timing

The LoRA layers add two low-rank matrix multiplications (`q_proj`, `v_proj`) in
every one of the 28 transformer layers. On this CPU that is consistent with the
observed **~5–10% decode overhead**. The 23.4% gap on the `code` prompt is
larger than the others and is best treated as run-to-run noise plus a different
number of generated tokens, not as a stable property of the adapter.

The load side is unambiguous: the adapter adds **0.17 s** and **2.09 MB** on top
of a base load that took **0.49 s** with the weights already in the OS file
cache. Loading the base weights without a warm cache takes far longer than
0.17 s, so the adapter attach step is a negligible fraction of startup either
way. Against a 40 s generation, it is noise.

**Conclusion on cost:** the adapter is cheap enough that it should be judged
only on answer quality. That is the subject of the next section.

---

## 6. What the fine-tuning changed

### 6.1 Format — the clearest, most consistent change

The adapter is much more likely to answer **code-first**. On `explain`, the base
model wrote two sentences of prose and then a code block. The fine-tuned model
opened directly with a `python` code block and put its entire answer inside
comments:

```python
# Listaların əsasında:
# 1. Dəyərlərdən ibarətdir.
# 2. İndeksli olmayan dəyişikliklər qalır.
...
```

On `code`, the adapter produced a self-contained, runnable snippet with inline
comments:

```python
i = 1 # başlangıç qiyməti
while i <= 5:
    print(i) # 1-ci qiymət
    i += 1 # i yi artır
```

This is a real behavioural change: the fine-tuning data was chat-style tutoring
with code examples, and the adapter learned the **shape** of that answer —
heading, code block, short comments.

It is also visible in the token/word ratio. On `explain` both variants used
roughly 128 tokens, but the base used 53 words and the adapter only 34: the
adapter spends a larger share of its budget on formatting tokens (newlines,
`#`, backticks) rather than prose.

### 6.2 Tone — shorter, flatter, more template-like

The base model is conversational: on `quiz` it opens with
"Tabii, burada sizin istifadəçinin seçdiksiyini göstərmək üçün..."
(ungrammatical, but chatty). The adapter opens with a bare numbered list and no
framing.

The base model also mixes in Turkish forms ("Soru", "doğru", "kullanır"). The
adapter is more consistently Azerbaijani in its function words, which is one of
the few measurable benefits of the domain data — but it is a surface-level gain.

### 6.3 Accuracy — a mixed result, with two regressions

**`explain` (base better).** The base answer is vague and gets cut off, but its
code samples are syntactically valid and its claims about lists and tuples are
mostly right. The adapter's comments contain statements that are not true or not
meaningful, e.g. "İndeksli olmayan dəyişikliklər qalır" ("changes that are not
indexed remain"). The response looks like a tutorial but does not teach the
actual difference.

**`code` (adapter better).** The adapter delivered the complete task: a correct
`while` loop, the right comparison (`i <= 5`), the increment, and an explanation
of each line. The base model spent its budget on an introduction and its code
block is cut off mid-line (`print(i`). For a beginner who wants runnable code,
the adapter's answer is the more useful one.

**`quiz` (both fail; base closer).** The base model produced a three-question
format, but the questions are about functions rather than lists, and it ran out
of budget before showing the answer key. The adapter produced one question and
options that are not about Python at all:

```
1) Python dili nədir?
   a) Aşağıdakı bütün cümlələri təsir edin.
   b) Cümlənin başqa bir cümləsi olub-olmadığına görə.
```

This is a **regression**: the adapter learned the visual template of a quiz
(numbered item, lettered options) without the content. It also ignored two
explicit instructions — "3 questions" and "answer key at the end".

**`check` (both fail; adapter worse pedagogically).** The correct answer is
"wrong, the result is 7". The base model says:

> "Düzdür. `[3, 7, 8][1]` metodun cümlənin əsasında 7-ci elementi döndürür."

It confirms the student ("Düzdür" = "correct") and in the same breath states
that index 1 returns the 7th element. The answer is self-contradictory and
wrong, but at least the correct value appears in the text.

The adapter says only:

> "Düzdür."

This is **confidently wrong and gives the student no way to notice the error**.
In a tutoring context, a one-word false confirmation is worse than a rambling
answer that contains the correct number.

### 6.4 Task-following — template yes, constraints no

The adapter follows *format* instructions well (code block, numbered options,
short verdict). It does not follow *content constraints*: it ignored "3
questions" and "answer key" in the quiz, and it did not check the arithmetic in
the `check` prompt before agreeing.

The base model is the opposite: it is more willing to explain and to produce
several items, but it drifts off-topic and is easily cut off by the token
budget.

### 6.5 Hard out-of-domain prompt — no reasoning gain

Both variants fell for the classic trick:

| Variant | Answer |
|---|---|
| Base | "İndi üçüncü yerdədəsin. Marafa üçün üçüncü yerdən geyinmişsin." |
| Fine-Tuned | "İndi üçüncü yerdədəsən. Marafa 3-cü yerdə gedən idmançın keçdi." |

Correct answer: **second**. Overtaking the runner in second place means you take
that position.

The adapter's sentence is more grammatical, but its reasoning is just as wrong:
it says you passed the athlete who was in third place, which is not what the
prompt describes. The base model adds a nonsensical clause
("geyinmişsin" — "you got dressed"). This is the clearest evidence that the
fine-tuning **did not transfer any reasoning ability**; it changed the surface
style of the answer while leaving the underlying failure intact.

---

## 7. Is the adapter worth it?

| Dimension | Verdict |
|---|---|
| Load cost | **Yes, negligible** — 0.17 s and 2.09 MB |
| Inference cost | **Small** — roughly 5–10% slower decode on full-length answers |
| Format | **Improved** — code-first, structured, consistent |
| Tone | **Changed** — terser, less chatty, more Azerbaijani |
| Accuracy | **Mixed** — one win (`code`), two regressions (`quiz`, `check`), one tie (`explain`) |
| Reasoning | **No change** — both fail the out-of-domain trick |
| Reliability | **Worse** — the adapter is more confidently wrong on the `check` prompt |

With 65 training examples, this is the expected outcome: the adapter can learn
answer *style* but not enough content coverage to be a better tutor. A larger,
more diverse dataset and an evaluation set with a correct-answer check would be
needed before deploying it over the base model.

**Practical recommendation:** keep the adapter for the code-generation use case,
where its structured, complete snippets are clearly better. Do not use it as the
default for explanation, quiz generation or answer checking without a human
review step.

---

## 8. Bonus: merged model (`merge_and_unload`)

A third, optional variant loads a **separate copy** of the base, attaches the
adapter and calls `merge_and_unload()`. This bakes the LoRA delta into the base
weights, so generation no longer runs the adapter's low-rank matmuls.

| Prompt | Base | Fine-Tuned (adapter) | Merged | Merged vs FT |
|---|---:|---:|---:|---:|
| Explain | 46.2 s / 129 tok | 50.6 s / 128 tok | **42.6 s / 128 tok** | −7.9 s (−15.7%) |
| Quiz | 39.7 s / 128 tok | 35.5 s / 128 tok | **33.8 s / 128 tok** | −1.7 s (−4.8%) |

**Build cost:** 1.3 s with the base weights already in the OS file cache. A cold
start adds the normal ~10 s base load, because merging requires a second,
independent copy of the model in memory. In the UI this variant is therefore
opt-in — it needs roughly 3 GB of extra RAM next to the already-loaded
adapter-attached model.

**Speed:** the merged model was faster than the adapter-attached model on both
prompts, which is consistent with removing two low-rank matrix multiplications
per layer. The effect is small and the run-to-run variance on this CPU is large
(the fine-tuned explain time was 42.4 s in the main benchmark and 50.6 s here),
so the honest summary is: **merging does not make inference slower, and may make
it a few percent faster; the real cost is the extra load time and memory.**

**Output equivalence — an unexpected detail.** In exact arithmetic, merging then
generating should produce the same logits as keeping the adapter separate. In
`bfloat16` it does not: the merged run and the adapter run diverge after a few
tokens (text similarity 0.88 on `explain`, 0.20 on `quiz`). Merging rounds the
LoRA delta into the base weights, and greedy decoding amplifies the tiny
numerical differences into different word choices. This is worth knowing if a
project relies on a merged checkpoint as a "bit-exact" reference: it is
mathematically equivalent, not numerically identical.

Representative divergence on `explain`:

| Variant | Line 3 |
|---|---|
| Fine-Tuned | `# 3. Ekspektif bir dəyər olmayan dəyişikliklər qalır.` |
| Merged | `# 3. Ekspektif bir dəyəri artırmaq üçün eklənməz.` |

Both are meaningless, which is itself consistent with the accuracy finding in
§6.3 — the merge changes the wording, not the competence.

---

## 9. Bonus: voting and logging

The Streamlit panel lets the reader vote **Base / Fine-Tuned / Tie** after each
comparison. Votes and runs are appended as JSONL:

- `logs/runs.jsonl` — prompt, settings, both responses, timings.
- `logs/votes.jsonl` — winner, prompt, timings, timestamp.

The mechanism is wired into the UI and covered by the unit tests; the vote
summary is rendered in `reports/RESULTS.md` once votes exist. The offline
benchmark does not cast votes — voting is a human judgement step.

One vote is recorded in `logs/votes.jsonl`: the automated UI smoke test
(`scripts/05_ui_smoke_test.py`) clicked **Base** for the `explain` prompt. That
matches the manual reading in §6.3 — the adapter's explanation is less accurate —
but it should be treated as a smoke-test entry, not as a user study. The panel is
ready for real reviewers to add their own votes.

Each vote record contains the run id, prompt, winner, both response times and a
timestamp, so votes can be joined back to the full responses in
`logs/runs.jsonl`.

---

## 10. Limitations

1. **Small adapter training set.** 65 examples cannot cover Python tutoring;
   the quiz regression is consistent with memorising templates instead of
   content.
2. **One sample per prompt.** Greedy decoding makes each run deterministic, but
   a single sample cannot estimate how often each failure occurs. The
   observations are qualitative, not a benchmark score.
3. **CPU-only timings.** PyTorch 2.12.0+cpu on an i5-11400H. Absolute seconds
   will differ on other hardware; the relative overhead is the meaningful part.
4. **Wall-clock includes prefill.** `tokens_per_second` therefore understates
   pure decode speed, especially on short answers (see the `check` row).
5. **The offline benchmark has no TTFT.** TTFT is measured only in the streaming
   UI.
6. **bfloat16 on CPU.** This halves memory versus float32 and matched float32
   speed in a spot check, but it is not bit-identical to the fp16 GPU inference
   used during the mandatory training.
7. **No automated correctness metric.** The five prompts were judged by reading
   the answers; the `check` and `hard` prompts have objective answers, the other
   three require judgement.
8. **Merged model is equivalent by construction.** It is expected to produce the
   same text as the adapter-attached model; the bonus measures cost, not
   quality.

---

## 11. Reproducibility

```powershell
cd C:\Users\User\base-vs-finetuned-panel
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

.\run.ps1 scripts\00_check_env.py                     # packages, adapter, cache, prompts
.\run.ps1 scripts\01_run_benchmark.py --max-new-tokens 128
.\run.ps1 scripts\02_merged_benchmark.py              # bonus merged model
.\run.ps1 scripts\03_make_report.py                   # RESULTS.md + timings.png
.\run.ps1 scripts\04_validate_submission.py
.\run.ps1 -m unittest discover -s tests -v            # 18 tests
```

Outputs:

| File | Content |
|---|---|
| `reports/results.json` | every prompt, both variants, timings, full text, load info |
| `reports/results.csv` | flat table for spreadsheets |
| `reports/merged_benchmark.json` | merged build time and three-variant timings |
| `reports/RESULTS.md` | generated tables and full responses |
| `reports/timings.png` | response-time chart |
| `logs/runs.jsonl` | runs produced through the UI |
| `logs/votes.jsonl` | human votes |

---

## 12. References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &
  Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR
  2022*. <https://arxiv.org/abs/2106.09685>
- Hugging Face PEFT documentation — adapter loading and `disable_adapter()`:
  <https://huggingface.co/docs/peft>
- Qwen Team (2024). Qwen2.5: A Party of Foundation Models.
  <https://qwenlm.github.io/blog/qwen2.5/>
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA:
  Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.
  <https://arxiv.org/abs/2305.14314>
