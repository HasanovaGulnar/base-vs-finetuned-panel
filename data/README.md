# Data

This folder is copied from the mandatory task repository
[`HasanovaGulnar/qwen-local-assistant`](https://github.com/HasanovaGulnar/qwen-local-assistant).

| File | Rows | Purpose |
|---|---:|---|
| `all_examples.jsonl` | 72 | Full Azerbaijani Python-tutor dataset |
| `train.jsonl` | 65 | Training split used for the QLoRA adapter (seed 42) |
| `validation.jsonl` | 7 | Validation split |

Each line is a chat-style JSON object with `messages` (system / user / assistant).
The dataset is small on purpose: it is a learning project, and its size is one of
the limitations discussed in `reports/REPORT.md`.

These files are **not** used at inference time. The comparison panel only needs
the adapter in `adapters/qwen25-python-tutor/`. The data is included so the
fine-tuning domain can be inspected without leaving the project.
