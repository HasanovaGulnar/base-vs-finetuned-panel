# Reference files from the mandatory task

These files are copied unchanged from
[`HasanovaGulnar/qwen-local-assistant`](https://github.com/HasanovaGulnar/qwen-local-assistant)
so this project is self-documenting and the comparison can be traced back to the
original app.

| File | What it is |
|---|---|
| `app.py` | The mandatory Streamlit chat app (4-bit QLoRA, CUDA-only) |
| `train_qlora.py` | The training script that produced the adapter |
| `compare_before_after.py` | The original GPU-only before/after comparison script |
| `before_after.txt` | The original example output recorded during the mandatory task |

They are **not imported** by the comparison panel. The panel re-implements the
comparison in `src/engine.py` so it can run on CPU without `bitsandbytes`, toggle
the adapter without reloading the base model, and measure timing per variant.

Why the original files are kept:

1. Provenance — the adapter settings (`r=4`, `alpha=8`, target modules
   `q_proj`/`v_proj`) can be checked against the training script.
2. The original `before_after.txt` is a useful reference point for how the
   adapter behaved under 4-bit GPU inference, which is different from the
   bfloat16 CPU inference used in this project.
