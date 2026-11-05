"""Base vs Fine-Tuned — Qwen2.5-1.5B Python tutor comparison panel."""

from __future__ import annotations

import json
import os

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st

from src.analysis import compare_timings
from src.config import (
    ADAPTER_DIR,
    DEFAULT_MAX_NEW_TOKENS,
    LOGS_DIR,
    RUNS_LOG,
    VARIANT_LABELS,
    VOTES_LOG,
    ensure_directories,
)
from src.engine import ComparisonEngine
from src.logging_utils import log_run, log_vote, new_run_id, read_jsonl
from src.prompts import PROMPTS, get_prompt

st.set_page_config(
    page_title="Base vs Fine-Tuned — Qwen2.5-1.5B",
    page_icon="⚖️",
    layout="wide",
)

ensure_directories()

if "runs" not in st.session_state:
    st.session_state.runs = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None


@st.cache_resource(show_spinner=False)
def get_engine() -> ComparisonEngine:
    return ComparisonEngine()


def _variant_metrics(result: dict) -> str:
    ttft = result.get("ttft_s")
    ttft_text = f" · ilk token {ttft:.2f}s" if ttft is not None else ""
    return (
        f"**{result['elapsed_s']:.1f} s** · "
        f"{result['output_tokens']} token · "
        f"{result['tokens_per_second']:.2f} tok/s{ttft_text}"
    )


def _render_results(record: dict) -> None:
    results = record["results"]
    variants = [v for v in VARIANT_LABELS if v in results]

    st.markdown("### Nəticə")
    st.caption(
        f"Prompt: **{record['title']}** · max_new_tokens={record['settings']['max_new_tokens']} · "
        f"temperature={record['settings']['temperature']}"
    )

    cols = st.columns(len(variants))
    for col, variant in zip(cols, variants):
        result = results[variant]
        with col:
            st.subheader(VARIANT_LABELS[variant])
            st.markdown(_variant_metrics(result))
            st.caption(
                f"{result['output_words']} söz · {result['output_chars']} simvol · "
                f"prompt {result['prompt_tokens']} token"
            )
            if result.get("error"):
                st.error(result["error"])
            with st.container(border=True):
                st.markdown(result["text"] or "_Model boş cavab qaytardı._")

    base = results.get("base")
    tuned = results.get("finetuned")
    if base and tuned and not base.get("error") and not tuned.get("error"):
        cmp = compare_timings(base["elapsed_s"], tuned["elapsed_s"])
        faster_label = (
            "Base" if cmp["faster"] == "base"
            else "Fine-Tuned" if cmp["faster"] == "finetuned"
            else "bərabər"
        )
        st.markdown("### Vaxt müqayisəsi")
        m1, m2, m3 = st.columns(3)
        m1.metric("Base", f"{base['elapsed_s']:.1f} s")
        m2.metric(
            "Fine-Tuned",
            f"{tuned['elapsed_s']:.1f} s",
            delta=f"{cmp['diff_s']:+.1f} s",
            delta_color="inverse",
        )
        m3.metric(
            "Daha sürətli",
            faster_label,
            delta=f"{abs(cmp['diff_pct']):.1f}% fərq",
            delta_color="off",
        )
        st.caption(
            "Hər iki variant eyni model çəkisi, eyni tokenizer və eyni decoding "
            "parametrləri ilə ardıcıl işlədilib; fərq yalnız LoRA adapteridir."
        )


def _render_vote(record: dict) -> None:
    st.markdown("### Hansı cavab daha yaxşıdır?")
    if record.get("vote"):
        st.success(f"Səs qeydə alındı: **{record['vote']}**")
        return

    options = [
        ("base", "Base daha yaxşı"),
        ("finetuned", "Fine-Tuned daha yaxşı"),
        ("tie", "Bərabər"),
    ]
    columns = st.columns(3)
    for column, (key, label) in zip(columns, options):
        if column.button(label, key=f"vote_{record['run_id']}_{key}", width="stretch"):
            log_vote(
                VOTES_LOG,
                {
                    "run_id": record["run_id"],
                    "prompt_id": record["prompt_id"],
                    "title": record["title"],
                    "vote": key,
                    "base_s": record["results"]["base"]["elapsed_s"],
                    "finetuned_s": record["results"]["finetuned"]["elapsed_s"],
                },
            )
            record["vote"] = key
            st.session_state.last_run = record
            st.rerun()


# --------------------------------------------------------------------- sidebar
st.title("⚖️ Base vs Fine-Tuned — Qwen2.5-1.5B Python Tutor")
st.caption(
    "Eyni prompt bir kliklə həm base modelə, həm də QLoRA adapterli modelə göndərilir. "
    "Cavablar yan-yana, vaxt və token sayı ilə birlikdə göstərilir. Bütün inferensiya lokaldır."
)

with st.sidebar:
    st.header("Model")
    try:
        with st.spinner("Model və adapter yüklənir (ilk açılışda ~15 saniyə)..."):
            engine = get_engine()
    except Exception as error:  # noqa: BLE001 - shown to the user
        st.error(f"Model yüklənmədi: {type(error).__name__}: {error}")
        st.code(
            "Base model HF cache-də olmalıdır:\n"
            "  C:\\Users\\User\\.cache\\huggingface\\hub\\models--Qwen--Qwen2.5-1.5B-Instruct\n"
            f"Adapter: {ADAPTER_DIR}",
            language="text",
        )
        st.stop()

    info = engine.load_info
    st.markdown(
        f"**Cihaz:** `{info['device']}` · **dtype:** `{info['dtype']}`  \n"
        f"**Base load:** {info['base_load_s']:.1f}s · "
        f"**Adapter attach:** {info['adapter_attach_s']:.2f}s  \n"
        f"**Adapter:** {info['adapter_size_mb']:.1f} MB · "
        f"{info['adapter_parameters']:,} trainable parametr"
    )
    st.caption(
        f"Model: `{info['model_id']}` · torch {info['torch_version']} · "
        f"{info['torch_threads']} CPU thread"
    )

    st.divider()
    st.header("Parametrlər")
    max_new_tokens = st.slider(
        "Maksimum yeni token",
        min_value=32,
        max_value=256,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=16,
        help="Bu kompüterdə ~3 token/saniyə: 128 token təxminən 40–45 saniyə çəkir.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = greedy (deterministik, müqayisə üçün tövsiyə olunur).",
    )
    with_merged = st.checkbox(
        "Üçüncü sütun: merged model (bonus)",
        value=False,
        help=(
            "merge_and_unload() ilə ayrıca model yükləyir. Əlavə ~3 GB RAM və "
            "~15 saniyə yükləmə vaxtı tələb edir."
        ),
    )
    st.divider()
    if st.button("Söhbəti / nəticələri təmizlə", width="stretch"):
        st.session_state.runs = []
        st.session_state.last_run = None
        st.rerun()

    with st.expander("Bu panel necə işləyir?"):
        st.markdown(
            "1. Base model bir dəfə yüklənir və `PeftModel` ilə sarılır.\n"
            "2. **Base** cavabı `model.disable_adapter()` kontekstində yaradılır — "
            "yəni eyni çəkilər, adapter söndürülmüş.\n"
            "3. **Fine-Tuned** cavabı adapter aktiv olan eyni modeldən gəlir.\n"
            "4. Variantlar CPU-da **ardıcıl** işlədilir ki, vaxt ölçmələri "
            "bir-birini yavaşlatmasın.\n"
            "5. İlk token vaxtı (TTFT) və ümumi cavab müddəti ayrıca ölçülür."
        )

# ---------------------------------------------------------------------- prompt
st.markdown("### Test promptu")
titles = [prompt["title"] for prompt in PROMPTS] + ["6. Öz promptum"]
choice = st.radio("Prompt seç", titles, horizontal=True, label_visibility="collapsed")

if choice == "6. Öz promptum":
    preset = {
        "id": "custom",
        "title": "Öz promptum",
        "domain": "custom",
        "system": (
            "Sən Azərbaycan dilində Python öyrənən yeni başlayanlara kömək edən "
            "müəllimsən. Sadə və dəqiq izah ver, lazım olanda işlək Python kodu göstər."
        ),
        "user": "",
    }
    system_prompt = st.text_area(
        "System prompt", value=preset["system"], height=90, key="custom_system"
    )
    user_prompt = st.text_area(
        "İstifadəçi promptu",
        value="",
        height=110,
        placeholder="Məsələn: Python-da dictionary nədir? Qısa izah et.",
        key="custom_user",
    )
else:
    preset = get_prompt(PROMPTS[titles.index(choice)]["id"])
    system_prompt = preset["system"]
    user_prompt = st.text_area(
        "İstifadəçi promptu",
        value=preset["user"],
        height=110,
        key=f"prompt_{preset['id']}",
    )
    st.caption(f"System prompt: {system_prompt}")

run_clicked = st.button(
    "🚀 Hər iki modelə göndər",
    type="primary",
    width="stretch",
    disabled=not user_prompt.strip(),
)

# ------------------------------------------------------------------- execution
if run_clicked:
    variants = ["base", "finetuned"] + (["merged"] if with_merged else [])
    st.markdown("### Canlı nəticə")

    columns = st.columns(len(variants))
    holders: dict[str, tuple] = {}
    for column, variant in zip(columns, variants):
        with column:
            st.subheader(VARIANT_LABELS[variant])
            metrics_placeholder = st.empty()
            text_placeholder = st.empty()
            text_placeholder.caption("Növbə gözləyir...")
            holders[variant] = (metrics_placeholder, text_placeholder)

    results: dict[str, dict] = {}
    for variant in variants:
        metrics_placeholder, text_placeholder = holders[variant]
        text_placeholder.markdown("_Cavab yaradılır..._")

        def on_token(text: str, placeholder=text_placeholder) -> None:
            placeholder.markdown(text + "▌")

        result = engine.generate(
            variant,
            system_prompt,
            user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            on_token=on_token,
        )
        results[variant] = result.to_dict()
        text_placeholder.markdown(result.text or "_Model boş cavab qaytardı._")
        if result.error:
            text_placeholder.error(result.error)
        metrics_placeholder.markdown(_variant_metrics(result.to_dict()))

    record = {
        "run_id": new_run_id(),
        "prompt_id": preset["id"],
        "title": preset["title"],
        "domain": preset["domain"],
        "system": system_prompt,
        "user": user_prompt,
        "settings": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "results": results,
        "load_info": engine.load_info,
    }
    log_run(RUNS_LOG, record)
    st.session_state.last_run = record
    st.session_state.runs.append(record)
    st.rerun()

# --------------------------------------------------------------------- results
last_run = st.session_state.get("last_run")
if last_run:
    st.divider()
    _render_results(last_run)

    if "base" in last_run["results"] and "finetuned" in last_run["results"]:
        _render_vote(last_run)

    st.download_button(
        "Nəticəni JSON kimi yüklə",
        data=json.dumps(last_run, indent=2, ensure_ascii=False),
        file_name=f"{last_run['run_id']}.json",
        mime="application/json",
    )

# --------------------------------------------------------------------- history
runs = st.session_state.get("runs", [])
votes = read_jsonl(VOTES_LOG)

with st.expander(f"Jurnal — {len(runs)} run, {len(votes)} səs", expanded=False):
    if not runs:
        st.caption("Hələ heç bir müqayisə aparılmayıb.")
    else:
        rows = []
        for run in runs:
            base = run["results"].get("base", {})
            tuned = run["results"].get("finetuned", {})
            rows.append(
                {
                    "vaxt": run["run_id"],
                    "prompt": run["title"],
                    "base_s": base.get("elapsed_s"),
                    "finetuned_s": tuned.get("elapsed_s"),
                    "base_tokens": base.get("output_tokens"),
                    "finetuned_tokens": tuned.get("output_tokens"),
                    "səs": run.get("vote", "—"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

    if votes:
        counts: dict[str, int] = {}
        for vote in votes:
            counts[vote["vote"]] = counts.get(vote["vote"], 0) + 1
        st.markdown("**Səslərin cəmi:** " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    st.caption(
        f"Run jurnalı: `{RUNS_LOG}` · Səs jurnalı: `{VOTES_LOG}` · Qovluq: `{LOGS_DIR}`"
    )

st.divider()
st.caption(
    "Base və Fine-Tuned eyni `Qwen/Qwen2.5-1.5B-Instruct` çəkisindən istifadə edir; "
    "fərq yalnız `adapters/qwen25-python-tutor` LoRA adapteridir."
)
