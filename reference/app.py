import os
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import queue
import threading
import time
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = ROOT / "adapters" / "qwen25-python-tutor"

st.set_page_config(page_title="Python Study Assistant", page_icon="🐍")
st.title("Python Study Assistant")
st.caption("Azərbaycan dilində Python köməkçisi · Qwen2.5-1.5B + öz QLoRA adapterim · Lokal inferensiya")

@st.cache_resource(show_spinner=False)
def load_local_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU tapılmadı. Bu layihə NVIDIA GPU ilə işləmək üçün qurulub.")
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError(f"Öz adapterim tapılmadı: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        local_files_only=True,
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER_DIR),
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = True
    return tokenizer, model

try:
    with st.spinner("Lokal model və adapter yüklənir..."):
        tokenizer, model = load_local_model()
except Exception as error:
    st.error(f"Modeli yükləmək mümkün olmadı: {type(error).__name__}: {error}")
    st.info("Base modelin əvvəlcə Hugging Face-dən endirildiyini və adapter fayllarının mövcud olduğunu yoxla.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Tapşırıq")
    mode = st.selectbox(
        "Rejim",
        ["Python izahı", "Quiz yarat", "Cavabı yoxla"],
    )
    max_new_tokens = st.slider("Cavab uzunluğu", min_value=64, max_value=192, value=128, step=16)
    st.caption("Model və adapter bu kompüterdə yüklənir. Tətbiq hosted LLM API çağırmır.")
    if st.button("Söhbəti təmizlə"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_text = st.chat_input("Python mövzusu və ya sualını yaz...")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    if mode == "Quiz yarat":
        system_text = (
            "Sən Azərbaycan dilində Python müəllimisən. İstifadəçinin seçdiyi mövzu üzrə "
            "3 qısa suallıq quiz yarat. Cavab açarını sonda ayrıca göstər. "
            "Kod suallarının cavablarını yoxla və aydın Azərbaycan dilində yaz."
        )
        model_user_text = (
            f"Python mövzusu: {user_text}\n"
            "Bu mövzu üzrə 3 suallıq quiz yarat və cavab açarını sonda göstər."
        )
    elif mode == "Cavabı yoxla":
        system_text = (
            "Sən Azərbaycan dilində Python müəllimisən. Şagirdin cavabını yoxla, "
            "düzgün və səhv hissələri qısa izah et. Səhvdirsə doğru cavabı göstər."
        )
        model_user_text = f"Şagirdin cavabını yoxla və izah et:\n{user_text}"
    else:
        system_text = (
            "Sən Azərbaycan dilində Python öyrənən yeni başlayanlara kömək edən müəllimsən. "
            "Sadə və dəqiq izah ver, lazım olanda işlək Python kodu göstər."
        )
        model_user_text = user_text

    model_messages = [{"role": "system", "content": system_text}]
    model_messages.extend(st.session_state.messages)
    model_messages.append({"role": "user", "content": model_user_text})

    try:
        encoded = tokenizer.apply_chat_template(
            model_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {
            key: value.to("cuda:0")
            for key, value in encoded.items()
            if hasattr(value, "to")
        }

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors = queue.Queue()

        generation_args = {
            **encoded,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        def generate_in_background():
            try:
                model.generate(**generation_args)
            except Exception as error:
                errors.put(error)
                try:
                    streamer.end()
                except Exception:
                    pass

        started = time.perf_counter()
        response = ""

        with st.chat_message("assistant"):
            answer_box = st.empty()
            with st.spinner("Model cavab yaradır..."):
                worker = threading.Thread(
                    target=generate_in_background,
                    daemon=True,
                )
                worker.start()

                try:
                    for piece in streamer:
                        response += piece
                        answer_box.markdown(response + "▌")
                except Exception as stream_error:
                    errors.put(stream_error)

                worker.join()

            answer_box.markdown(response or "Model boş cavab qaytardı.")
            elapsed = time.perf_counter() - started

            if not errors.empty():
                error = errors.get()
                error_text = str(error)
                if "out of memory" in error_text.lower():
                    st.error(
                        "GPU yaddaşı çatmadı. Digər GPU tətbiqlərini bağla və "
                        "yan paneldə cavab uzunluğunu azalt."
                    )
                else:
                    st.error(f"Model cavabı yaradarkən xəta baş verdi: {error_text}")
            else:
                st.caption(f"Cavab müddəti: {elapsed:.1f} saniyə")

        if not errors.qsize() and response:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as error:
        if "out of memory" in str(error).lower():
            st.error("GPU yaddaşı çatmadı. Cavab uzunluğunu azaldıb yenidən yoxla.")
        else:
            st.error(f"İstək emal edilərkən xəta baş verdi: {type(error).__name__}: {error}")
