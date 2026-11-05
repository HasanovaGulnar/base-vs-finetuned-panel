from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = ROOT / "adapters" / "qwen25-python-tutor"
OUTPUT_FILE = ROOT / "examples" / "before_after.txt"
DEVICE = torch.device("cuda:0")

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(f"Adapter tapılmadı: {ADAPTER_DIR}")

print("Tokenizer lokal fayllardan yüklənir...")
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

print("Base model lokal cache-dən yüklənir...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=torch.float16,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
base_model.eval()

cases = [
    (
        "Quiz yaratma",
        [
            {
                "role": "system",
                "content": "Sən Azərbaycan dilində Python öyrənən yeni başlayanlara kömək edən müəllimsən. Dəqiq, sadə cavab ver.",
            },
            {
                "role": "user",
                "content": "Quiz yarat: Python siyahıları mövzusunda 3 sual ver və cavab açarını sonda göstər.",
            },
        ],
    ),
    (
        "Cavabı yoxlama",
        [
            {
                "role": "system",
                "content": "Sən Azərbaycan dilində Python öyrənən yeni başlayanlara kömək edən müəllimsən. Cavabı yoxla, düzgün və səhv hissələri qısa izah et.",
            },
            {
                "role": "user",
                "content": "Cavabımı yoxla: `[3, 7, 8][1]` nəticəsi 8-dir.",
            },
        ],
    ),
]

def generate_reply(model, messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
        if hasattr(value, "to")
    }
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=140,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][prompt_length:],
        skip_special_tokens=True,
    ).strip()

results = []
for title, messages in cases:
    print(f"Base model cavab yaradır: {title}")
    base_reply = generate_reply(base_model, messages)
    results.append((title, messages[-1]["content"], base_reply, None))

print("Öz adapterimiz PeftModel.from_pretrained ilə yüklənir...")
model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
    local_files_only=True,
)
model.eval()
del base_model

for index, (title, prompt, base_reply, _) in enumerate(results):
    print(f"Fine-tuned model cavab yaradır: {title}")
    tuned_reply = generate_reply(model, cases[index][1])
    results[index] = (title, prompt, base_reply, tuned_reply)

sections = []
for title, prompt, base_reply, tuned_reply in results:
    sections.extend([
        f"=== {title} ===",
        f"Prompt:\n{prompt}",
        f"\nBase model:\n{base_reply}",
        f"\nÖz QLoRA adapterimiz:\n{tuned_reply}",
        "",
    ])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text("\n".join(sections), encoding="utf-8")
print("Adapter yüklənməsi və müqayisə tamamlandı.")
print("Nəticə faylı:", OUTPUT_FILE)
print(OUTPUT_FILE.read_text(encoding="utf-8"))
