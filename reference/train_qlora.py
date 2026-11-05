import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_FILE = ROOT / "data" / "train.jsonl"
VALIDATION_FILE = ROOT / "data" / "validation.jsonl"
SOURCE_FILE = ROOT / "data" / "all_examples.jsonl"
ADAPTER_DIR = ROOT / "adapters" / "qwen25-python-tutor"
LOG_DIR = ROOT / "logs"

SEED = 42
EPOCHS = 3
MAX_LENGTH = 256
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
LORA_RANK = 4
LORA_ALPHA = 8

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU görünmür; QLoRA train üçün CUDA tələb olunur.")

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
LOG_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise ValueError(f"Fayl boşdur: {path}")
    return rows

train_rows = load_jsonl(TRAIN_FILE)
validation_rows = load_jsonl(VALIDATION_FILE)

for path, rows in ((TRAIN_FILE, train_rows), (VALIDATION_FILE, validation_rows)):
    for index, row in enumerate(rows, start=1):
        roles = [message.get("role") for message in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{path.name}, sətir {index}: rollar düzgün deyil: {roles}")

print("GPU:", torch.cuda.get_device_name(0))
print(f"Train nümunələri: {len(train_rows)} | Validation nümunələri: {len(validation_rows)}")
print("Tokenizer yüklənir...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def encode_row(row):
    messages = row["messages"]
    full_result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_result = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )

    def extract_ids(result):
        if hasattr(result, "keys") and "input_ids" in result.keys():
            result = result["input_ids"]
        if hasattr(result, "detach"):
            result = result.detach().cpu()
        if hasattr(result, "tolist"):
            result = result.tolist()
        if result and isinstance(result[0], list):
            result = result[0]
        return [int(token) for token in result]

    full_ids = extract_ids(full_result)
    prompt_ids = extract_ids(prompt_result)

    common_prefix = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        common_prefix += 1

    answer_ids = full_ids[common_prefix:common_prefix + MAX_LENGTH]
    if not answer_ids:
        raise ValueError(
            f"Assistant cavabı üçün token tapılmadı. "
            f"Prompt={len(prompt_ids)}, full={len(full_ids)}"
        )

    prompt_budget = min(common_prefix, MAX_LENGTH - len(answer_ids))
    kept_prompt = full_ids[common_prefix - prompt_budget:common_prefix]
    input_ids = kept_prompt + answer_ids
    labels = [-100] * len(kept_prompt) + answer_ids

    return {"input_ids": input_ids, "labels": labels}
train_data = [encode_row(row) for row in train_rows]
validation_data = [encode_row(row) for row in validation_rows]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("Base model 4-bit rejimdə yüklənir...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

device = torch.device("cuda:0")
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=LEARNING_RATE,
)

def make_batch(example):
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels

def evaluate():
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for example in validation_data:
            input_ids, attention_mask, labels = make_batch(example)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_total += float(output.loss.detach().cpu())
            del output, input_ids, attention_mask, labels
    return loss_total / len(validation_data)

source_hash = hashlib.sha256(SOURCE_FILE.read_bytes()).hexdigest()
config = {
    "model_id": MODEL_ID,
    "source_dataset": str(SOURCE_FILE.relative_to(ROOT)),
    "source_dataset_sha256": source_hash,
    "train_examples": len(train_data),
    "validation_examples": len(validation_data),
    "seed": SEED,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "target_modules": ["q_proj", "v_proj"],
    "gpu": torch.cuda.get_device_name(0),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
(ROOT / "logs" / "training_config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

history = []
print("QLoRA train başlayır...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_rows = list(train_data)
    random.Random(SEED + epoch).shuffle(epoch_rows)
    train_loss_total = 0.0
    update_count = 0

    for start in range(0, len(epoch_rows), GRADIENT_ACCUMULATION):
        group = epoch_rows[start:start + GRADIENT_ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        group_loss_total = 0.0

        for example in group:
            input_ids, attention_mask, labels = make_batch(example)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Train loss etibarsızdır: {loss.item()}")
            group_loss_total += float(loss.detach().cpu())
            (loss / len(group)).backward()
            del output, loss, input_ids, attention_mask, labels

        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        update_count += 1
        train_loss_total += group_loss_total
        print(
            f"Epoch {epoch}/{EPOCHS} | update {update_count} | "
            f"loss {group_loss_total / len(group):.4f}",
            flush=True,
        )

    train_loss = train_loss_total / len(epoch_rows)
    validation_loss = evaluate()
    model.train()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "cuda_peak_gb": torch.cuda.max_memory_allocated(0) / 1024**3,
    })

    with (LOG_DIR / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(
        f"Epoch {epoch} tamamlandı | train loss={train_loss:.4f} | "
        f"validation loss={validation_loss:.4f} | "
        f"GPU peak={history[-1]['cuda_peak_gb']:.2f} GB",
        flush=True,
    )

print("Adapter və tokenizer saxlanılır...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("QLoRA train tamamlandı.")
print("Adapter:", ADAPTER_DIR)
print("Loss qeydləri:", LOG_DIR / "training_log.csv")
print("Train konfiqurasiyası:", LOG_DIR / "training_config.json")

