"""Load the Qwen base model once and expose base / adapter / merged variants.

Both required variants share one copy of the base weights:

* ``finetuned`` -> the ``PeftModel`` with the LoRA adapter active
* ``base``      -> exactly the same instance inside ``model.disable_adapter()``

This guarantees the prompt, tokenizer, precision and decoding settings are
identical; the only difference is the adapter.  A third, optional ``merged``
variant loads a separate copy of the base and calls ``merge_and_unload()`` so
the load cost / inference speed of the merged checkpoint can be measured too.
"""

from __future__ import annotations

import gc
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import ADAPTER_DIR, DEFAULT_MAX_NEW_TOKENS, MODEL_ID

VARIANT_KEYS = ("base", "finetuned", "merged")

TokenCallback = Callable[[str], None]


def _resolve_device() -> str:
    if os.environ.get("COMPARE_DEVICE", "").lower() == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    override = os.environ.get("COMPARE_DTYPE", "").lower()
    if override in {"float32", "fp32"}:
        return torch.float32
    if override in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if override in {"float16", "fp16"}:
        return torch.float16
    return torch.float16 if device == "cuda" else torch.bfloat16


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


@dataclass
class GenerationResult:
    variant: str
    text: str
    elapsed_s: float
    ttft_s: float | None
    prompt_tokens: int
    output_tokens: int
    output_chars: int
    output_words: int
    tokens_per_second: float
    max_new_tokens: int
    temperature: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ComparisonEngine:
    """Owns the tokenizer and the three callable model variants."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        adapter_dir: str | Path = ADAPTER_DIR,
        local_files_only: bool = True,
        warmup: bool = True,
    ) -> None:
        self.model_id = model_id
        self.adapter_dir = Path(adapter_dir)
        self.local_files_only = local_files_only

        self.device = _resolve_device()
        self.dtype = _resolve_dtype(self.device)

        if self.device == "cpu":
            threads = int(os.environ.get("COMPARE_THREADS", "0")) or min(12, os.cpu_count() or 4)
            torch.set_num_threads(threads)

        self._lock = threading.Lock()
        self._merged_model = None
        self._merged_build_s: float | None = None

        started = time.perf_counter()
        self.tokenizer = self._load_tokenizer()
        tokenizer_s = time.perf_counter() - started

        started = time.perf_counter()
        base = self._load_base_model()
        base_load_s = time.perf_counter() - started

        started = time.perf_counter()
        self.model = PeftModel.from_pretrained(
            base,
            str(self.adapter_dir),
            local_files_only=self.local_files_only,
        )
        adapter_attach_s = time.perf_counter() - started

        self.model.eval()
        self.model.config.use_cache = True
        if self.tokenizer.pad_token_id is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        self.load_info = {
            "model_id": self.model_id,
            "adapter_dir": str(self.adapter_dir),
            "device": self.device,
            "dtype": _dtype_name(self.dtype),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "tokenizer_load_s": round(tokenizer_s, 3),
            "base_load_s": round(base_load_s, 3),
            "adapter_attach_s": round(adapter_attach_s, 3),
            "total_load_s": round(tokenizer_s + base_load_s + adapter_attach_s, 3),
            "base_parameters": sum(p.numel() for p in self.model.get_base_model().parameters()),
            "adapter_parameters": sum(
                p.numel() for name, p in self.model.named_parameters() if "lora" in name.lower()
            ),
            "adapter_size_mb": round(
                sum(f.stat().st_size for f in self.adapter_dir.glob("adapter_model*")) / 1_048_576, 2
            ),
            "warmup_s": None,
        }

        if warmup:
            self.load_info["warmup_s"] = round(self._warmup(), 3)

    # ------------------------------------------------------------------ load
    def _load_tokenizer(self):
        source = self.adapter_dir if (self.adapter_dir / "tokenizer.json").exists() else self.model_id
        return AutoTokenizer.from_pretrained(
            str(source),
            local_files_only=self.local_files_only,
        )

    def _load_base_model(self):
        kwargs = {
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model.eval()
        if self.device == "cuda":
            model.to("cuda")
        return model

    def _warmup(self) -> float:
        """One tiny generation per required variant so timings exclude lazy init."""
        messages = [
            {"role": "system", "content": "Qısa cavab ver."},
            {"role": "user", "content": "Salam."},
        ]
        started = time.perf_counter()
        for variant in ("base", "finetuned"):
            try:
                self.generate_messages(variant, messages, max_new_tokens=2, temperature=0.0)
            except Exception:  # noqa: BLE001 - warmup must never block startup
                pass
        return time.perf_counter() - started

    # --------------------------------------------------------------- variants
    @property
    def merged_ready(self) -> bool:
        return self._merged_model is not None

    @property
    def merged_build_s(self) -> float | None:
        return self._merged_build_s

    def ensure_merged(self) -> float:
        """Build the merged model on first use and return the build time."""
        if self._merged_model is not None:
            return self._merged_build_s or 0.0
        started = time.perf_counter()
        base = self._load_base_model()
        peft_model = PeftModel.from_pretrained(
            base,
            str(self.adapter_dir),
            local_files_only=self.local_files_only,
        )
        merged = peft_model.merge_and_unload()
        merged.eval()
        merged.config.use_cache = True
        if self.tokenizer.pad_token_id is not None:
            merged.generation_config.pad_token_id = self.tokenizer.pad_token_id
        merged.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self._merged_model = merged
        self._merged_build_s = time.perf_counter() - started
        self.load_info["merged_build_s"] = round(self._merged_build_s, 3)
        gc.collect()
        return self._merged_build_s

    def _select(self, variant: str):
        if variant == "merged":
            self.ensure_merged()
            return self._merged_model, nullcontext()
        if variant == "base":
            return self.model, self.model.disable_adapter()
        if variant == "finetuned":
            return self.model, nullcontext()
        raise ValueError(f"Unknown variant: {variant}")

    # ------------------------------------------------------------- generation
    def build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def encode(self, messages: list[dict]):
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        return encoded

    def _generation_kwargs(self, max_new_tokens: int, temperature: float) -> dict:
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def generate(
        self,
        variant: str,
        system: str,
        user: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        return self.generate_messages(
            variant,
            self.build_messages(system, user),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            on_token=on_token,
        )

    def generate_base(
        self,
        system: str,
        user: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        """Call the plain base model (adapter disabled)."""
        return self.generate(
            "base", system, user, max_new_tokens, temperature, on_token
        )

    def generate_finetuned(
        self,
        system: str,
        user: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        """Call the same base model with the LoRA adapter active."""
        return self.generate(
            "finetuned", system, user, max_new_tokens, temperature, on_token
        )

    def generate_messages(
        self,
        variant: str,
        messages: list[dict],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        with self._lock:
            return self._generate_locked(
                variant, messages, max_new_tokens, temperature, on_token
            )

    def _generate_locked(
        self,
        variant: str,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        on_token: TokenCallback | None,
    ) -> GenerationResult:
        try:
            model, adapter_context = self._select(variant)
            encoded = self.encode(messages)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            return GenerationResult(
                variant=variant,
                text="",
                elapsed_s=0.0,
                ttft_s=None,
                prompt_tokens=0,
                output_tokens=0,
                output_chars=0,
                output_words=0,
                tokens_per_second=0.0,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature or 0.0),
                error=f"{type(exc).__name__}: {exc}",
            )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        kwargs = self._generation_kwargs(max_new_tokens, temperature)

        if temperature and temperature > 0:
            torch.manual_seed(42)

        if on_token is None:
            started = time.perf_counter()
            try:
                with torch.inference_mode(), adapter_context:
                    output = model.generate(**encoded, **kwargs)
                elapsed = time.perf_counter() - started
                generated = output[0][prompt_tokens:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                ttft = None
                error = None
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                elapsed = time.perf_counter() - started
                text, ttft, error = "", None, f"{type(exc).__name__}: {exc}"
        else:
            text, elapsed, ttft, error = self._generate_streaming(
                variant, model, encoded, kwargs, on_token
            )

        output_tokens = len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0
        return GenerationResult(
            variant=variant,
            text=text,
            elapsed_s=round(elapsed, 3),
            ttft_s=round(ttft, 3) if ttft is not None else None,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_chars=len(text),
            output_words=len(text.split()),
            tokens_per_second=round(output_tokens / elapsed, 3) if elapsed > 0 else 0.0,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature or 0.0),
            error=error,
        )

    def _generate_streaming(self, variant, model, encoded, kwargs, on_token):
        tokenizer = self.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        errors: queue.Queue = queue.Queue()

        def worker() -> None:
            # The adapter toggle is created inside the worker so the flag is
            # set and cleared on the same thread that runs generate().
            context = model.disable_adapter() if variant == "base" else nullcontext()
            try:
                with torch.inference_mode(), context:
                    model.generate(**encoded, streamer=streamer, **kwargs)
            except Exception as exc:  # noqa: BLE001
                errors.put(exc)
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001
                    pass

        started = time.perf_counter()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        text = ""
        ttft = None
        try:
            for piece in streamer:
                if ttft is None:
                    ttft = time.perf_counter() - started
                text += piece
                on_token(text)
        finally:
            thread.join()
        elapsed = time.perf_counter() - started

        error = None
        if not errors.empty():
            exc = errors.get()
            error = f"{type(exc).__name__}: {exc}"
        return text.strip(), elapsed, ttft, error


def format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.1f} s"
