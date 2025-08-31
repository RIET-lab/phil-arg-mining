from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moralkg import Config, get_logger

LOGGER = get_logger(__name__)


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    use_cuda = torch.cuda.is_available()
    dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    device = torch.device("cuda:0" if use_cuda else "cpu") # TODO: Allow other device indices to be used
    return device, dtype


def load_generator_model(
    base_model: str,
    adapter_dir: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    cache_dir: Optional[str] = None,
):
    from peft import PeftModel

    if device is None or dtype is None:
        device, dtype = get_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=str(base_model), use_fast=True, cache_dir=cache_dir)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}, cache_dir=cache_dir
    )
    model = PeftModel.from_pretrained(
        model, str(Path(adapter_dir).resolve()), local_files_only=True
    )

    # Ensure model uses the specified device consistently
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info(
        "Loaded generator: base=%s, adapter=%s, device=%s, dtype=%s",
        base_model,
        adapter_dir,
        device,
        dtype,
    )
    return model, tokenizer


def load_generator_model_from_config(*, cache_dir: Optional[str] = None):
    """
    Convenience loader using moralkg Config paths.
    Falls back to HF ids if local dirs are not set.
    """
    # Create a config instance in order to set up models
    cfg = None
    try:
        cfg = Config.load()
    except Exception:
        cfg = None
    if cfg is not None:
        base_cfg = Config.get("paths.models.end2end.base")
        adapter_cfg = Config.get("paths.models.end2end.finetune")

        base_model = (base_cfg.get("dir") or base_cfg.get("hf")) if isinstance(base_cfg, dict) else base_cfg
        adapter_dir = adapter_cfg.get("dir") if isinstance(adapter_cfg, dict) else adapter_cfg

    if not base_model:
        raise ValueError("Config paths.models.end2end.base is not set")
    if not adapter_dir:
        raise ValueError("Config paths.models.end2end.finetune.dir is not set")
    return load_generator_model(str(base_model), str(adapter_dir), cache_dir=cache_dir)


def build_input_ids(tokenizer, system_text: str, user_text: str, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device, _ = get_device_and_dtype()
    
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors=None,
    )
    encoded = tokenizer(text, return_tensors="pt").to(device)
    return encoded.input_ids.to(device)


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_new_tokens: int,
    temperature: float,
):
    if device is None:
        device, _ = get_device_and_dtype()
    
    if dtype is None:
        _, dtype = get_device_and_dtype()

    # Ensure input_ids are on the correct device
    input_ids = input_ids.to(device)
    
    # Ensure model is on the correct device
    model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import time as _time

    start = _time.perf_counter()
    do_sample = float(temperature) > 0.0
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else 0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = _time.perf_counter()

    generated_only = outputs[0, input_ids.shape[-1] :]
    text = tokenizer.decode(generated_only, skip_special_tokens=True)

    input_token_count = int(input_ids.numel())
    output_token_count = int(generated_only.numel())
    latency_seconds = max(end - start, 1e-9)
    output_tokens_per_second = float(output_token_count) / latency_seconds

    metrics = {
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "latency_seconds": latency_seconds,
        "output_tokens_per_second": output_tokens_per_second,
        "total_tokens": input_token_count + output_token_count,
        "device": str(device),
        "dtype": str(dtype),
    }
    LOGGER.debug(
        "Generated output: tokens_in=%d tokens_out=%d total=%d latency=%.3fs tps=%.2f temp=%.2f max_new=%d device=%s",
        input_token_count,
        output_token_count,
        input_token_count + output_token_count,
        latency_seconds,
        output_tokens_per_second,
        float(temperature),
        int(max_new_tokens),
        device,
    )
    return text, metrics


def generate_chat(
    model: torch.nn.Module,
    tokenizer,
    *,
    system_text: str,
    user_text: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_new_tokens: int,
    temperature: float,
):
    """
    High-level helper: build chat input_ids and generate text+metrics.
    """
    if device is None or dtype is None:
        device, dtype = get_device_and_dtype()
    
    input_ids = build_input_ids(tokenizer, system_text, user_text, device=device)
    return generate(
        model,
        tokenizer,
        input_ids,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )