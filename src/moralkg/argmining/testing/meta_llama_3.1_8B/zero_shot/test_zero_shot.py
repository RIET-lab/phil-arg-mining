#!/usr/bin/env python3

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def render_template(template_text: str, context: dict) -> str:
    """
    Render a template string using Python's str.format with the given context.
    """
    try:
        return template_text.format(**context)
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise KeyError(
            f"Missing placeholder variable '{missing_key}' in template context."
        )


def build_input_ids(
    tokenizer, system_text: str, user_text: str, paper_text: str
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{user_text} {paper_text}"},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors=None,
        )
        encoded = tokenizer(text, return_tensors="pt")
    except Exception:
        raise Exception
    return encoded.input_ids.to(device)


def load_model_and_tokenizer(
    base_model: str,
    adapter_dir: str,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    use_cuda = torch.cuda.is_available()
    dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map='auto')
    model = PeftModel.from_pretrained(
        model, str(Path(adapter_dir).resolve()), local_files_only=True
    )

    device = torch.device("cuda")
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.inference_mode()
def generate(
    model: torch.nn.Module, tokenizer, input_ids: torch.Tensor, max_new_tokens: int
) -> tuple[str, dict]:
    # Input token count
    input_token_count = int(input_ids.numel())

    # Time the generation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    generated_only = outputs[0, input_ids.shape[-1] :]
    output_token_count = int(generated_only.numel())
    latency_seconds = max(end - start, 1e-9)
    output_tokens_per_second = float(output_token_count) / latency_seconds

    text = tokenizer.decode(generated_only, skip_special_tokens=True)
    metrics = {
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "latency_seconds": latency_seconds,
        "output_tokens_per_second": output_tokens_per_second,
        "total_tokens": input_token_count + output_token_count,
    }
    return text, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal PEFT inference for Llama 3.1 8B.")
    p.add_argument("-b", "--base", type=str, required=True)
    p.add_argument("-a", "--adapter", type=str, required=True)
    p.add_argument("-s", "--system", type=str, required=True)
    p.add_argument("-u", "--user", type=str, required=True)
    p.add_argument("-p", "--paper", type=str, required=True)
    p.add_argument("-t", "--max-new-tokens", type=int, default=8192)
    p.add_argument(
        "--schema",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "schemas" / "model_response.json"),
        help=(
            "Path to the model response JSON schema file injected into prompts."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    system_template = read_text_file(Path(args.system))
    user_template = read_text_file(Path(args.user))
    paper_text = read_text_file(Path(args.paper))
    model_response_text = read_text_file(Path(args.schema))

    template_context = {
        "paper": paper_text,
        "model_response": model_response_text,
    }

    # Render templates
    system_text = render_template(system_template, template_context)
    user_text = render_template(user_template, template_context)

    model, tokenizer = load_model_and_tokenizer(args.base, args.adapter)
    input_ids = build_input_ids(tokenizer, system_text, user_text, paper_text)
    output_text, metrics = generate(
        model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens
    )

    metrics_line = (
        f"\n[metrics] input_tokens={metrics['input_tokens']} "
        f"output_tokens={metrics['output_tokens']} "
        f"total_tokens={metrics['total_tokens']} "
        f"latency_s={metrics['latency_seconds']:.3f} "
        f"output_tps={metrics['output_tokens_per_second']:.2f}"
    )
    print(metrics_line)

    # Send results to timestamped file
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y_%m_%d_%H%M")
    results_path = results_dir / f"{timestamp}.txt"
    results_content = f"{metrics_line}\n\n{output_text}\n"
    results_path.write_text(results_content, encoding="utf-8")
    print(f"[saved] {results_path}")


if __name__ == "__main__":
    main()
