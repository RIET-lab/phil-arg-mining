#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import rootutils
rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import torch
from transformers import AutoTokenizer

from moralkg.config import Config
from moralkg.logging import get_logger
from moralkg.argmining.models.generator import (
    build_input_ids,
    generate,
    load_generator_model,
)
from moralkg.argmining.models.rag import (
    RagParams,
    create_in_memory_faiss_retriever,
    get_qwen3_embeddings,
    split_text_into_chunks,
)


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_template(template_text: str, context: dict) -> str:
    try:
        return template_text.format(**context)
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise KeyError(
            f"Missing placeholder variable '{missing_key}' in template context."
        )


def device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    # retained for backward compatibility if imported, but unused here
    use_cuda = torch.cuda.is_available()
    dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_cuda else torch.float32)
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, dtype


def ensure_models_root() -> Path:
    models_root = Path("/models")
    models_root.mkdir(parents=True, exist_ok=True)
    return models_root


def resolve_under_models(relative_or_config_path: Optional[str]) -> Path:
    models_root = ensure_models_root()
    if not relative_or_config_path:
        return models_root
    p = Path(relative_or_config_path)
    if p.is_absolute():
        return p
    parts = list(p.parts)
    if parts and parts[0] == "models":
        parts = parts[1:]
    return models_root.joinpath(*parts)


def load_rag_params(cfg: Config, logger) -> RagParams:
    # Defaults with safe fallbacks if config values are missing
    default_chunk_size = 800
    default_overlap = 120
    default_top_k = 10

    rag_cfg = cfg.get("snowball.phase_1.hparams.e2e.rag", {}) or {}
    chunk_size_cfg = rag_cfg.get("chunk_size", None)
    overlap_cfg = rag_cfg.get("chunk_overlap", None)
    top_k_cfg = rag_cfg.get("top_k", None)

    if chunk_size_cfg in (None, "", "TODO"):
        logger.warning(
            f"RAG chunk_size not set in config; using default {default_chunk_size} tokens."
        )
        chunk_size_tokens = default_chunk_size
    else:
        try:
            chunk_size_tokens = int(chunk_size_cfg)
        except Exception:
            logger.warning(
                f"Invalid RAG chunk_size value '{chunk_size_cfg}'; using default {default_chunk_size}."
            )
            chunk_size_tokens = default_chunk_size

    if isinstance(overlap_cfg, list) and overlap_cfg:
        overlap_tokens = int(overlap_cfg[0])
    elif isinstance(overlap_cfg, int):
        overlap_tokens = overlap_cfg
    else:
        logger.warning(
            f"RAG chunk_overlap not set in config; using default {default_overlap} tokens."
        )
        overlap_tokens = default_overlap

    try:
        top_k = int(top_k_cfg) if top_k_cfg is not None else default_top_k
    except Exception:
        logger.warning(
            f"Invalid RAG top_k value '{top_k_cfg}'; using default {default_top_k}."
        )
        top_k = default_top_k

    return RagParams(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=overlap_tokens,
        top_k=top_k,
    )


def build_retriever(
    paper_text: str,
    *,
    tokenizer,
    rag_params: RagParams,
    logger,
):
    chunks = split_text_into_chunks(
        paper_text,
        tokenizer,
        chunk_size_tokens=rag_params.chunk_size_tokens,
        chunk_overlap_tokens=rag_params.chunk_overlap_tokens,
    )
    logger.info(
        f"Split paper into {len(chunks)} chunks (size={rag_params.chunk_size_tokens}, overlap={rag_params.chunk_overlap_tokens})."
    )
    emb = get_qwen3_embeddings(cache_dir=str(resolve_under_models("models/embeddings/qwen3_0_6b")))
    retriever = create_in_memory_faiss_retriever(chunks, top_k=rag_params.top_k, embeddings=emb)
    return retriever


def build_context_from_retriever(retriever, query_text: str, logger) -> str:
    docs = retriever.get_relevant_documents(query_text)
    logger.info(f"Retrieved {len(docs)} context chunks for RAG.")
    context = "\n\n".join(d.page_content for d in docs)
    return context


def parse_args(cfg: Config) -> argparse.Namespace:
    repo_root = Path(rootutils.setup_root(__file__, indicator=".git"))
    prompts_dir = repo_root / "models/meta_llama_3.1_8B/prompts"

    base_hf = cfg.get("snowball.models.meta_llama_3.1_8B.base.hf", "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    adapter_dir_cfg = cfg.get("snowball.models.meta_llama_3.1_8B.finetune.dir", "models/meta_llama_3.1_8B/finetune")
    adapter_dir_default = resolve_under_models(adapter_dir_cfg)

    p = argparse.ArgumentParser(description="RAG zero-shot PEFT inference for Llama 3.1 8B.")
    p.add_argument("-b", "--base", type=str, default=base_hf)
    p.add_argument("-a", "--adapter", type=str, default=str(adapter_dir_default))
    p.add_argument("-s", "--system", type=str, default=str(prompts_dir / "rag_zero_shot_system.txt"))
    p.add_argument("-u", "--user", type=str, default=str(prompts_dir / "rag_zero_shot_user.txt"))
    p.add_argument("-p", "--paper", type=str, required=True)
    # Sanitize max_new_tokens from config
    cfg_max_new_tokens = cfg.get("snowball.phase_1.hparams.e2e.decoding.max_new_tokens", None)
    try:
        default_max_new_tokens = int(cfg_max_new_tokens) if cfg_max_new_tokens not in (None, "", "TODO") else 8192
    except Exception:
        default_max_new_tokens = 8192
    p.add_argument("-t", "--max-new-tokens", type=int, default=default_max_new_tokens)
    p.add_argument(
        "--temperature",
        type=float,
        default=float(cfg.get("snowball.phase_1.hparams.e2e.decoding.temperature", 0.0) or 0.0),
    )
    p.add_argument(
        "--schema",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "schemas" / "model_response.json"),
        help="Path to the model response JSON schema file injected into prompts.",
    )
    return p.parse_args()


def main() -> None:
    logger = get_logger("argmining.zero_shot_rag")
    cfg = Config.load()
    args = parse_args(cfg)

    logger.info("Loading templates and paper text...")
    system_template = read_text_file(Path(args.system))
    user_template = read_text_file(Path(args.user))
    paper_text = read_text_file(Path(args.paper))
    model_response_text = read_text_file(Path(args.schema))

    # Load model/tokenizer
    logger.info("Loading model and tokenizer...")
    models_cache_dir = resolve_under_models("models/hf_cache")
    model, tokenizer = load_generator_model(
        base_model=args.base,
        adapter_dir=args.adapter,
        cache_dir=str(models_cache_dir),
    )

    # RAG setup
    rag_params = load_rag_params(cfg, logger)
    retriever, vector_store = build_retriever(
        paper_text=paper_text,
        tokenizer=tokenizer,
        rag_params=rag_params,
        logger=logger,
    )

    # Query with the full paper to select topical chunks (self-retrieval heuristic)
    context_text = build_context_from_retriever(retriever, paper_text, logger)

    # Prepare prompts
    template_context: Dict[str, Any] = {
        "paper": paper_text,
        "context": context_text,
        "model_response": model_response_text,
    }
    system_text = render_template(system_template, template_context)
    user_text = render_template(user_template, template_context)

    input_ids = build_input_ids(tokenizer, system_text, user_text)

    logger.info("Generating...")
    output_text, metrics = generate(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
    )

    metrics_line = (
        f"\n[metrics] input_tokens={metrics['input_tokens']} "
        f"output_tokens={metrics['output_tokens']} "
        f"total_tokens={metrics['total_tokens']} "
        f"latency_s={metrics['latency_seconds']:.3f} "
        f"output_tps={metrics['output_tokens_per_second']:.2f}"
    )
    print(metrics_line)
    logger.info(metrics_line)

    # Persist results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y_%m_%d_%H%M")
    results_path = results_dir / f"{timestamp}.txt"
    results_content = f"{metrics_line}\n\n{output_text}\n"
    write_text_file(results_path, results_content)
    print(f"[saved] {results_path}")
    logger.info(f"Saved results to {results_path}")

    # Ephemeral store: clear references so it can be GC'ed
    try:
        del vector_store
    except Exception:
        pass


if __name__ == "__main__":
    main()

