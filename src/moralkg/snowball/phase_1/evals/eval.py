"""
Modular generation-time evaluator that loads a (merged or base+LoRA) model,
runs deterministic generation over a validation split, parses JSON outputs,
and computes task metrics via a pluggable Metrics implementation.

This coexists with the legacy `evaluator.py` (which evaluates already-built
ArgumentMap objects). Use this Evaluator for HPO loops that need to score a
checkpoint produced by different trainers (LLaMA-Factory / Unsloth+TRL).

TODO: Figure out if model loading is necessary, since generating argument maps doesn't need to happen in this module.
TODO: Refactor this class to be less monolithic, e.g. separate out model loading, inference, and evaluation.
TODO: Refactor both this class and the End2EndEvaluator to share more code and distribute responsibilities appropriately.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .retrievers import BaseRetriever, NoopRetriever
from .prompts import BasePromptBuilder, DefaultPromptBuilder
from .metrics_modular import BaseMetrics
from .datasets import BaseDatasetAdapter
from ....argmining.schemas import ArgumentMap
from .metrics import combined_score as legacy_combined_score


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class ModelConfig:
    base_model: Optional[str] = None  # required if loading LoRA adapters
    dtype: str = "float16"  # "float16" | "bfloat16"
    device_map: str = "auto"
    merge_adapters: bool = True  # merge PEFT adapters for eval


class Evaluator:
    """
    Composable external evaluator:
      - loads (merged) model or base+adapters from checkpoint_dir
      - (optional) runs retriever to add context (RAG)
      - builds prompts (with/without CoT)
      - decodes -> parses -> computes metrics and cost
    """

    def __init__(
        self,
        dataset: BaseDatasetAdapter,
        metrics: BaseMetrics,
        retriever: Optional[BaseRetriever] = None,
        prompt_builder: Optional[BasePromptBuilder] = None,
        model_cfg: Optional[ModelConfig] = None,
        gen_cfg: Optional[GenerationConfig] = None,
        use_rag: bool = False,
        use_cot: bool = False,
    ):
        self.dataset = dataset
        self.metrics = metrics
        self.retriever = retriever or NoopRetriever()
        self.prompt_builder = prompt_builder or DefaultPromptBuilder()
        self.model_cfg = model_cfg or ModelConfig()
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.use_rag = use_rag
        self.use_cot = use_cot
        self.model = None
        self.tok = None

    # ---------- public API ----------
    def evaluate_checkpoint(self, checkpoint_dir: str, split: str = "validation") -> Dict[str, float]:
        self._load_model(checkpoint_dir)
        preds, golds = self._infer_and_collect(split)
        return self.metrics.compute(preds, golds)

    # ---------- model loading ----------
    def _load_model(self, checkpoint_dir: str) -> None:
        ckpt = pathlib.Path(checkpoint_dir)
        dtype = torch.bfloat16 if self.model_cfg.dtype == "bfloat16" else torch.float16

        # Case A: merged model exists
        if (ckpt / "config.json").exists() and any(
            (ckpt / f).exists() for f in ["pytorch_model.bin", "model.safetensors"]
        ):
            self.tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt, torch_dtype=dtype, device_map=self.model_cfg.device_map
            )
            return

        # Case B: adapters only -> need base model
        base = self.model_cfg.base_model or self._infer_base_from_peft(ckpt)
        if base is None:
            raise ValueError("Base model required for adapters; set ModelConfig.base_model")
        self.tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=dtype, device_map=self.model_cfg.device_map
        )
        self.model = PeftModel.from_pretrained(base_model, ckpt)
        if self.model_cfg.merge_adapters:
            self.model = self.model.merge_and_unload()

    def _infer_base_from_peft(self, ckpt: pathlib.Path) -> Optional[str]:
        cfg = ckpt / "adapter_config.json"
        if cfg.exists():
            data = json.loads(cfg.read_text())
            return data.get("base_model_name_or_path")
        return None

    # ---------- inference ----------
    @torch.inference_mode()
    def _infer_and_collect(self, split: str):
        preds: List[Tuple[str, Dict[str, Any]]] = []
        golds: Dict[str, Dict[str, Any]] = {}

        for ex_id, text, gold in self.dataset.iter(split):
            contexts = self.retriever.retrieve(text, k=3) if self.use_rag else []
            prompt = self.prompt_builder.build(text, contexts=contexts, use_cot=self.use_cot)
            inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                do_sample=self.gen_cfg.do_sample,
            )
            gen = self.tok.decode(out[0], skip_special_tokens=True)
            parsed = self._extract_json(gen)
            preds.append((ex_id, parsed))
            golds[ex_id] = gold
        return preds, golds

    # ---------- parsing ----------
    def _extract_json(self, text: str) -> Dict[str, Any]:
        # Simple last-JSON-block heuristic
        matches = list(re.finditer(r"\{.*\}", text, flags=re.S))
        if not matches:
            return {}
        try:
            return json.loads(matches[-1].group(0))
        except Exception:
            return {}

    # ---------- direct map evaluation (merged legacy functionality) ----------
    def evaluate_argument_maps_single(self, gold_map: ArgumentMap, pred_map: ArgumentMap, threshold: float = 0.7) -> Dict[str, float]:
        """Evaluate a single predicted map against a gold map using legacy metrics."""
        return legacy_combined_score(gold_map, pred_map, threshold)

    def evaluate_argument_maps_batch(self, gold_maps: List[ArgumentMap], pred_maps: List[ArgumentMap], threshold: float = 0.7) -> Dict[str, Any]:
        """Evaluate lists of maps and return per-map and average metrics."""
        if len(gold_maps) != len(pred_maps):
            raise ValueError("gold_maps and pred_maps must be same length")
        per_map = []
        sums: Dict[str, float] = {}
        count = 0
        for g, p in zip(gold_maps, pred_maps):
            m = legacy_combined_score(g, p, threshold)
            per_map.append({"gold_id": g.id, "pred_id": p.id, **m})
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            count += 1
        avg = {f"avg_{k}": (sums[k] / count if count else 0.0) for k in sums}
        return {"average_metrics": avg, "per_map_results": per_map}


