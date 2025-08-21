from pathlib import Path
from typing import Any, Callable, Tuple

import os
from moralkg import Config, get_logger

from .rag import RAG
from .cot import CoT

# TODO: implement and use logging across all classes.

def _download_model(path: str | Path, local: str | Path | None = None) -> str:
    """
    Download a Hugging Face repo snapshot to a local directory.
    If local is None, default to a directory under ./models/<repo_name>.
    Returns the absolute local directory path.
    """
    logger = get_logger(__name__)
    from huggingface_hub import snapshot_download

    repo_id = str(path)
    if local is None:
        repo_name = repo_id.split("/")[-1]
        local = Path("models") / repo_name
    local_dir = Path(local).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model '%s' to '%s'", repo_id, str(local_dir))
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    return str(local_dir)

def _check_for_model(config: dict) -> str:
    """
    Given a config dict with optional keys {"dir", "hf"}, resolve a local path with model files.
    - If "dir" is set and exists: return absolute path.
    - If "dir" is set but does not exist:
        - If "hf" is set: download to that directory and return it.
        - Else: raise.
    - If "dir" not set:
        - If "hf" is set: download to ./models/<repo_name> and return it.
        - Else: raise.
    """
    logger = get_logger(__name__)
    local_dir = (config.get("dir") if isinstance(config, dict) else None) or None
    hf_repo = (config.get("hf") if isinstance(config, dict) else None) or None

    if local_dir:
        p = Path(local_dir).resolve()
        if p.exists() and any(p.iterdir()):
            logger.info("Using existing local model directory: %s", str(p))
            return str(p)
        if hf_repo:
            logger.warning("Local dir missing or empty (%s); downloading from HF repo %s", str(p), hf_repo)
            return _download_model(hf_repo, local_dir)
        raise FileNotFoundError(f"Configured local model directory not found or empty: {p}")

    if hf_repo:
        logger.info("No local dir configured; downloading from HF repo %s", hf_repo)
        return _download_model(hf_repo)

    raise ValueError("Invalid model config: expected keys 'dir' or 'hf'")


def _get_device_and_dtype() -> Tuple["os.PathLike", Any]:
    """
    Model-agnostic device and dtype selection. Returns (device, dtype).
    """
    import torch
    use_cuda = torch.cuda.is_available()
    dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, dtype

# TODO: internal tooling to discover and use GPUs so that models are properly parallelized

class End2End:
    """
    End-to-end argument mining via LLM prompting.
    """
    def __init__(
        self,
        temperature: float = 0.7,
        max_new_tokens: int = 8192,
        rag: bool = False,
        rag_embedding_dims: int | None = None,
        rag_chunk_size: int | None = None,
        rag_chunk_overlap: int | None = None,
        rag_top_k: int | None = None,
        cot: bool = False,
        cot_steps: int | None = None,
        **kwargs: Any,
    ) -> None:

        # Flags and decoding params
        self.rag = bool(rag)
        self.cot = bool(cot)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)

        # RAG params
        self.embedding_dims = rag_embedding_dims
        self.chunk_size = rag_chunk_size
        self.chunk_overlap = rag_chunk_overlap
        self.top_k = rag_top_k or 5

        # CoT params
        self.cot_steps = cot_steps or 2

        # Misc
        self.kwargs = kwargs
        self.logger = get_logger(__name__)

        # Setup models
        self.model = Config.get("paths.models.end2end.base")
        self.adapter = Config.get("paths.models.end2end.finetune")
        self.embedder = Config.get("paths.models.end2end.embedder")

        # Resolve local model directories using config (model-agnostic)
        base_local = _check_for_model(self.model)
        adapter_local = _check_for_model(self.adapter)

        # Load generator (HF with PEFT adapter) - end2end-specific, defer import to avoid circular
        from .generator import load_generator_model
        self._hf_model, self._hf_tokenizer = load_generator_model(str(base_local), str(adapter_local))

        # Prepare RAG if enabled
        self._rag = None
        if self.rag:
            self._rag = RAG(
                embedder=self.embedder.get("hf") if isinstance(self.embedder, dict) else None,
                chunk_size=self.chunk_size or 1000,
                chunk_overlap=self.chunk_overlap or 100,
                top_k=self.top_k,
                keep_index=bool(self.kwargs.get("keep_embeddings", False)),
            )

    def generate(
        self,
        system_prompt: list[str] | str,
        user_prompt: list[str] | str,
        prompt_files: list[dict] | dict,
    ) -> dict:
        """
        Compose prompts, optionally retrieve context (RAG), optionally orchestrate reasoning (CoT), and generate output.
        Returns: {"text": str, "trace": dict}
        """
        debug = bool(self.kwargs.get("debug", False))

        # Normalize prompts
        system_text = self._normalize_prompt(system_prompt)
        user_text = self._normalize_prompt(user_prompt)

        # Resolve prompt_files: separate documents corpus from variables
        prompt_files = prompt_files or {}
        if isinstance(prompt_files, list):
            # Merge list of dicts left-to-right
            merged: dict[str, Any] = {}
            for entry in prompt_files:
                if not isinstance(entry, dict):
                    raise TypeError("prompt_files list must contain dict entries")
                merged.update(entry)
            prompt_files = merged
        if not isinstance(prompt_files, dict):
            raise TypeError("prompt_files must be a dict or list of dicts")

        documents = prompt_files.pop("documents", None)
        few_shot_examples = prompt_files.pop("few_shot_examples", None)

        # Interpolate variables in prompts from remaining prompt_files
        system_text = self._render_template(system_text, prompt_files)
        user_text = self._render_template(user_text, prompt_files)

        trace: dict[str, Any] = {
            "modes": {"rag": self.rag, "cot": self.cot},
            "decoding": {"temperature": self.temperature, "max_new_tokens": self.max_new_tokens},
            "files": list(prompt_files.keys()),
        }

        # Prepare RAG index if requested
        retrieval_callback: Callable[[str, int], list[dict]] | None = None
        if self._rag is not None:
            self._rag.build()
            if documents is not None:
                self._rag.add(documents)
            retrieval_callback = lambda q, k: self._rag.retrieve(q, k)
            trace["rag"] = {"top_k": self.top_k}

        try:
            if not self.cot:
                # Zero/Few shot with optional RAG
                context_blocks: list[str] = []
                citations: list[dict[str, Any]] = []
                if retrieval_callback is not None:
                    contexts = retrieval_callback(user_text, self.top_k)
                    for i, ctx in enumerate(contexts, start=1):
                        context_blocks.append(f"[{i}] {ctx.get('text','').strip()}")
                        citations.append({k: ctx.get(k) for k in ("id", "chunk_id", "score", "offsets", "metadata")})
                    trace["retrieved"] = citations

                composed_user = self._compose_user_prompt(user_text, few_shot_examples, context_blocks)
                from .generator import build_input_ids, generate as hf_generate
                input_ids = build_input_ids(self._hf_tokenizer, system_text, composed_user)
                text, metrics = hf_generate(
                    self._hf_model,
                    self._hf_tokenizer,
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                trace["metrics"] = metrics
                return {"text": text, "trace": trace}

            # CoT orchestration
            cot = CoT(
                steps=self.cot_steps,
                step_prompts=self.kwargs.get("step_prompts"),
                retrieval_step_positions=self.kwargs.get("retrieval_step_positions", [1] if self.rag else []),
                debug=debug,
            )
            generator_cb = lambda prompt: self._call_hf_generator(system_text, prompt)
            result = cot.run(
                user_prompt=user_text,
                generator=generator_cb,
                retrieve=retrieval_callback,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                few_shot_examples=few_shot_examples,
            )
            trace.update({"cot": result.get("steps", [])})
            return {"text": result.get("final", ""), "trace": trace}
        finally:
            if self._rag is not None and not bool(self.kwargs.get("keep_embeddings", False)):
                self._rag.destroy()

    def _render_template(self, template_text: str, context: dict) -> str:
        try:
            return template_text.format(**{k: self._read_path_or_text(v) for k, v in context.items()})
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise KeyError(f"Missing placeholder variable '{missing_key}' in prompt_files.")

    def _read_path_or_text(self, value: Any) -> str:
        # If value is a path to a file, read it; if dir, concatenate files; else return as string
        try:
            p = Path(str(value))
            if p.exists():
                if p.is_file():
                    return p.read_text(encoding="utf-8", errors="ignore")
                if p.is_dir():
                    parts: list[str] = []
                    for child in sorted(p.rglob("*")):
                        if child.is_file():
                            try:
                                parts.append(child.read_text(encoding="utf-8", errors="ignore"))
                            except Exception:
                                continue
                    return "\n\n".join(parts)
        except Exception:
            pass
        return str(value)

    def _compose_user_prompt(
        self,
        user_text: str,
        context_blocks: list[str] | None,
    ) -> str:
        blocks: list[str] = []
        if context_blocks:
            blocks.append("Context:\n" + "\n\n".join(context_blocks))
        blocks.append(user_text)
        return "\n\n".join(blocks)

    def _call_hf_generator(self, system_text: str, user_text: str) -> str:
        from .generator import generate_chat
        text, _ = generate_chat(
            self._hf_model,
            self._hf_tokenizer,
            system_text=system_text,
            user_text=user_text,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return text


class ADUR:
    """
    Argumentative Discourse Unit Recognition via pre-trained language model classification.
    """
    def __init__(
        self,
        model: str | Path | dict,
        *,
        extract_major_adus: bool = False,
        extraction_method: str | None = None,
        map: dict | None = None,
        **kwargs
    ):
        self.logger = get_logger(__name__)
        self.model = model
        self.major_only = extract_major_adus
        self.method = extraction_method if extraction_method in ["centroid", "pairwise"] else "centroid"
        # Default label mapping per project docs
        self.map = map or {}
        self.kwargs = kwargs

        # Device selection for pipelines that expect index
        try:
            import torch  # type: ignore
            self._device_index = 0 if torch.cuda.is_available() else -1
        except Exception:
            self._device_index = -1

        # Resolve local path or HF repo
        resolved = self._resolve_model_ref(self.model)

        # Load the ADUR (NER) pipeline via pytorch_ie AutoPipeline (SAM models)
        try:
            from pytorch_ie import AutoPipeline  # type: ignore
        except Exception as exc:
            raise ImportError(
                "pytorch_ie is required for ADUR with SAM models. Install pytorch-ie and pie_modules."
            ) from exc

        self._ner_pipeline = AutoPipeline.from_pretrained(
            resolved,
            device=self._device_index,
            taskmodule_kwargs={"combine_token_scores_method": "product"},
        )

    def _resolve_model_ref(self, model_ref: str | Path | dict) -> str:
        # Accept dict from config, or string/path
        if isinstance(model_ref, dict):
            return _check_for_model(model_ref)
        return str(model_ref)

    def _create_document(self, text: str):
        from pytorch_ie.annotations import LabeledSpan  # type: ignore
        from pytorch_ie.documents import (  # type: ignore
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        )

        document = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(text)
        document.id = "document"
        document.metadata = {}
        document.labeled_partitions.append(LabeledSpan(start=0, end=len(text), label="text"))
        return document

    def _translate_label(self, label: str) -> str:
        return self.map.get(label, label)

    def _extract_major_claims(self, adus: list[dict]) -> set[int]:
        # Choose candidates labelled as claim after mapping
        claim_indices: list[int] = [i for i, a in enumerate(adus) if a.get("label") == "claim"]
        if not claim_indices:
            return set()
        claim_texts = [adus[i]["text"] for i in claim_indices]

        # Embed with a small general embedding model (same as tests)
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except Exception as exc:
            self.logger.warning("sentence-transformers unavailable; skipping major ADU extraction: %s", exc)
            return set()

        model_name = self.kwargs.get("embedder_hf", "Qwen/Qwen3-Embedding-0.6B")
        embedder = SentenceTransformer(model_name)
        embeddings = embedder.encode(claim_texts, convert_to_tensor=True, show_progress_bar=False)
        import torch  # type: ignore
        centroid = embeddings.mean(dim=0, keepdim=True)
        sims = util.cos_sim(embeddings, centroid).squeeze()  # type: ignore
        sims_np = sims.detach().cpu().numpy()

        k = max(1, int(0.03 * len(claim_indices)))
        # Select top-k by similarity
        top_local = sims_np.argsort()[::-1][:k]
        return {claim_indices[int(i)] for i in top_local}

    def generate(self, input_file: str | Path) -> dict:
        """
        Run ADUR on a file and return extracted ADUs.

        Returns a dict with at least: {"adus": [...], "statistics": {...}}
        If self.major_only is True, adds major claim marking (label -> "major_claim").
        """
        input_path = Path(input_file)
        text = input_path.read_text(encoding="utf-8", errors="ignore")

        document = self._create_document(text)
        # Run NER/ADUR
        self._ner_pipeline(document, inplace=True)

        # Collect predictions
        adus: list[dict] = []
        for span in document.labeled_spans.predictions:
            start = getattr(span, "start", None)
            end = getattr(span, "end", None)
            label = getattr(span, "label", "")
            score = getattr(span, "score", None)
            adus.append(
                {
                    "text": document.text[start:end] if start is not None and end is not None else "",
                    "label": self._translate_label(label),
                    "original_label": label,
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )

        if self.major_only and adus:
            major_indices = self._extract_major_claims(adus)
            for i in major_indices:
                try:
                    adus[i]["label"] = "major_claim"
                    adus[i]["major"] = True
                except Exception:
                    continue

        stats = {
            "total_adus": len(adus),
            "adu_types": self._count_by_key(adus, "label"),
        }

        return {"adus": adus, "statistics": stats}

    def _count_by_key(self, items: list[dict], key: str) -> dict:
        result: dict[str, int] = {}
        for it in items:
            v = str(it.get(key))
            result[v] = result.get(v, 0) + 1
        return result

    """
    Class To-do:
    - load in model
    - NOTE: model loading/setup should occur in the INIT and persist across function calls to avoid unnecessary overhead by reloading models.
    """


class ARE:
    """
    Argumentative Relation Extraction via pre-trained language model classification.
    """
    def __init__(self, model: str | Path | dict, *, map: dict | None = None, **kwargs):
        self.logger = get_logger(__name__)
        self.model = model
        # Default relation mapping per project docs
        self.map = map or {}
        self.kwargs = kwargs

        # Device index for pytorch_ie
        try:
            import torch  # type: ignore
            self._device_index = 0 if torch.cuda.is_available() else -1
        except Exception:
            self._device_index = -1

        # Load pipelines (ADUR + ARE) as SAM requires spans first
        try:
            from pytorch_ie import AutoPipeline  # type: ignore
        except Exception as exc:
            raise ImportError(
                "pytorch_ie is required for ARE with SAM models. Install pytorch-ie and pie_modules."
            ) from exc

        adur_ref = self.kwargs.get("adur_model") or {"hf": "ArneBinder/sam-adur-sciarg"}
        are_ref = self._resolve_model_ref(self.model)
        adur_resolved = self._resolve_model_ref(adur_ref)

        self._ner_pipeline = AutoPipeline.from_pretrained(
            adur_resolved,
            device=self._device_index,
            taskmodule_kwargs={"combine_token_scores_method": "product"},
        )
        self._re_pipeline = AutoPipeline.from_pretrained(
            are_ref,
            device=self._device_index,
            taskmodule_kwargs={"collect_statistics": False},
        )

    def _resolve_model_ref(self, model_ref: str | Path | dict) -> str:
        if isinstance(model_ref, dict):
            return _check_for_model(model_ref)
        return str(model_ref)

    def _create_document(self, text: str):
        from pytorch_ie.annotations import LabeledSpan  # type: ignore
        from pytorch_ie.documents import (  # type: ignore
            TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions,
        )

        document = TextDocumentWithLabeledSpansBinaryRelationsAndLabeledPartitions(text)
        document.id = "document"
        document.metadata = {}
        document.labeled_partitions.append(LabeledSpan(start=0, end=len(text), label="text"))
        return document

    def _translate_relation(self, label: str) -> str:
        return self.map.get(label, label)

    def generate(self, input_file: str | Path) -> dict:
        """
        Run ADUR then ARE on a file and return extracted ADUs and Relations.

        Returns a dict with: {"adus": [...], "relations": [...], "statistics": {...}}
        """
        input_path = Path(input_file)
        text = input_path.read_text(encoding="utf-8", errors="ignore")

        document = self._create_document(text)

        # ADUR first
        self._ner_pipeline(document, inplace=True)

        # Move predicted entities to main layer for relation extraction
        predicted_entities = list(document.labeled_spans.predictions)
        document.labeled_spans.clear()
        document.labeled_spans.predictions.clear()
        document.labeled_spans.extend(predicted_entities)

        # ARE
        self._re_pipeline(document, inplace=True)

        # Collect ADUs
        adus: list[dict] = []
        for span in document.labeled_spans:
            start = getattr(span, "start", None)
            end = getattr(span, "end", None)
            label = getattr(span, "label", "")
            score = getattr(span, "score", None)
            # Reuse ADUR default mapping for ADU labels where available
            mapped_label = label
            if label in {"background_claim", "own_claim", "data"}:
                mapped_label = "claim" if label in {"background_claim", "own_claim"} else "premise"
            adus.append(
                {
                    "text": document.text[start:end] if start is not None and end is not None else "",
                    "label": mapped_label,
                    "original_label": label,
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )

        # Collect Relations and apply mapping/filtering
        relations: list[dict] = []
        for rel in document.binary_relations.predictions:
            raw_label = getattr(rel, "label", "")
            # Drop semantically_same and parts_of_same for simplicity
            if raw_label in {"semantically_same", "parts_of_same"}:
                continue
            label = self._translate_relation(raw_label)
            head = rel.head
            tail = rel.tail
            relations.append(
                {
                    "head_text": document.text[head.start:head.end],
                    "tail_text": document.text[tail.start:tail.end],
                    "label": label,
                    "original_label": raw_label,
                    "head_start": head.start,
                    "head_end": head.end,
                    "tail_start": tail.start,
                    "tail_end": tail.end,
                    "score": getattr(rel, "score", None),
                }
            )

        stats = {
            "total_adus": len(adus),
            "total_relations": len(relations),
            "adu_types": self._count_by_key(adus, "label"),
            "relation_types": self._count_by_key(relations, "label"),
        }

        return {"adus": adus, "relations": relations, "statistics": stats}

    def _count_by_key(self, items: list[dict], key: str) -> dict:
        result: dict[str, int] = {}
        for it in items:
            v = str(it.get(key))
            result[v] = result.get(v, 0) + 1
        return result

    """
    Class To-do:
    - 
    """
