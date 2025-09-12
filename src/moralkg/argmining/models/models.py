from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from types import SimpleNamespace

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


class _ShimLabeledSpans(list):
    """A tiny shim that provides the minimal API expected by the codebase:
    - acts like a list of span-like objects
    - has .predictions attribute (self) and clear()/extend()
    This lets the transformers fallback interoperate with code that expects
    pytorch_ie's LabeledSpans container.
    """
    def __init__(self, initial=None):
        initial = initial or []
        super().__init__(initial)
        # keep .predictions for backward-compatibility
        self.predictions = self

    def clear(self):
        super().clear()


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
        # Fail loudly: do not auto-download from HF. Require explicit local 'dir'.
        raise FileNotFoundError(
            f"Configured local model directory not found or empty: {p}. "
            "Automatic HF downloads are disabled; set 'dir' to a valid local snapshot."
        )

    # Do not accept bare HF refs: require explicit local directories in config
    raise ValueError(
        "Invalid model config: expected key 'dir' pointing to a local model directory. "
        "Automatic HF downloads are disabled for debugging."
    )


def _has_taskmodule_metadata(model_dir: str) -> bool:
    """Heuristic: return True if model_dir contains files suggesting a SAM/taskmodule snapshot.

    We look for taskmodule_config.json / taskmodule_config.yaml or any files referencing 'taskmodule' in filenames.
    """
    try:
        p = Path(model_dir)
        if not p.exists() or not p.is_dir():
            return False
        for fname in ("taskmodule_config.json", "taskmodule_config.yaml"):
            if (p / fname).exists():
                return True
        # fallback: any file name containing 'taskmodule'
        for f in p.rglob("*"):
            if "taskmodule" in f.name.lower():
                return True
    except Exception:
        return False
    return False


def _get_device_and_dtype(device_index: int | None = None) -> Tuple[Any, Any]:
    """
    Model-agnostic device and dtype selection. Returns (device, dtype).
    Accepts an optional device_index; if None will consult MORALKG_CUDA_DEVICE env var.
    """
    import torch
    use_cuda = torch.cuda.is_available()
    dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    )
    chosen = device_index
    if chosen is None:
        try:
            env_val = os.environ.get("MORALKG_CUDA_DEVICE")
            chosen = int(env_val) if env_val is not None else None
        except Exception:
            chosen = None

    if use_cuda:
        idx = chosen if chosen is not None else 0
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")
    return device, dtype


def _make_transformer_ner_pipeline(model_dir: str, device_index: int | None = None):
    """
    Create a small transformers-based token-classification pipeline as a fallback
    for model snapshots that don't include pytorch-ie taskmodule metadata.
    Returns a callable pipeline that accepts a document and populates
    document.labeled_spans.predictions with SimpleNamespace-like objects
    having attributes (start, end, label, score).
    """
    try:
        from transformers import pipeline as _hf_pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("'transformers' package is required for the NER fallback") from exc

    hf_device = int(device_index) if isinstance(device_index, int) and device_index >= 0 else -1

    class TransformerNERPipeline:
        def __init__(self, model_dir: str, device: int = -1):
            self._pipe = _hf_pipeline(
                task="token-classification",
                model=str(model_dir),
                aggregation_strategy="simple",
                device=device,
            )

        def __call__(self, document, inplace=True):
            text = getattr(document, "text", "")
            try:
                preds = self._pipe(text)
            except Exception:
                preds = []

            spans = []
            for p in preds:
                start = p.get("start")
                end = p.get("end")
                label = p.get("entity_group") or p.get("entity") or p.get("label")
                score = p.get("score")
                spans.append(SimpleNamespace(start=start, end=end, label=label, score=score))

            try:
                ls = document.labeled_spans
                if hasattr(ls, "predictions") and hasattr(ls.predictions, "clear"):
                    ls.predictions.clear()
                    ls.predictions.extend(spans)
                else:
                    # Wrap into shim for downstream compatibility
                    document.labeled_spans = _ShimLabeledSpans(spans)
            except Exception:
                document.labeled_spans = _ShimLabeledSpans(spans)

            return document

    return TransformerNERPipeline(model_dir, device=hf_device)


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

        # Get device and dtype for consistent tensor placement
        # Allow an explicit device index via kwargs: 'device_index' or 'cuda_device'
        device_index = None
        if isinstance(self.kwargs.get('device_index'), int):
            device_index = int(self.kwargs.get('device_index'))
        elif isinstance(self.kwargs.get('cuda_device'), int):
            device_index = int(self.kwargs.get('cuda_device'))

        self.device, self.dtype = _get_device_and_dtype(device_index=device_index)
        self.logger.info("Using device: %s, dtype: %s", self.device, self.dtype)

        # Create a config instance in order to set up models
        cfg = None
        try:
            cfg = Config.load()
        except Exception:
            cfg = None
        if cfg is not None:
            # Setup models
            self.model = cfg.get("paths.models.end2end.base", None)
            self.adapter = cfg.get("paths.models.end2end.finetune", None)
            self.embedder = cfg.get("paths.models.end2end.embedder", None)

        # Resolve local model directories using config (model-agnostic)
        base_local = _check_for_model(self.model)
        adapter_local = _check_for_model(self.adapter)

        # Load generator (HF with PEFT adapter) - end2end-specific, defer import to avoid circular
        from .generator import load_generator_model
        self._hf_model, self._hf_tokenizer = load_generator_model(
            base_model=str(base_local), 
            adapter_dir=str(adapter_local),
            device=self.device,
            dtype=self.dtype
        )

        # Prepare RAG if enabled
        self._rag = None
        if self.rag:
            self._rag = RAG(
                embedder=self.embedder.get("hf") if isinstance(self.embedder, dict) else None,
                chunk_size=self.chunk_size or 1000,
                chunk_overlap=self.chunk_overlap or 100,
                top_k=self.top_k,
                keep_index=bool(self.kwargs.get("keep_embeddings", False)),
                device=self.device,  # Pass device to RAG
            )

    def generate(
        self,
        system_prompt: list[str] | str,
        user_prompt: list[str] | str,
        prompt_files: dict | list[dict] | None, # Must be dict or list of dicts. Optional.
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
        prompt_files = prompt_files or {} # If prompt_files is None, use empty dict
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
        #few_shot_examples = prompt_files.pop("few_shot_examples", None) # Deprecated; this is now handled by generate_llm_prompts.py

        # Interpolate variables in prompts from remaining prompt_files
        #system_text = self._render_template(system_text, prompt_files) # Deprecated; this is now handled by generate_llm_prompts.py
        #user_text = self._render_template(user_text, prompt_files) # Deprecated; this is now handled by snowball_phase_1

        trace: dict[str, Any] = {
            "modes": {"rag": self.rag, "cot": self.cot},
            "decoding": {"temperature": self.temperature, "max_new_tokens": self.max_new_tokens},
            "files": list(prompt_files.keys()),
            "device": str(self.device),
            "dtype": str(self.dtype),
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

                #composed_user = self._compose_user_prompt(user_text, few_shot_examples, context_blocks) # TODO: Decide whether to handle few-shot examples here or elsewhere for CoT
                composed_user = self._compose_user_prompt(user_text, context_blocks)
                from .generator import build_input_ids, generate as hf_generate
                input_ids = build_input_ids(self._hf_tokenizer, system_text, composed_user, device=self.device)
                
                text, metrics = hf_generate(
                    self._hf_model,
                    self._hf_tokenizer,
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    device=self.device,
                    dtype=self.dtype
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
            generator_cb = lambda prompt: self._call_hf_generator(system_text, prompt, device=self.device)
            result = cot.run(
                user_prompt=user_text,
                generator=generator_cb,
                retrieve=retrieval_callback,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                #few_shot_examples=few_shot_examples, # TODO: Decide whether to handle few-shot examples here or elsewhere for CoT
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

    def _normalize_prompt(self, prompt: str) -> str:
        # Normalize whitespace and line breaks in the prompt
        return "\n".join(line.strip() for line in prompt.splitlines() if line.strip())

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
            device=self.device,  # Pass device to ensure consistency
        )
        return text


class ADUR:
    """
    Argumentative Discourse Unit Recognition via pre-trained language model classification.
    """
    def __init__(
        self,
        model: str | Path | dict | None = None,
        *,
        extract_major_adus: bool = False,
        extraction_method: str | None = None,
        map: dict | None = None,
        use_model_2: bool = False,
        **kwargs
    ):
        self.logger = get_logger(__name__)
        self.major_only = extract_major_adus
        self.method = extraction_method if extraction_method in ["centroid", "pairwise"] else "centroid"
        # Default label mapping per project docs
        self.map = map or {}
        self.kwargs = kwargs

        # Load model from config if not provided
        if model is None:
            cfg = None
            try:
                cfg = Config.load()
            except Exception:
                cfg = None
            
            if cfg is not None:
                # Choose between model_1 and model_2 based on parameter
                model_key = "model_2" if use_model_2 else "model_1"
                model = cfg.get(f"paths.models.adur.{model_key}", None)
                
            if model is None:
                # Fallback to default SAM model if config fails
                model = {"hf": "ArneBinder/sam-adur-sciarg"}
                self.logger.warning("Could not load ADUR model from config, using default SAM model")

        self.model = model

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

        # Try to create the pipeline with our preferred taskmodule kwargs.
        # Some pytorch-ie / pie_modules versions do not accept all kwargs; in
        # that case retry once without passing taskmodule_kwargs so the
        # pipeline can still be instantiated. We keep the failure loud and
        # surface a clear remediation message when we have to fall back.
        try:
            self._ner_pipeline = AutoPipeline.from_pretrained(
                resolved,
                device=self._device_index,
                taskmodule_kwargs={"combine_token_scores_method": "product"},
            )
        except TypeError as exc:
            # Detect the common unexpected-kwarg failure originating from
            # HyperparametersMixin / TaskModule __init__ chains.
            msg = str(exc)
            if "combine_token_scores_method" in msg or "unexpected keyword" in msg.lower():
                # Emit a strong warning with remediation guidance and retry.
                self.logger.warning(
                    "AutoPipeline.from_pretrained rejected taskmodule kwargs: %s\n"
                    "Retrying without taskmodule_kwargs. If this succeeds, consider "
                    "upgrading/downgrading your pytorch-ie / pie-modules packages to a "
                    "version that supports the 'combine_token_scores_method' hyperparameter.",
                    msg,
                )
                # Retry without taskmodule_kwargs
                self._ner_pipeline = AutoPipeline.from_pretrained(
                    resolved,
                    device=self._device_index,
                )
            else:
                # Re-raise for any other unexpected TypeError
                raise
        except KeyError as exc:
            # This commonly means the model snapshot does not include a
            # taskmodule_config.json or the AutoTaskModule expected keys like
            # 'taskmodule_type'. In practice many roberta-style checkpoints
            # are plain token-classification models and don't ship SAM task
            # metadata. Provide a lightweight transformers-based fallback that
            # performs token-classification and populates the document's
            # labeled_spans.predictions to preserve downstream behaviour.
            msg = str(exc)
            self.logger.warning(
                "AutoPipeline.from_pretrained failed with KeyError '%s' - using transformers token-classification fallback."
                " This fallback performs best-effort span extraction but may differ from the original SAM behavior.",
                msg,
            )

            # Lazy import and create a small wrapper around transformers' pipeline
            try:
                from transformers import pipeline as _hf_pipeline  # type: ignore

                # Map device index to transformers' device arg (int: cuda_index or -1)
                hf_device = int(self._device_index) if isinstance(self._device_index, int) else -1

                class TransformerNERPipeline:
                    def __init__(self, model_dir: str, device: int = -1):
                        # aggregation_strategy='simple' returns merged spans with start/end offsets
                        self._pipe = _hf_pipeline(
                            task="token-classification",
                            model=str(model_dir),
                            aggregation_strategy="simple",
                            device=device,
                        )

                    def __call__(self, document, inplace=True):
                        text = getattr(document, "text", "")
                        try:
                            preds = self._pipe(text)
                        except Exception:
                            # If transformer pipeline fails, leave predictions empty
                            preds = []

                        spans = []
                        for p in preds:
                            start = p.get("start")
                            end = p.get("end")
                            label = p.get("entity_group") or p.get("entity") or p.get("label")
                            score = p.get("score")
                            spans.append(SimpleNamespace(start=start, end=end, label=label, score=score))

                        # Ensure the document has labeled_spans and a predictions list
                        try:
                            ls = document.labeled_spans
                            if hasattr(ls, "predictions") and hasattr(ls.predictions, "clear"):
                                ls.predictions.clear()
                                ls.predictions.extend(spans)
                            else:
                                ls.predictions = spans
                        except Exception:
                            document.labeled_spans = SimpleNamespace(predictions=spans)

                        return document

                self._ner_pipeline = TransformerNERPipeline(resolved, device=hf_device)
            except Exception:
                # If transformers isn't available or the wrapper fails, raise a clearer error
                raise RuntimeError(
                    "Failed to instantiate a transformers-based fallback for ADUR. "
                    "Install 'transformers' (and a compatible model snapshot) or provide a SAM snapshot with taskmodule_config.json."
                ) from exc

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
            "adu_types": _count_by_key(adus, "label"),
        }

        return {"adus": adus, "statistics": stats}

    def _count_by_key(self, items: list[dict], key: str) -> dict:
        result: dict[str, int] = {}
        for it in items:
            v = str(it.get(key))
            result[v] = result.get(v, 0) + 1
        return result


def _count_by_key(items: list[dict], key: str) -> dict:
    result: dict[str, int] = {}
    for it in items:
        v = str(it.get(key))
        result[v] = result.get(v, 0) + 1
    return result


class ARE:
    """
    Argumentative Relation Extraction via pre-trained language model classification.
    """
    def __init__(
        self, 
        model: str | Path | dict | None = None, 
        *, 
        map: dict | None = None, 
        use_model_2: bool = False,
        adur_model: str | Path | dict | None = None,
        use_adur_model_2: bool = False,
        **kwargs
    ):
        self.logger = get_logger(__name__)
        # Default relation mapping per project docs
        self.map = map or {}
        self.kwargs = kwargs

        # Load models from config if not provided
        cfg = None
        try:
            cfg = Config.load()
        except Exception:
            cfg = None

        # Handle ARE model
        if model is None and cfg is not None:
            model_key = "model_2" if use_model_2 else "model_1"
            model = cfg.get(f"paths.models.are.{model_key}", None)
        
        if model is None:
            # Fallback to default SAM model
            model = {"hf": "ArneBinder/sam-are-sciarg"}
            self.logger.warning("Could not load ARE model from config, using default SAM model")

        # Handle ADUR model (needed for preprocessing)
        if adur_model is None and cfg is not None:
            adur_model_key = "model_2" if use_adur_model_2 else "model_1"
            adur_model = cfg.get(f"paths.models.adur.{adur_model_key}", None)
        
        if adur_model is None:
            # Fallback to default SAM ADUR model
            adur_model = {"hf": "ArneBinder/sam-adur-sciarg"}
            self.logger.warning("Could not load ADUR model from config, using default SAM model")

        self.model = model
        self.adur_model = adur_model

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

        are_resolved = self._resolve_model_ref(self.model)
        adur_resolved = self._resolve_model_ref(self.adur_model)

        # Create ADUR (NER) pipeline with defensive fallback like ADUR class
        # Decide deterministically whether ADUR (for preprocessing) should be SAM or transformers
        adur_has_taskmeta = _has_taskmodule_metadata(adur_resolved)
        if adur_has_taskmeta:
            try:
                self._ner_pipeline = AutoPipeline.from_pretrained(
                    adur_resolved,
                    device=self._device_index,
                    taskmodule_kwargs={"combine_token_scores_method": "product"},
                )
            except TypeError as exc:
                msg = str(exc)
                if "combine_token_scores_method" in msg or "unexpected keyword" in msg.lower():
                    self.logger.warning(
                        "ADUR AutoPipeline.from_pretrained rejected taskmodule kwargs: %s -- retrying without kwargs.",
                        msg,
                    )
                    self._ner_pipeline = AutoPipeline.from_pretrained(adur_resolved, device=self._device_index)
                else:
                    raise
        else:
            # ADUR fallback
            try:
                self._ner_pipeline = _make_transformer_ner_pipeline(adur_resolved, device_index=self._device_index)
                self.logger.info("Using transformers-based NER fallback for ADUR (no taskmodule metadata found for ADUR model).")
            except Exception as exc:
                raise RuntimeError("Failed to create ADUR transformers fallback: %s" % exc) from exc

        # Decide deterministically whether ARE should be SAM or NullARE fallback
        are_has_taskmeta = _has_taskmodule_metadata(are_resolved)
        if are_has_taskmeta:
            try:
                self._re_pipeline = AutoPipeline.from_pretrained(
                    are_resolved,
                    device=self._device_index,
                    taskmodule_kwargs={"collect_statistics": False},
                )
            except TypeError as exc:
                msg = str(exc)
                if "collect_statistics" in msg or "unexpected keyword" in msg.lower():
                    self.logger.warning(
                        "ARE AutoPipeline.from_pretrained rejected taskmodule kwargs: %s -- retrying without kwargs.",
                        msg,
                    )
                    self._re_pipeline = AutoPipeline.from_pretrained(are_resolved, device=self._device_index)
                else:
                    raise
        else:
            # Deterministic NullARE fallback when ARE snapshot lacks taskmodule metadata
            self.logger.warning(
                "ARE model lacks taskmodule metadata; using NullARE fallback (no relations will be produced)."
            )

            class NullARE:
                def __call__(self, document, inplace=True):
                    try:
                        if not hasattr(document, 'binary_relations'):
                            document.binary_relations = SimpleNamespace(predictions=[])
                        else:
                            if hasattr(document.binary_relations, 'predictions'):
                                try:
                                    document.binary_relations.predictions.clear()
                                except Exception:
                                    document.binary_relations.predictions = []
                    except Exception:
                        document.binary_relations = SimpleNamespace(predictions=[])
                    return document

            self._re_pipeline = NullARE()

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
            "adu_types": _count_by_key(adus, "label"),
            "relation_types": _count_by_key(relations, "label"),
        }

        return {"adus": adus, "relations": relations, "statistics": stats}