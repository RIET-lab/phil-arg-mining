
# TODO: Implement and use logging across all classes.
# TODO: Load documents as a batch, rather than jump through hoops to load them one by one. Batch loading should also be much faster anyway.
# TODO: Fix the issue with sam_are_sciarg's config.json and taskmodule_config.json having unexpected keys that cause AutoPipeline.from_pretrained to fail.
# -> This might be because the config files are being replaced by new downloads from HF on each attempt to fix them. Consider downloading once to a temp dir, fixing, and then enforcing loading from there.
# -> Alternatively, import the config files directly and remove unexpected keys before passing to AutoPipeline.from_pretrained via the 'config' argument.

import os
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Callable
from types import SimpleNamespace

# Optional heavy deps; imported lazily where possible
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


from moralkg import Config, get_logger

from .rag import RAG
from .cot import CoT


# ----------------------------- Utilities ------------------------------------

def _run_sam_single(pipeline, item):
    """
    Run a pytorch_ie AutoPipeline on a single item without in-place mutation
    and return the (first) processed document.

    - Accepts a Document-like object (e.g., your _make_document(text) result).
    - Feeds as a singleton sequence to avoid InplaceNotSupportedException.
    - Uses inplace=False to avoid mutating immutable containers.
    - Materializes TaskEncodingSequence (if returned) to a list.
    """
    # Run on a singleton batch with no in-place mutation
    seq = pipeline([item], inplace=False)

    # Some pipelines return a TaskEncodingSequence that must be materialized
    if hasattr(seq, "materialize") and callable(getattr(seq, "materialize")):
        docs = seq.materialize()
    else:
        # Fallback: try to listify
        docs = list(seq)

    if not docs:
        raise RuntimeError("SAM pipeline returned no documents")
    return docs[0]


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


def _ensure_torch():
    if torch is None:
        raise RuntimeError(
            "This module requires PyTorch. Please `pip install torch` "
            "compatible with your Python/CUDA setup."
        )


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



def _has_taskmodule_metadata(model_dir: str) -> bool:
    """Heuristic: SAM snapshots ship a taskmodule_config.json next to weights."""
    if not model_dir:
        return False
    tm_path = os.path.join(str(model_dir), "taskmodule_config.json")
    return os.path.isfile(tm_path)


def _resolve_local_dir(model_dir: str) -> str:
    """This code only loads from a local directory; no auto-downloads."""
    if not model_dir or not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir!r}. "
            "Provide a local snapshot containing config.json & weights."
        )
    cfg = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    return model_dir


def _simple_sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """Tiny, dependency-free sentence splitter:
    split on newlines and punctuation+space. Returns (start, end, sent_text).
    """
    spans: List[Tuple[int, int, str]] = []
    if not text:
        return spans
    import re
    cursor = 0
    for para in text.splitlines(True):
        s = para.strip()
        if not s:
            cursor += len(para)
            continue
        parts = re.split(r'(?<=[\.\?\!])\s+', s)
        for part in parts:
            if not part:
                continue
            start = text.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            spans.append((start, end, part))
            cursor = end
        cursor += len(para) - len(para.strip())
    return spans


# ------------------------ Document / Shim containers -------------------------

class _ShimDocument(SimpleNamespace):
    # Minimal duck-typing to satisfy pytorch_ie pipelines
    def as_type(self, _t):
        # Many pipelines just call .as_type(document_type) and then read fields.
        # Returning self keeps our SimpleNamespace-compatible object flowing through.
        return self
    
class _ShimLabeledSpans(list):
    """A list that also exposes a .predictions view (SAM-like)."""
    def __init__(self, initial=None):
        super().__init__(initial or [])
        self.predictions = self

    def clear(self):  # type: ignore[override]
        super().clear()


class _ShimRelations(list):
    """A list that also exposes a .predictions view (SAM-like)."""
    def __init__(self, initial=None):
        super().__init__(initial or [])
        self.predictions = self

    def clear(self):  # type: ignore[override]
        super().clear()


def _make_document(text: str):
    # HF-only convenience container
    doc = _ShimDocument()
    doc.text = text
    doc.labeled_spans = _ShimLabeledSpans()
    doc.binary_relations = _ShimRelations()
    return doc

def _make_sam_document(pipeline, text: str):
    """
    Build the correct pytorch_ie Document for this pipeline.
    Prefers pipeline.taskmodule.document_type if available; falls back to TextDocument.
    Handles constructors that:
      - accept keyword 'text'
      - accept positional text only
      - accept no args (we setattr afterwards)
    """
    from pytorch_ie.documents import TextBasedDocument, TextDocumentWithLabeledSpans
    # Set the document class as TextDocumentWithLabeledSpans by default, since ADUR and ARE expect LabeledSpans
    TextDocumentCls = TextBasedDocument
    DocumentSpanCls = TextDocumentWithLabeledSpans

    tm = getattr(pipeline, "taskmodule", None) 
    if tm is not None:
        TextDocumentCls = getattr(tm, "document_type", None)
        if TextDocumentCls is not None and not callable(TextDocumentCls):
            TextDocumentCls = None
    if DocumentSpanCls is None:
        DocumentSpanCls = TextDocumentWithLabeledSpans

    return DocumentSpanCls(text=text)

# -------------------- Lightweight HF classifier wrapper ----------------------


class _HFSequenceClassifier:
    """Thin wrapper around AutoModelForSequenceClassification for batched inference.
    Handles both single-text and (text, text_pair) classification.
    """
    def __init__(
        self,
        model_dir: str,
        device,
        dtype,
        use_fp16: Optional[bool] = None,
        force_cpu: bool = False,
    ):
        _ensure_torch()
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        if force_cpu:
            device = torch.device("cpu")
            dtype = torch.float32

        self.device = device
        # Only use FP16 on CUDA; otherwise float32.
        self.dtype = torch.float16 if (use_fp16 and device.type == "cuda") else (
            dtype if device.type == "cuda" else torch.float32
        )

        self.cfg = AutoConfig.from_pretrained(model_dir)
        self.id2label = {int(k): v for k, v in getattr(self.cfg, "id2label", {}).items()}
        self.label2id = {k: int(v) for k, v in getattr(self.cfg, "label2id", {}).items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # If the model was saved in full-precision only, keep compute in fp32
        # but allow autocast on CUDA if dtype is fp16.
        self._autocast_enabled = (self.device.type == "cuda" and self.dtype == torch.float16)

        # Convenience
        self.num_labels = int(getattr(self.model.config, "num_labels", len(self.id2label) or 2))

    def _softmax_top(self, logits):
        probs = torch.softmax(logits, dim=-1)
        scores, ids = probs.max(dim=-1)
        return ids.detach().cpu().tolist(), scores.detach().cpu().tolist()

    @torch.no_grad()
    def classify_texts(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        out: List[dict] = []
        if not texts:
            return out
        autocast = torch.cuda.amp.autocast if self._autocast_enabled else None
        rng = range(0, len(texts), batch_size)
        for i in rng:
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if autocast:
                with autocast(dtype=self.dtype):  # type: ignore
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits
            ids, scores = self._softmax_top(logits)
            for idx, sc in zip(ids, scores):
                label = self.id2label.get(idx, str(idx))
                out.append({"label": label, "score": float(sc)})
        return out

    @torch.no_grad()
    def classify_pairs(
        self, heads: List[str], tails: List[str], batch_size: int = 16
    ) -> List[dict]:
        assert len(heads) == len(tails), "heads/tails length mismatch"
        out: List[dict] = []
        if not heads:
            return out
        autocast = torch.cuda.amp.autocast if self._autocast_enabled else None
        rng = range(0, len(heads), batch_size)
        for i in rng:
            h = heads[i : i + batch_size]
            t = tails[i : i + batch_size]
            enc = self.tokenizer(
                h, t, truncation=True, padding=True, max_length=512, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if autocast:
                with autocast(dtype=self.dtype):  # type: ignore
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits
            ids, scores = self._softmax_top(logits)
            for idx, sc in zip(ids, scores):
                label = self.id2label.get(idx, str(idx))
                out.append({"label": label, "score": float(sc)})
        return out


# ------------------------------- ADUR ----------------------------------------


class ADUR:
    """Argumentative Discourse Unit Recognition.
    If a SAM (pytorch_ie) pipeline is available for the given local dir, uses it.
    Otherwise, uses a HF sequence classifier and turns positive sentence labels
    into span predictions.
    """

    def __init__(
        self,
        model_dir: str,
        device_index: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.kwargs = dict(kwargs)
        self.model_dir = _resolve_local_dir(model_dir)

        # Device + dtype
        self.device, self.dtype = _get_device_and_dtype(device_index)
        self._device_index = device_index if device_index is not None else (0 if self.device.type == "cuda" else -1)

        # Prefer SAM if taskmodule metadata exists
        self._mode = "sam" if _has_taskmodule_metadata(self.model_dir) else "hf"
        self._sam_pipeline = None
        self._hf = None

        if self._mode == "sam":
            from pytorch_ie import AutoPipeline  # type: ignore

            #from pytorch_ie.taskmodules import LabeledSpanExtractionByTokenClassificationTaskModule

            self._sam_pipeline = AutoPipeline.from_pretrained(
                self.model_dir,
                device=self._device_index,
            )

            self.logger.info("ADUR using SAM AutoPipeline.")

        if self._mode == "hf":
            # Build HF classifier and determine 'positive' label
            self._hf = _HFSequenceClassifier(
                self.model_dir, device=self.device, dtype=self.dtype, use_fp16=self.kwargs.get("use_fp16", True)
            )
            # Choose 'positive' ADU label; default keep anything not NON-ARGUMENT
            self._positive_label = None
            for k in self._hf.label2id:
                if k.upper() in {"ARGUMENT", "ADU", "POSITIVE"}:
                    self._positive_label = k
                    break
            self.logger.info("ADUR(HF) positive label = %s", self._positive_label or "not NON-ARGUMENT")

    def _adur_hf(self, document) -> None:
        text: str = document.text
        spans = _simple_sentence_spans(text)
        if not spans:
            document.labeled_spans = _ShimLabeledSpans()
            return
        inputs = [s for (_s, _e, s) in spans]
        bs = int(self.kwargs.get("batch_size", 32))
        results = self._hf.classify_texts(inputs, batch_size=bs)  # type: ignore
        preds = []
        for (start, end, _), res in zip(spans, results):
            label = str(res.get("label", ""))
            score = float(res.get("score", 0.0))
            keep = (label == self._positive_label) if self._positive_label else (
                label.upper() not in {"NON-ARGUMENT", "NONE", "O", "NEGATIVE", "NO"}
            )
            if keep:
                preds.append(SimpleNamespace(start=start, end=end, label=label, score=score))
        document.labeled_spans = _ShimLabeledSpans(preds)

    def generate(self, text: str) -> dict:
        """Return a dict with:
            - spans: list[{start,end,label,score}]
            - document: object (with .labeled_spans.predictions)
        """
        if self._mode == "sam":
            document = _make_sam_document(self._sam_pipeline, text)
            # Feed a sequence, request non-inplace processing, then materialize
            document = _run_sam_single(self._sam_pipeline, document)
            spans = [
                {"start": s.start, "end": s.end,
                "label": getattr(s, "label", "ARGUMENT"),
                "score": float(getattr(s, "score", 1.0))}
                for s in getattr(document.labeled_spans, "predictions", [])
            ]
        else:
            document = _make_document(text)
            self._adur_hf(document)
            spans = [
                {
                    "start": s.start,
                    "end": s.end,
                    "label": getattr(s, "label", "ARGUMENT"),
                    "score": float(getattr(s, "score", 1.0)),
                } 
                for s in document.labeled_spans.predictions
            ]
        return {"spans": spans, "document": document}


# --------------------------------- ARE ---------------------------------------


class ARE:
    """Argumentative Relation Extraction.
    Runs ADUR to get spans, then classifies ordered span pairs with either:
      - SAM/PIE relation extractor (if available), or
      - HF sequence-classifier on (head_text, tail_text) pairs.
    """

    def __init__(
        self,
        model_dir: str,
        adur_model_dir: Optional[str] = None,
        device_index: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.kwargs = dict(kwargs)
        self.model_dir = _resolve_local_dir(model_dir)

        # Device + dtype shared by both submodels
        self.device, self.dtype = _get_device_and_dtype(device_index)
        self._device_index = device_index if device_index is not None else (0 if self.device.type == "cuda" else -1)

        # Build / share ADUR
        if adur_model_dir is None:
            raise ValueError("ARE requires adur_model_dir to produce ADU spans before relation extraction.")
        self.adur = ADUR(
            model_dir=_resolve_local_dir(adur_model_dir),
            device_index=device_index,
            logger=logging.getLogger("ARE.ADU"),
            **{k: v for k, v in kwargs.items() if k in {"use_fp16", "batch_size"}}
        )

        # ARE: prefer SAM if taskmodule is present; else HF
        self._mode = "sam" if _has_taskmodule_metadata(self.model_dir) else "hf"
        self._sam_pipeline = None
        self._hf = None

        if self._mode == "sam":
            from pytorch_ie import AutoPipeline  # type: ignore
            #from pytorch_ie.taskmodules import LabeledSpanExtractionByTokenClassificationTaskModule

            # Find the config.json and taskmodule_config.json files from this model_dir, and cache them to a temp dir
            # to avoid repeated downloads from HF on each attempt to fix unexpected keyword args.
            config_files = ["config.json", "taskmodule_config.json"]
            temp_dir = os.path.join("/tmp", f"sam_are_temp_{os.getpid()}")
            os.makedirs(temp_dir, exist_ok=True)
            for cfg_file in config_files:
                src_path = os.path.join(self.model_dir, cfg_file)
                dst_path = os.path.join(temp_dir, cfg_file)
                if os.path.isfile(src_path):
                    try:
                        with open(src_path, "r", encoding="utf-8") as f_src:
                            data = json.load(f_src)
                        with open(dst_path, "w", encoding="utf-8") as f_dst:
                            json.dump(data, f_dst, indent=2)
                        self.logger.info("Copied %s to temporary directory %s", cfg_file, temp_dir)
                    except Exception as ex:
                        self.logger.error("Failed to copy %s to %s: %s", cfg_file, temp_dir, str(ex))
                else:
                    self.logger.warning("Expected config file not found: %s", src_path)

            attempts = 0
            while not self._sam_pipeline:
                try:
                    self._sam_pipeline = AutoPipeline.from_pretrained(
                        self.model_dir,
                        device=self._device_index,
                        local_files_only=True,
                        config=self._get_config() # TODO: Switch this out with a proper argument format, and make it use the appropriate files
                    )
                except Exception as e: # TODO: Also update this part to use the appropriate files
                    # Catch failures caused by "HyperparametersMixin.init() got an unexpected keyword argument '<something>'"
                    # Record the keyword to the log
                    # Remove the keyword from the config.json and taskmodulesconfig.json files in the model_dir
                    # Loop until the pipeline loads successfully
                    if hasattr(e, "args") and e.args:
                        msg = str(e.args[0])
                        if "unexpected keyword argument" in msg:
                            kw = msg.split("unexpected keyword argument")[-1].strip().strip("'\"")
                            self.logger.error("Removing unexpected keyword argument from config: %s", kw)
                            # Remove from config.json
                            cfg_path = os.path.join(self.model_dir, "config.json")
                            try:
                                with open(cfg_path, "r", encoding="utf-8") as f:
                                    cfg = json.load(f)
                                if kw in cfg:
                                    del cfg[kw]
                                    with open(cfg_path, "w", encoding="utf-8") as f:
                                        json.dump(cfg, f, indent=2)
                                    self.logger.info("Removed %s from %s", kw, cfg_path)
                            except Exception as ex:
                                self.logger.error("Failed to update %s: %s", cfg_path, str(ex))
                            # Remove from taskmodule_config.json
                            tm_cfg_path = os.path.join(self.model_dir, "taskmodule_config.json")
                            try:
                                with open(tm_cfg_path, "r", encoding="utf-8") as f:
                                    tm_cfg = json.load(f)
                                if kw in tm_cfg:
                                    del tm_cfg[kw]
                                    with open(tm_cfg_path, "w", encoding="utf-8") as f:
                                        json.dump(tm_cfg, f, indent=2)
                                    self.logger.info("Removed %s from %s", kw, tm_cfg_path)
                            except Exception as ex:
                                self.logger.error("Failed to update %s: %s", tm_cfg_path, str(ex))

                            attempts += 1
                            if attempts > 10:
                                self.logger.error("Too many attempts to fix config; aborting. Possibly caused by replacement config files being downloaded after each attempted fix.")
                                raise e
                            continue  # Retry loading the pipeline
                    # If we reach here, we couldn't handle the exception
                    self.logger.error("Failed to load SAM ARE pipeline from %s: %s", self.model_dir, str(e))
                    raise e

            self.logger.info("ARE using SAM AutoPipeline.")

        if self._mode == "hf":
            self._hf = _HFSequenceClassifier(
                self.model_dir, device=self.device, dtype=self.dtype, use_fp16=self.kwargs.get("use_fp16", True)
            )
            # Identify negative class (to skip emitting a relation)
            self._negative_labels = {
                "no-relation", "no_relation", "no relation", "none", "o", "outside"
            }

    def _are_hf(self, document) -> None:
        text: str = document.text
        spans = list(getattr(document.labeled_spans, "predictions", []))
        if not spans:
            document.binary_relations = _ShimRelations()
            return

        # Build ordered pairs (i->j, i != j)
        pairs: List[Tuple[Any, Any]] = [(h, t) for i, h in enumerate(spans) for j, t in enumerate(spans) if i != j]
        heads = [text[h.start:h.end] for (h, _t) in pairs]
        tails = [text[t.start:t.end] for (_h, t) in pairs]

        bs = int(self.kwargs.get("are_batch_size", 16))
        results = self._hf.classify_pairs(heads, tails, batch_size=bs)  # type: ignore

        rels = []
        for (head, tail), res in zip(pairs, results):
            raw_label = str(res.get("label", ""))
            norm = raw_label.replace("_", "-").lower().strip()
            if norm in self._negative_labels:
                continue
            rels.append(SimpleNamespace(label=raw_label, score=float(res.get("score", 0.0)), head=head, tail=tail))
        document.binary_relations = _ShimRelations(rels)

    def generate(self, text: str) -> dict:
        """Return a dict with:
            - spans: list[{start,end,label,score}]             (from ADUR)
            - relations: list[{head,tail,label,score}]         (indices into spans)
            - document: object with SAM-like .labeled_spans.predictions etc.
        """
        # First ensure spans (ADUR)
        adu_out = self.adur.generate(text) # TODO: Add an option to load existing ADUs
        document = adu_out["document"]

        if self._mode == "sam":
            # SAM relation extractor expects a document with spans
            self._sam_pipeline(document, inplace=True)  # type: ignore
            spans = [
                {"start": s.start, "end": s.end,
                "label": getattr(s, "label", "ARGUMENT"),
                "score": float(getattr(s, "score", 1.0))}
                for s in getattr(document.labeled_spans, "predictions", [])
            ]
            span_to_idx = {id(s): i for i, s in enumerate(getattr(document.labeled_spans, "predictions", []))}
            relations = []
            for r in getattr(document.binary_relations, "predictions", []):
                head_idx = span_to_idx.get(id(getattr(r, "head", None)), -1)
                tail_idx = span_to_idx.get(id(getattr(r, "tail", None)), -1)
                if head_idx >= 0 and tail_idx >= 0:
                    relations.append({
                        "head": head_idx,
                        "tail": tail_idx,
                        "label": getattr(r, "label", "Relation"),
                        "score": float(getattr(r, "score", 1.0)),
                    })
        else:
            # HF ARE
            self._are_hf(document)
            spans_ns = list(getattr(document.labeled_spans, "predictions", []))
            spans = [
                {"start": s.start, "end": s.end, "label": getattr(s, "label", "ARGUMENT"), "score": float(getattr(s, "score", 1.0))}
                for s in spans_ns
            ]
            span_to_idx = {id(s): i for i, s in enumerate(spans_ns)}
            relations = []
            for r in getattr(document.binary_relations, "predictions", []):
                head_idx = span_to_idx.get(id(getattr(r, "head", None)), -1)
                tail_idx = span_to_idx.get(id(getattr(r, "tail", None)), -1)
                if head_idx < 0 or tail_idx < 0:
                    continue
                relations.append({
                    "head": head_idx,
                    "tail": tail_idx,
                    "label": getattr(r, "label", "Relation"),
                    "score": float(getattr(r, "score", 0.0)),
                })

        return {
            "spans": spans,
            "relations": relations,
            "document": document,
        }

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
        dry_run: bool = False,
        **kwargs: Any,
    ) -> None:

        # Flags and decoding params
        self.rag = bool(rag)
        self.cot = bool(cot)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.dry_run = bool(dry_run)

        # RAG params
        self.embedding_dims = rag_embedding_dims
        self.chunk_size = rag_chunk_size
        self.chunk_overlap = rag_chunk_overlap
        self.top_k = rag_top_k or 5

        # CoT params
        self.cot_steps = cot_steps or 6

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
        if not dry_run:
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
        prompt_files: dict | list[dict] | None, # Must be dict or list of dicts. Optional. Used for RAG context.
        cot_strategy: str | None = None,
    ) -> dict:
        """
        Compose prompts, optionally retrieve context (RAG), optionally orchestrate reasoning (CoT), and generate output.
        Returns: {"text": str, "trace": dict}
        """
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
                
                if self.dry_run:
                    return {"text": "[dry_run]", "trace": trace} # Skip actual generation
                
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
                step_prompts=prompt_files.get('step_prompts') if isinstance(prompt_files, dict) else None,
                dry_run=self.dry_run,
                logger=self.logger,
            )

            # If step_prompts are provided, prefer chat-style, per-step invocations so
            # strategies like system_stepwise (fresh system per step) can be exercised.
            step_prompts = (prompt_files or {}).get('step_prompts') if isinstance(prompt_files, dict) else None
            if step_prompts:
                # Use generate_step_chat which accepts (system_text, user_text) -> str
                result = cot.run_chat_sequence(
                    initial_system=system_text,
                    user_prompt=user_text,
                    generator_chat_callable=lambda sys_txt, usr_txt: self.generate_step_chat(sys_txt, usr_txt),
                    retrieve=retrieval_callback,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )
            else:
                generator_cb = lambda prompt: self._call_hf_generator(system_text, prompt)
                result = cot.run(
                    user_prompt=user_text,
                    generator=generator_cb,
                    retrieve=retrieval_callback,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
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

    # TODO: Provide a helper that can run the HF generator in chat-mode per-step
    # for experiments that require changing the system message between steps.
    # For now End2End relies on CoT.run passing a single composed user prompt
    # (or on the CoT.step_prompts mapping). Implement `generate_step_chat`
    # if you need to perform one chat call per step with fresh context.
    def generate_step_chat(self, system_text: str, user_text: str) -> str:
        """Call the chat generator for a single system/user pair.

        This helper exists as a small wrapper to centralize chat calls, and
        can be extended to preserve or reset chat state per step.
        """
        return self._call_hf_generator(system_text, user_text)
