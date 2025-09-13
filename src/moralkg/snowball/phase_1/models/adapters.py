"""Adapters to normalize raw ADUR/ARE outputs into canonical schema objects.

These helpers convert model-specific outputs (as produced by the ADUR/ARE
classes in `moralkg.argmining.models.models`) into pydantic-backed schema
instances (ADU, Relation, ArgumentMap) and lightweight dicts suitable for
serialization by the phase_1 checkpointing utilities.

TODO: Make sure these are compatible with the existing parser module in `src.moralkg.argmining.parsers`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from pathlib import Path

from moralkg.argmining.schemas import ADU as ADUModel, Relation as RelationModel, ArgumentMap
from moralkg.argmining.schemas.adu import SpanPosition


logger = logging.getLogger(__name__)


def _coerce_adu_type(label: Optional[str]) -> str:
    if not label:
        return "Claim"
    v = str(label).strip().lower()
    if v in {"major claim", "major_claim", "major", "thesis"}:
        return "Major Claim"
    if v in {"claim", "premise", "premise_claim", "supporting_claim", "ARGUMENT"}:
        return "Claim"
    # Default to Claim for most label variants to be compatible with evaluator
    return "Claim"


def normalize_adur_output(raw: Dict[str, Any], source_text: Optional[str] = None) -> Dict[str, Any]:
    """Normalize raw ADUR output to canonical dict with ADUs and statistics."""
    # Base ADUs
    adus_raw: List[Dict[str, Any]] = raw.get("adus", []) if isinstance(raw, dict) else []

    # Derive ADUs from spans (if any)
    spans = (raw.get("spans") or []) if isinstance(raw, dict) else []
    for span in spans:
        start, end = span.get("start"), span.get("end")
        text = span.get("text")
        if text is None and source_text is not None and start is not None and end is not None:
            try:
                text = source_text[int(start):int(end)]
            except Exception:
                text = None
        adus_raw.append({
            "text": text,
            "label": span.get("label"),
            "start": span.get("start"),
            "end": span.get("end"),
            "score": span.get("score"),
        })

    #if not isinstance(adus_raw, list) or not adus_raw:
    #    logger.error("ADUR output missing expected 'adus' list")
    #    raise ValueError("Invalid ADUR output: missing 'adus' list")

    def _safe_positions(item: Dict[str, Any]) -> List["SpanPosition"]:
        """Extract and validate positions from an ADU item dict."""
        start, end = item.get("start"), item.get("end")
        if start is None or end is None:
            return []
        try:
            return [SpanPosition(start=int(start), end=int(end))]
        except Exception:
            logger.warning("Invalid start/end for ADU %s: %s,%s", item.get("id") or item.get("adu_id"), start, end)
            return []

    adus: List[Dict[str, Any]] = []
    for i, item in enumerate(adus_raw, start=1):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict ADU entry at index %d", i - 1)
            continue

        adu_id = item.get("id") or item.get("adu_id") or f"adu-{i}"
        label = _coerce_adu_type(item.get("label") or item.get("type"))
        try:
            adu_obj = ADUModel(
                id=str(adu_id),
                type=label,
                text=str(item.get("text") or item.get("label_text") or ""),
                quote=(str(item["quote"]) if item.get("quote") is not None else None),
                isImplicit=bool(item.get("isImplicit") or False),
                positions=_safe_positions(item),
            )
        except Exception as exc:
            logger.error("Failed to construct ADUModel for %s: %s", adu_id, exc)
            raise

        # Prefer Pydantic v2 .model_dump(); fall back to v1 .dict()
        dump = getattr(adu_obj, "model_dump", None)
        adus.append(dump() if callable(dump) else adu_obj.dict())

    # Statistics
    types = defaultdict(int)
    for a in adus:
        types[str(a.get("type") or "Claim")] += 1

    return {"adus": adus, "statistics": {"total_adus": len(adus), "adu_types": dict(types)}}


def _coerce_relation_type(label: Optional[str]) -> str:
    if not label:
        return "unknown"
    v = str(label).strip().lower()
    if v in {"support", "supports", "pro"}:
        return "support"
    if v in {"attack", "attacks", "con"}:
        return "attack"
    return "unknown"


def normalize_are_output(raw: Dict[str, Any], source_text: Optional[str] = None) -> Dict[str, Any]:
    """Normalize raw ARE output to canonical dict with ADUs, relations, and statistics.

    Args:
      raw: expected to contain keys 'adus' and 'relations'
      source_text: optional original document text (used to validate positions)

    Returns:
      A dict with keys: 'adus', 'relations', 'statistics'
    """
    adur_res = normalize_adur_output(raw, source_text=source_text)
    adus = adur_res.get("adus", [])

    relations_raw = raw.get("relations") if isinstance(raw, dict) else None
    if relations_raw is None:
        # Some ARE outputs attach relations under 'relations' while others use 'rels'
        relations_raw = raw.get("rels") or []

    relations: List[Dict[str, Any]] = []
    for i, r in enumerate(relations_raw, start=1):
        if not isinstance(r, dict):
            logger.warning("Skipping non-dict relation at index %d", i - 1)
            continue

        rel_type = _coerce_relation_type(r.get("label") or r.get("type") or r.get("relation"))
        # Prefer explicit ADU ids if provided, else try to map from texts
        src = r.get("head") or r.get("src") or r.get("source") or r.get("head_text")
        tgt = r.get("tail") or r.get("tgt") or r.get("target") or r.get("tail_text")

        # If src/tgt are spans (start/end), convert them to textual references if possible
        # For simplicity: if provided as numeric spans, leave them as-is inside metadata
        src_id = None
        tgt_id = None

        # Try to match by exact text to an ADU id (best-effort)
        if isinstance(src, str):
            for a in adus:
                if a.get("text") == src or a.get("text", "").strip() == src.strip():
                    src_id = a.get("id")
                    break
        if isinstance(tgt, str):
            for a in adus:
                if a.get("text") == tgt or a.get("text", "").strip() == tgt.strip():
                    tgt_id = a.get("id")
                    break

        # Fall back to using provided src/tgt values
        final_src = src_id or (src if src is not None else f"adu-{i}-src")
        final_tgt = tgt_id or (tgt if tgt is not None else f"adu-{i}-tgt")

        rel_id = r.get("id") or f"rel-{i}"
        try:
            rel_obj = RelationModel(id=str(rel_id), src=str(final_src), tgt=str(final_tgt), type=rel_type)
        except Exception as exc:
            logger.error("Failed to construct RelationModel for %s: %s", rel_id, exc)
            raise

        # Prefer model_dump() for Pydantic v2; fallback to dict() if unavailable
        try:
            relations.append(rel_obj.model_dump())
        except Exception:
            relations.append(rel_obj.dict())

    # Build ArgumentMap for statistics convenience
    try:
        am = ArgumentMap(id=str(raw.get("id") or "argument_map"), adus=[ADUModel(**a) for a in adus], relations=[RelationModel(**r) for r in relations])
        stats = am.map_statistics()
    except Exception:
        # Fallback simple stats
        stats = {"total_adus": len(adus), "total_relations": len(relations)}

    return {"adus": adus, "relations": relations, "statistics": stats}
