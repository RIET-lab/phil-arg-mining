from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from moralkg.argmining.parsers.parser import ModelOutputParser
from moralkg.argmining.schemas import ArgumentMap
from . import metrics as legacy_metrics
from moralkg.config import Config


class BaseMetrics(ABC):
    @abstractmethod
    def compute(
        self,
        preds: List[Tuple[str, Dict[str, Any]]],
        golds: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        ...


class Phase1Metrics(BaseMetrics):
    """
    Wraps existing Phase-1 metrics to fit the modular Evaluator interface.

    Returns keys used by the HPO cost function:
      - F1_ACC: ADU span F1 (identification/classification proxy)
      - F1_ARI: duplicates F1_ACC unless a separate ARI metric is provided
      - F1_ARC: relation macro-F1 (support/attack)
      - F1_ARIC: placeholder (0.0) until a combined edge-type metric is defined
      - MSE_edges: squared error on number of ADUs
      - MSE_relations: squared error on number of relations
    """

    def __init__(self, fuzzy_threshold: float | None = None):
        cfg = Config.load()
        # Read fuzzy threshold from config if not provided
        self.fuzzy_threshold = (
            float(fuzzy_threshold)
            if fuzzy_threshold is not None
            else float(cfg.get("snowball.phase_1.eval.fuzzy_thr", 0.7))
        )
        # Count metric is fixed to RMSE per design; remove config toggle
        self.count_metric = "rmse"
        self.parser = ModelOutputParser()

    def _normalize_struct(self, struct: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept either our native model_response schema (ADUs/relations) or
        ACC/ARI/ARC-style outputs and normalize to the former.
        """
        if isinstance(struct, dict) and ("ADUs" in struct or "relations" in struct):
            return struct

        if not isinstance(struct, dict):
            return {"ADUs": {}, "relations": []}

        adus: Dict[str, Any] = {}
        relations: List[Dict[str, Any]] = []

        # Build ADUs from ACC if present
        acc = struct.get("ACC") or []
        for i, item in enumerate(acc):
            span_text = item.get("span") or item.get("text") or ""
            label = (item.get("label") or "unknown").lower()
            adu_type = {
                "argument": "argument",
                "claim": "claim",
                "premise": "premise",
            }.get(label, "unknown")
            adus[f"a{i+1}"] = {"text": span_text, "type": adu_type, "quote": span_text}

        # Edges from ARC (typed) else ARI (untyped)
        arc = struct.get("ARC") or []
        for e in arc:
            head = e.get("head")
            tail = e.get("tail")
            etype = (e.get("type") or "unknown").lower()
            if head is None or tail is None:
                continue
            # indices assume 0-based; if your generator uses 1-based, adjust here
            src_id = f"a{int(head)+1}"
            tgt_id = f"a{int(tail)+1}"
            relations.append({"src": src_id, "tgt": tgt_id, "type": etype})

        if not relations:
            ari = struct.get("ARI") or []
            for e in ari:
                head = e.get("head")
                tail = e.get("tail")
                if head is None or tail is None:
                    continue
                src_id = f"a{int(head)+1}"
                tgt_id = f"a{int(tail)+1}"
                relations.append({"src": src_id, "tgt": tgt_id, "type": "unknown"})

        return {"ADUs": adus, "relations": relations}

    def _to_argument_map(self, struct: Dict[str, Any], map_id: str, source_text: str | None = None) -> ArgumentMap:
        normalized = self._normalize_struct(struct)
        return self.parser.parse_dict(normalized, map_id=map_id, source_text=source_text)

    def compute(
        self,
        preds: List[Tuple[str, Dict[str, Any]]],
        golds: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        f1_acc_vals: List[float] = []
        f1_arc_vals: List[float] = []
        mse_edges_vals: List[float] = []
        mse_rels_vals: List[float] = []
        scaled_rmse_vals: List[float] = []

        for ex_id, pred_json in preds:
            gold_json = golds.get(ex_id, {})
            try:
                pred_map = self._to_argument_map(pred_json, ex_id)
            except Exception:
                # On parse failure, use empty prediction
                pred_map = ArgumentMap(id=ex_id, adus=[], relations=[], source_text=None, source_metadata=None, metadata={})
            try:
                gold_map = self._to_argument_map(gold_json, ex_id)
            except Exception:
                # If gold cannot be parsed, skip this example
                continue

            adu_metrics = legacy_metrics.fuzzy_match_f1(gold_map, pred_map, threshold=self.fuzzy_threshold)
            rel_metrics = legacy_metrics.relation_f1_score(gold_map, pred_map, threshold=self.fuzzy_threshold)
            cnt_metrics = legacy_metrics.count_rmse(gold_map, pred_map)
            scaled_err_key = "scaled_rmse"

            f1_acc_vals.append(float(adu_metrics.get("f1", 0.0)))
            f1_arc_vals.append(float(rel_metrics.get("macro_f1", 0.0)))
            scaled_rmse_vals.append(float(cnt_metrics.get(scaled_err_key, 0.0)))

            # Squared error on counts (MSE-style; per-example)
            adu_diff = len(gold_map.adus) - len(pred_map.adus)
            rel_diff = len(gold_map.relations) - len(pred_map.relations)
            mse_edges_vals.append(float(adu_diff * adu_diff))
            mse_rels_vals.append(float(rel_diff * rel_diff))

        # Aggregate
        def _avg(vals: List[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        metrics: Dict[str, float] = {
            "F1_ACC": _avg(f1_acc_vals),
            # Until a dedicated ARI scorer is provided, reuse ADU span F1 as proxy
            "F1_ARI": _avg(f1_acc_vals),
            "F1_ARC": _avg(f1_arc_vals),
            # Placeholder; wire a real metric if available
            "F1_ARIC": 0.0,
            "MSE_edges": _avg(mse_edges_vals),
            "MSE_relations": _avg(mse_rels_vals),
            "SCALED_RMSE": _avg(scaled_rmse_vals),
        }
        return metrics
