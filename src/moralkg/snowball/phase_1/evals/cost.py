from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
from moralkg.config import Config


class BaseCost(ABC):
    @abstractmethod
    def compute(self, metrics: Dict[str, float]) -> float:
        ...


class WeightedCost(BaseCost):
    """
    Phase 1 loss per snowball.md:
      - Equal weights over: span F1, relation macro-F1, count error, GED
      - We treat (1 - scaled_rmse) and GED similarity as rewards; loss = 1 - combined_reward
    """

    def __init__(self, w_span: float | None = None, w_rel: float | None = None, w_cnt: float | None = None, w_ged: float | None = None):
        cfg = Config.load()
        cfg_w = cfg.get("snowball.phase_1.eval.loss.w", [0.25, 0.25, 0.25, 0.25]) or [0.25, 0.25, 0.25, 0.25]
        self.w_span = float(w_span) if w_span is not None else float(cfg_w[0])
        self.w_rel = float(w_rel) if w_rel is not None else float(cfg_w[1])
        self.w_cnt = float(w_cnt) if w_cnt is not None else float(cfg_w[2])
        self.w_ged = float(w_ged) if w_ged is not None else float(cfg_w[3])

    def compute(self, m: Dict[str, float]) -> float:
        span_f1 = float(m.get("f1", m.get("F1_ACC", 0.0)))
        rel_f1 = float(m.get("macro_f1", m.get("F1_ARC", 0.0)))
        inv_cnt_err = float(1.0 - m.get("scaled_rmse", m.get("SCALED_RMSE", 1.0)))
        ged_sim = float(m.get("ged_sim", 0.0))

        combined_reward = (
            self.w_span * span_f1
            + self.w_rel * rel_f1
            + self.w_cnt * inv_cnt_err
            + self.w_ged * ged_sim
        )
        # Convert to loss for minimization
        return float(1.0 - combined_reward)
