from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any


class BaseDatasetAdapter(ABC):
    """Standard interface the Evaluator expects."""

    @abstractmethod
    def iter(self, split: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Yield (example_id, input_text, gold_struct)."""
        ...


class ListAdapter(BaseDatasetAdapter):
    """Minimal placeholder adapter (replace with the PE/CDCP/AbstRCT loaders)."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data  # dict(split -> list of (id, text, gold))

    def iter(self, split: str):
        for ex_id, text, gold in self.data.get(split, []):
            yield ex_id, text, gold


