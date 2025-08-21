from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        ...


class NoopRetriever(BaseRetriever):
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        return []


class YourFAISSRetriever(BaseRetriever):
    def __init__(self, index_path: str):
        self.index_path = index_path
        # TODO: load embeddings/index

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        # TODO: embed query, search index, return texts
        return []


