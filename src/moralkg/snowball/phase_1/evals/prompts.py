from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BasePromptBuilder(ABC):
    @abstractmethod
    def build(self, input_text: str, *, contexts: List[str], use_cot: bool) -> str:
        ...


class DefaultPromptBuilder(BasePromptBuilder):
    def build(self, input_text: str, *, contexts: List[str], use_cot: bool) -> str:
        parts = []
        if contexts:
            parts.append("Context:\n" + "\n---\n".join(contexts))
        parts.append("Task: Analyze the argument and output JSON labels for ACC/ARI/ARC/ARIC.")
        if use_cot:
            parts.append("Think step-by-step briefly before final JSON.")
        parts.append("Input:\n" + input_text.strip())
        parts.append(
            "Output JSON keys: "
            '{"ACC":[{"span":"..","label":"Claim|Premise"}],'
            '"ARI":[{"head":i,"tail":j}],'
            '"ARC":[{"head":i,"tail":j,"type":"support|attack"}],'
            '"ARIC":[...]}'
        )
        return "\n\n".join(parts)


