"""
Old version of ADU (Argumentative Discourse Unit) schema definition.
Lacks some of the completeness/basic functionality of the new version, but we want to re-introduce the span-finding capability from this version later.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process


class ADUType(str, Enum):
    """Type classification for ADUs."""

    MAJOR_CLAIM = "Major Claim"  # NOTE: considered the thesis/argument of the paper
    CLAIM = "Claim"
    UNKNOWN = "Unknown"


class SpanPosition(BaseModel):
    """Position of text span in the source document."""

    start: int = Field(..., description="Start character index")
    end: int = Field(..., description="End character index")


class ADU(BaseModel):
    """
    Argumentative Discourse Unit.
    """

    id: str = Field(..., description="Unique identifier for the ADU")
    type: ADUType = Field(
        default=ADUType.UNKNOWN, description="Type of ADU (major claim, claim, or premise)"
    )
    text: str = Field(
        ..., description="ADU description (exact text, paraphrase, etc.)"
    )
    quote: str = Field(
        ...,
        description="Exact span in the text the ADU refers to",
    )
    positions: Optional[List[SpanPosition]] = Field(
        default_factory=list,
        description="Character positions of the ADU in the source document",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Space for any additional metadata"
    )

    def init(self, source_document: str, thr: float = 0.85) -> str:
        """
        Ensure ADU attributes (text, content, positions) are consistent and mutually informative.
        - If positions exist, check they're valid then extract from source_document and update text/content if needed.
        - If text exists but no positions, try to find positions via direct match then fuzzy-match.
        - Updates self.text, self.content, and self.positions as needed.
        Returns the resolved ADU text.
        """

        def _find_exact_spans(text, doc):
            """Find all (start, end) indices of exact text in doc."""
            return [m.span() for m in re.finditer(re.escape(text), doc)]

        def _fuzzy_best_span(text, doc, thr=0.85):
            """
            Find the span in doc most similar to text using rapidfuzz.
            Returns (start, end), score if above threshold, else (None, 0.0).
            """
            window = len(text)
            best = None
            best_score = 0.0
            for i in range(0, len(doc) - window + 1):
                candidate = doc[i : i + window]
                score = fuzz.ratio(text, candidate) / 100.0
                if score > best_score:
                    best_score = score
                    best = (i, i + window)
            if best_score >= thr:
                return best, best_score
            return None, best_score

        # TODO: don't assume positions are valid - compare quote from positions to provided quote or text (do exact then fuzzy match)
        # 1. If positions exist, extract text from source and update text/content if needed
        if self.positions and len(self.positions) > 0:
            spans = []
            for pos in self.positions:
                span_text = source_document[pos.start : pos.end]
                spans.append(span_text)
            joined_text = " ... ".join(spans)
            self.text = joined_text
            if (
                not self.content or self.content.strip() == ""
            ):  # only update content if empty
                self.content = joined_text
            return joined_text

        # 2. If text exists but no positions, try to find positions in source_document
        if self.text and self.text.strip() != "":
            # Try exact match
            matches = _find_exact_spans(self.text, source_document)
            if matches:
                self.positions = [SpanPosition(start=s, end=e) for s, e in matches]
                if not self.content or self.content.strip() == "":
                    self.content = self.text
                return self.text

            # Try fuzzy match
            best_span, _ = _fuzzy_best_span(self.text, source_document, threshold=thr)
            if best_span:
                s, e = best_span
                matched_text = source_document[s:e]
                self.positions = [SpanPosition(start=s, end=e)]
                self.text = matched_text
                if not self.content or self.content.strip() == "":
                    self.content = matched_text
                return matched_text

        # 3. Fallback: return whatever text is present
        return self.text if self.text else ""

    @staticmethod
    def fuzzy_similarity(text_a: str, text_b: str) -> float:
        """
        Return a normalized similarity score in [0,1] using rapidfuzz ratio.
        """
        if not text_a or not text_b:
            return 0.0
        try:
            return float(fuzz.ratio(text_a, text_b)) / 100.0
        except Exception:
            return 0.0
