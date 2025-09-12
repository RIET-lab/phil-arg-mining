"""Batch utilities for Snowball Phase 1.

Public API (lightweight):
- BatchArgMapper: a small helper to map PromptConfig items to generator calls and
  checkpoint intermediate outputs.
- run_file_mode: convenience runner for file-mode pipelines.
 - run_from_texts: convenience runner for text-oriented pipelines.

This module re-exports the minimal, import-time cheap symbols from
`generation.py` so callers can `from moralkg.snowball.phase_1.batch import BatchArgMapper`.
"""
from __future__ import annotations

# Re-export the public symbols. Keep imports local to avoid heavy work at
# import-time when the package is imported for unrelated reasons (tests, docs).
__all__ = ["BatchArgMapper", "run_from_texts"]

from .generation import BatchArgMapper, run_from_texts
