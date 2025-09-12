"""Prompt utilities for Snowball Phase 1.

Exports:
- PromptConfig, load_prompts, render_prompt

Keep prompt template data on disk; loading large template sets is performed
explicitly by callers via `load_prompts`.
"""
from __future__ import annotations

__all__ = ["PromptConfig", "load_prompts", "render_prompt"]

from .loader import PromptConfig, load_prompts, render_prompt
