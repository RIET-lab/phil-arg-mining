"""Model registry helpers for Phase 1.

This module provides a thin factory to construct the real End2End pipeline.
It intentionally does not provide a mock fallback — the pipeline code should
raise if required dependencies or model files are missing.
"""
from __future__ import annotations

from typing import Any


def create_end2end(real_kwargs: dict | None = None) -> Any:
    """Instantiate and return the real End2End model.

    Args:
      real_kwargs: forwarded to the End2End constructor.

    Raises:
      Any exception raised by the End2End constructor (no fallback performed).
    """
    from moralkg.argmining.models.models import End2End

    return End2End(**(real_kwargs or {}))
