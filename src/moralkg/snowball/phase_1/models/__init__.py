"""Model helpers for Snowball Phase 1.

This package provides factories and adapters for ADUR/ARE pipelines used in
phase 1. Importing this package is lightweight; heavy model instantiation is
performed by model instantiation functions in `registry`.
"""
from __future__ import annotations

__all__ = ["registry", "adapters", "create_end2end", "get_adur_instance", "get_are_instance", "validate_instance"]

from . import adapters
from . import registry

# Re-export some convenience factories from registry for simpler imports
try:
    from .registry import create_end2end, get_adur_instance, get_are_instance  # type: ignore
except Exception:
    # If registry import fails due to environment, keep package importable and surface errors
    def _fail_stub(*args, **kwargs):
        raise RuntimeError("models.registry model instantiation functions are unavailable in this environment")

    create_end2end = _fail_stub
    get_adur_instance = _fail_stub
    get_are_instance = _fail_stub

from .registry import validate_instance 
