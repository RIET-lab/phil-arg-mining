from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import rootutils

from .config import Config

__all__ = ["get_logger"]


_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


def _coerce_level(level: Union[str, int, None]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return _LEVELS.get(level.strip().upper(), logging.INFO)
    return logging.INFO


def setup(
    level: Union[str, int, None] = None,
    *,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    destination: Optional[str] = None,
    force: bool = True,
) -> None:
    """Configure root logging.

    If `destination` is a directory path (absolute or relative to repo root),
    logs are written to `<destination>/moralkg.log`. If `destination` is empty
    or equal to "none" (case-insensitive), logs are emitted to stdout.
    """
    resolved_level = _coerce_level(level)
    fmt = fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handlers: list[logging.Handler] = []
    dest = (destination or "").strip()
    if dest and dest.lower() not in {"none", "null", "false"}:
        logs_dir = Path(dest)
        if not logs_dir.is_absolute():
            logs_dir = _ROOT / logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "moralkg.log"
        handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=resolved_level,
        format=fmt,
        datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=force,
    )


_CONFIGURED: bool = False


def _ensure_configured(
    level: Union[str, int, None] = None,
    *,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """Ensure logging is configured based on config.yaml if available."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    # Avoid overriding if another part of the program already configured logging
    if logging.getLogger().hasHandlers():
        _CONFIGURED = True
        return
    cfg = None
    try:
        cfg = Config.load()
    except Exception:
        cfg = None
    cfg_level = level
    cfg_destination: Optional[str] = None
    if cfg is not None:
        cfg_level = cfg.get("general.log_level", level)
        cfg_destination = cfg.get("general.logs", None)
    setup(
        level=cfg_level,
        fmt=fmt,
        datefmt=datefmt,
        destination=cfg_destination,
        force=False,
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a standard, configured logger."""
    _ensure_configured()
    return logging.getLogger(name or "moralkg")
