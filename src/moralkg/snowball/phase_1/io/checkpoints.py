"""Checkpointing utilities for Snowball phase 1.

Provides simple, atomic save/load helpers for batch outputs. The implementation
is intentionally small and dependency-free: writes to a temporary file and
moves it into place to provide atomicity on POSIX systems.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import time


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def save_batch(outputs: Iterable[Dict[str, Any]], outdir: Path, name: str) -> Path:
    """Save a batch of outputs as a JSONL file and return the final path.

    Args:
      outputs: iterable of JSON-serializable objects
      outdir: destination directory (created if missing)
      name: base filename (without extension)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / f"{name}.jsonl"
    lines = []
    for o in outputs:
        lines.append(json.dumps(o, ensure_ascii=False))
    _atomic_write(target, "\n".join(lines) + "\n")
    return target


def save_individual(output: Dict[str, Any], outdir: Path, filename: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / filename
    _atomic_write(target, json.dumps(output, ensure_ascii=False))
    return target


def save_checkpoint(outputs: Iterable[Dict[str, Any]], outdir: Path, strategy: str, name: str | None = None) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = name or f"checkpoint_{strategy}_{timestamp}"
    payload = {
        "timestamp": timestamp,
        "checkpoint_name": name,
        "strategy": strategy,
        "num_outputs": 0,
        "outputs": [],
    }
    for o in outputs:
        payload["outputs"].append(o)
    payload["num_outputs"] = len(payload["outputs"])
    target = outdir / f"{name}.json"
    _atomic_write(target, json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def load_existing(outdir: Path, strategy: str | None = None) -> List[Dict[str, Any]]:
    outdir = Path(outdir)
    if not outdir.exists():
        return []
    results: List[Dict[str, Any]] = []
    # load all .json and .jsonl files
    for p in sorted(outdir.glob("*.jsonl")):
        # each line is a JSON object
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            results.append(json.loads(line))
    for p in sorted(outdir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if strategy and data.get("strategy") != strategy:
            continue
        # if it's a checkpoint with outputs, extend
        if isinstance(data.get("outputs"), list):
            results.extend(data.get("outputs"))
    return results
