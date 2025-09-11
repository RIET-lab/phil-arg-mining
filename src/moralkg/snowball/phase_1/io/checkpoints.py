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
import logging


def _atomic_write(path: Path, data: str) -> None:
    logger = logging.getLogger(__name__)
    tmp = path.with_suffix(path.suffix + ".tmp")
    logger.debug("Atomic write: writing temp file %s", tmp)
    tmp.write_text(data, encoding="utf-8")
    logger.debug("Moving temp file %s -> %s", tmp, path)
    tmp.replace(path)


def save_batch(outputs: Iterable[Dict[str, Any]], outdir: Path, name: str) -> Path:
    """Save a batch of outputs as a JSONL file and return the final path.

    Args:
      outputs: iterable of JSON-serializable objects
      outdir: destination directory (created if missing)
      name: base filename (without extension)
    """
    logger = logging.getLogger(__name__)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / f"{name}.jsonl"
    lines = []
    count = 0
    for o in outputs:
        lines.append(json.dumps(o, ensure_ascii=False))
        count += 1
    logger.info("Saving batch of %d outputs to %s", count, target)
    _atomic_write(target, "\n".join(lines) + "\n")
    logger.info("Saved batch file: %s", target)
    return target


def save_individual(output: Dict[str, Any], outdir: Path, filename: str) -> Path:
    logger = logging.getLogger(__name__)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / filename
    logger.debug("Saving individual output to %s", target)
    _atomic_write(target, json.dumps(output, ensure_ascii=False))
    logger.debug("Saved individual output: %s", target)
    return target


def save_checkpoint(outputs: Iterable[Dict[str, Any]], outdir: Path, strategy: str, name: str | None = None) -> Path:
    logger = logging.getLogger(__name__)
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
    count = 0
    for o in outputs:
        payload["outputs"].append(o)
        count += 1
    payload["num_outputs"] = count
    target = outdir / f"{name}.json"
    logger.info("Writing checkpoint '%s' with %d outputs to %s", name, count, target)
    _atomic_write(target, json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("Saved checkpoint: %s", target)
    return target


def load_existing(outdir: Path, strategy: str | None = None) -> List[Dict[str, Any]]:
    outdir = Path(outdir)
    if not outdir.exists():
        return []
    logger = logging.getLogger(__name__)
    results: List[Dict[str, Any]] = []
    # load all .json and .jsonl files
    for p in sorted(outdir.glob("*.jsonl")):
        # each line is a JSON object
        logger.debug("Loading JSONL file: %s", p)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except Exception as e:
                logger.warning("Skipping malformed line in %s: %s", p, e)
    for p in sorted(outdir.glob("*.json")):
        logger.debug("Loading JSON file: %s", p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not parse JSON file %s: %s", p, e)
            continue
        if strategy and data.get("strategy") != strategy:
            logger.debug("Skipping JSON %s (strategy mismatch: %s != %s)", p, data.get("strategy"), strategy)
            continue
        # if it's a checkpoint with outputs, extend
        if isinstance(data.get("outputs"), list):
            results.extend(data.get("outputs"))
        else:
            # Might be a single-output JSON file
            if isinstance(data, dict) and data:
                results.append(data)
    logger.info("Loaded %d existing outputs from %s (strategy=%s)", len(results), outdir, strategy)
    return results
