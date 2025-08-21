from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional


_lock = Lock()


@dataclass
class FailureRecord:
    filename: str
    error: str
    error_type: str
    timestamp: str


def load_failures(path: Optional[Path]) -> Dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_failures(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def record_failure(path: Optional[Path], filename: str, error: str, error_type: str) -> None:
    if path is None:
        return
    with _lock:
        data = load_failures(path)
        now = datetime.now().isoformat()
        if filename in data:
            entry = data[filename]
            entry["attempts"] = entry.get("attempts", 1) + 1
            entry["last_failure"] = now
            entry["last_error"] = error
            entry["error_type"] = error_type
        else:
            data[filename] = {
                "first_failure": now,
                "last_failure": now,
                "last_error": error,
                "error_type": error_type,
                "attempts": 1,
            }
        save_failures(path, data)


__all__ = ["load_failures", "save_failures", "record_failure"]


