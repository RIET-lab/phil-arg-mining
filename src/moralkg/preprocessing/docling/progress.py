from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional


_overall_lock = Lock()


@dataclass
class OverallProgress:
    total_files: int
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    files_partial: int = 0
    completed_workers: int = 0
    status: str = "initializing"

    def to_dict(self) -> Dict:
        now = datetime.now().isoformat()
        return {
            "total_files": self.total_files,
            "files_processed": self.files_processed,
            "files_succeeded": self.files_succeeded,
            "files_failed": self.files_failed,
            "files_partial": self.files_partial,
            "completed_workers": self.completed_workers,
            "start_time": now,
            "last_update": now,
            "status": self.status,
        }


def create_overall_progress(progress_dir: Path, total_files: int) -> Path:
    progress_dir.mkdir(parents=True, exist_ok=True)
    overall_file = progress_dir / "overall_progress.json"
    with overall_file.open("w") as f:
        json.dump(OverallProgress(total_files).to_dict(), f, indent=2)
    return overall_file


def update_overall_progress(progress_dir: Path, **updates) -> None:
    with _overall_lock:
        overall_file = progress_dir / "overall_progress.json"
        if not overall_file.exists():
            return
        data = json.loads(overall_file.read_text())
        for key, value in updates.items():
            if key.startswith("add_"):
                field = key[4:]
                data[field] = data.get(field, 0) + value
            else:
                data[key] = value
        data["last_update"] = datetime.now().isoformat()
        overall_file.write_text(json.dumps(data, indent=2))


def create_worker_progress(progress_dir: Path, worker_id: int, worker_total_files: int, overall_total_files: int) -> Path:
    progress_dir.mkdir(parents=True, exist_ok=True)
    p = progress_dir / f"progress_worker_{worker_id}.json"
    data = {
        "worker_id": worker_id,
        "worker_total_files": worker_total_files,
        "overall_total_files": overall_total_files,
        "files_processed": 0,
        "files_succeeded": 0,
        "files_failed": 0,
        "files_partial": 0,
        "current_file": None,
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "status": "starting",
    }
    p.write_text(json.dumps(data, indent=2))
    return p


def update_worker_progress(progress_file: Path, **updates) -> None:
    if not progress_file:
        return
    data = json.loads(progress_file.read_text())
    for k, v in updates.items():
        data[k] = v
    data["last_update"] = datetime.now().isoformat()
    progress_file.write_text(json.dumps(data, indent=2))


def cleanup_worker(progress_dir: Path, worker_id: int) -> None:
    p = progress_dir / f"progress_worker_{worker_id}.json"
    if p.exists():
        p.unlink()


__all__ = [
    "create_overall_progress",
    "update_overall_progress",
    "create_worker_progress",
    "update_worker_progress",
    "cleanup_worker",
]


