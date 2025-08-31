from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import tempfile
import rootutils
from moralkg.config import Config
from moralkg.logging import get_logger



@dataclass
class GrobidSettings:
    """Settings for configuring the Grobid client.

    Values default from `config.yaml` under `philpapers.papers.grobid`.
    """

    server: str
    batch_size: int = 1000
    sleep_time: int = 3
    timeout: int = 60
    coordinates: Optional[List[str]] = None

    @staticmethod
    def from_config(cfg: Optional[Config] = None) -> "GrobidSettings":
        cfg = cfg or Config.load()
        base = cfg.get("philpapers.papers.grobid", {}) or {}
        return GrobidSettings(
            server=str(base.get("server", "http://localhost:8070")),
            batch_size=int(base.get("batch_size", 1000)),
            sleep_time=int(base.get("sleep_time", 3)),
            timeout=int(base.get("timeout", 60)),
            coordinates=list(base.get("coordinates", [])) or None,
        )


def _resolve_repo_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    # Resolve relative to repo root
    repo_root = Path(rootutils.setup_root(__file__, indicator=".git"))
    return repo_root / p


def _write_temp_grobid_config(settings: GrobidSettings) -> Path:
    """Write a minimal Grobid client JSON config into a temp file.

    The upstream Grobid Python client expects a JSON config path.
    """
    # Use NamedTemporaryFile to ensure a stable path during the run
    tmp_dir = Path(tempfile.gettempdir())
    cfg_path = tmp_dir / "moralkg_grobid_config.json"
    payload = {
        "grobid_server": settings.server,
        "batch_size": settings.batch_size,
        "sleep_time": settings.sleep_time,
        "timeout": settings.timeout,
        # Coordinates are used by some services; pass-through if provided
        "coordinates": settings.coordinates or [],
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    return cfg_path


def process_references(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    settings: Optional[GrobidSettings] = None,
    n_workers: Optional[int] = None,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """Run Grobid `processReferences` on all PDFs in `input_dir`.

    - Reads connection and batching from `GrobidSettings` (defaults from config).
    - Writes TEI XML into `output_dir`.
    """
    try:
        from grobid_client.grobid_client import GrobidClient  # type: ignore
    except Exception as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            "grobid_client is not installed. Please `pip install grobid_client_python`."
        ) from exc

    logger = get_logger("grobid")

    settings = settings or GrobidSettings.from_config()
    cfg_path = _write_temp_grobid_config(settings)

    src = _resolve_repo_path(input_dir)
    dst = _resolve_repo_path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {src}")

    effective_workers = int(n_workers if n_workers is not None else 64)
    logger.info(
        "Starting Grobid processReferences | server=%s | src=%s | dst=%s | workers=%s",
        settings.server,
        str(src),
        str(dst),
        effective_workers,
    )

    client = GrobidClient(config_path=str(cfg_path))
    client.process(
        "processReferences",
        str(src),
        output=str(dst),
        n=effective_workers,
        force=bool(force),
        verbose=bool(verbose),
    )


__all__ = [
    "GrobidSettings",
    "process_references",
]


