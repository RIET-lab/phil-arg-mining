from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def should_skip_file(pdf_path: Path, output_dir: Path, expected_ext: str = "md") -> bool:
    """Return True if an output file with expected extension already exists."""
    base = pdf_path.stem
    return (output_dir / f"{base}.{expected_ext}").exists()


def list_input_files(
    input_dir: Path,
    specific_files: Optional[Iterable[str]] = None,
    skip_existing: bool = False,
    output_dir: Optional[Path] = None,
    expected_ext: str = "md",
) -> List[Path]:
    """Resolve input PDFs to process with optional skip-existing filter."""
    if specific_files:
        paths: List[Path] = []
        for name in specific_files:
            p = input_dir / name
            if p.exists() and p.suffix.lower() == ".pdf":
                paths.append(p)
        return _apply_skip(paths, skip_existing, output_dir, expected_ext)

    paths = list(input_dir.glob("*.pdf"))
    return _apply_skip(paths, skip_existing, output_dir, expected_ext)


def _apply_skip(
    paths: List[Path], skip_existing: bool, output_dir: Optional[Path], expected_ext: str
) -> List[Path]:
    if not skip_existing or output_dir is None:
        return paths
    return [p for p in paths if not should_skip_file(p, output_dir, expected_ext)]


__all__ = ["should_skip_file", "list_input_files"]


