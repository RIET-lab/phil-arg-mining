from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

_log = logging.getLogger(__name__)


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
    """Resolve input PDFs to process with optional skip-existing filter.

    This efficiently filters files upfront by building a set of existing output stems,
    rather than checking existence individually for each PDF.
    """
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

    # Build set of existing output stems for O(1) lookup instead of O(n) file existence checks
    if output_dir.exists():
        existing_stems = {f.stem.replace(f"_cleaned", "") for f in output_dir.glob(f"*.{expected_ext}")}
        # Also check for stems with _cleaned suffix
        for f in output_dir.glob(f"*.{expected_ext}"):
            stem = f.stem
            if stem.endswith("_cleaned"):
                existing_stems.add(stem[:-8])  # Remove _cleaned suffix
            existing_stems.add(stem)
    else:
        existing_stems = set()

    _log.info(f"Found {len(existing_stems)} existing output files")

    # Filter paths based on existing outputs
    to_process = [p for p in paths if p.stem not in existing_stems]

    _log.debug(f"Filtered {len(paths)} PDFs -> {len(to_process)} to process (skipped {len(paths) - len(to_process)})")

    return to_process


__all__ = ["should_skip_file", "list_input_files"]


