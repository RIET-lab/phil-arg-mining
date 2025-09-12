"""CLI helpers for running ADUR and ARE pipelines in Phase 1.

Provides simple programmatic entrypoints that can be used by thin scripts
or called from the orchestrator. We intentionally keep behavior explicit and
fail loudly on model validation/load errors.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .models import registry
from .io import checkpoints
from .batch.generation import run_file_mode

logger = logging.getLogger(__name__)


def _write_output_json(output: dict, outdir: Path, prefix: str = "adur") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_output.json"
    return checkpoints.save_individual(output, outdir, fname)


def run_adur_cmd(model_ref: Any, input_file: Path, outdir: Path, dry_run: bool = False):
    """Run ADUR on a single file and save the normalized output.

    model_ref may be a dict (e.g., {"dir": ...}) or a string path. If dry_run is True,
    only validate the model directory and return non-zero exit if invalid.
    """
    if dry_run:
        ok, details = registry.validate_pipeline(model_ref)
        if not ok:
            logger.error("Validation failed: %s", details)
            raise SystemExit(2)
        logger.info("Validation succeeded: %s", details)
        return

    # Create ADUR pipeline (this will raise loudly on failure)
    adur = registry.get_adur_pipeline(model_ref)

    # Use file-mode runner which normalizes and saves outputs via checkpoints
    from .models.adapters import normalize_adur_output
    outdir_path = run_file_mode(adur, [input_file], outdir, normalize_adur_output, prefix="adur")
    logger.info("ADUR file-mode run completed; outputs in %s", outdir_path)
    return outdir_path


def run_are_cmd(model_ref: Any, adur_model_ref: Any, input_file: Path, outdir: Path, dry_run: bool = False):
    if dry_run:
        ok1, d1 = registry.validate_pipeline(model_ref)
        ok2, d2 = registry.validate_pipeline(adur_model_ref)
        if not ok1 or not ok2:
            logger.error("Validation failed: ARE=%s ADUR=%s", d1 if not ok1 else "ok", d2 if not ok2 else "ok")
            raise SystemExit(2)
        logger.info("Validation succeeded for both ARE and ADUR")
        return

    are = registry.get_are_pipeline(model_ref, adur_model_ref)
    from .models.adapters import normalize_are_output
    outdir_path = run_file_mode(are, [input_file], outdir, normalize_are_output, prefix="are")
    logger.info("ARE file-mode run completed; outputs in %s", outdir_path)
    return outdir_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase1_cli")
    sub = parser.add_subparsers(dest="cmd")

    p_adur = sub.add_parser("run_adur")
    p_adur.add_argument("--model", required=True)
    p_adur.add_argument("--input", required=True)
    p_adur.add_argument("--outdir", required=True)
    p_adur.add_argument("--dry-run", action="store_true")

    p_are = sub.add_parser("run_are")
    p_are.add_argument("--model", required=True)
    p_are.add_argument("--adur-model", required=True)
    p_are.add_argument("--input", required=True)
    p_are.add_argument("--outdir", required=True)
    p_are.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run_adur":
        run_adur_cmd(args.model, Path(args.input), Path(args.outdir), dry_run=args.dry_run)
    elif args.cmd == "run_are":
        run_are_cmd(args.model, args.adur_model, Path(args.input), Path(args.outdir), dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
