#!/usr/bin/env python3
"""Run Phase 1 pipelines (pipeline2 and pipeline3) on full annotated data.

This script programmatically calls the CLI entrypoint in
`src.moralkg.snowball.phase_1.cli` so it runs in-process and shares the
same Config/PATH semantics as the library.

It will try two configurations for each pipeline:
 - ADUR=model_1 + ARE=model_1
 - ADUR=model_2 + ARE=model_2

Usage:
  python scrpts/run_phase1_pipelines.py [--dry-run]

The script expects to be run from the repository root. It will add `src`
into sys.path automatically when necessary.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import importlib
import logging

logger = logging.getLogger("run_phase1_pipelines")


def _ensure_src_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _call_cli_main(argv: list[str]):
    # Import the CLI module and call main(argv)
    mod = importlib.import_module("moralkg.snowball.phase_1.cli")
    if not hasattr(mod, "main"):
        raise RuntimeError("CLI module has no 'main' function")
    return mod.main(argv)


def run_pipeline(pipeline: str, adur_model: str, are_model: str, dry_run: bool = False):
    # pipeline: 'pipeline2' or 'pipeline3'
    logger.info("Running %s with ADUR=%s ARE=%s", pipeline, adur_model, are_model)
    args = [f"run_{pipeline}", "--adur-model", f"{adur_model}", "--are-model", f"{are_model}"]
    if dry_run:
        args.append("--dry-run")
    return _call_cli_main(args)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only validate model refs and don't run heavy work")
    args = parser.parse_args(argv)

    _ensure_src_on_path()

    # Two configs: model_1/model_1 and model_2/model_2
    combos = [("model_1", "model_1"), ("model_2", "model_2")]

    for pipeline in ("pipeline2", "pipeline3"):
        print(f"=== Starting {pipeline} ===")
        for adur_m, are_m in combos:
            try:
                run_pipeline(pipeline, adur_m, are_m, dry_run=args.dry_run)
            except SystemExit as se:
                # CLI may raise SystemExit on errors; propagate non-zero for failures
                if se.code and se.code != 0:
                    logger.error("Pipeline %s with %s/%s exited with %s", pipeline, adur_m, are_m, se.code)
                else:
                    logger.info("Pipeline %s with %s/%s exited cleanly", pipeline, adur_m, are_m)
            except Exception as e:
                logger.exception("Error running %s with %s/%s: %s", pipeline, adur_m, are_m, e)
    print("All runs attempted.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
