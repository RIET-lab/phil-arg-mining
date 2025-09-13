#!/usr/bin/env python3
"""Run Phase 1 pipelines (pipeline2 and pipeline3) on full annotated data.

This script programmatically calls the CLI entrypoint in
`src.moralkg.snowball.phase_1.cli` so it runs in-process and shares the
same Config/PATH semantics as the library.

It will try two configurations for each pipeline:
 - ADUR=model_1 + ARE=model_1
 - ADUR=model_2 + ARE=model_2

Usage:
  PYTHONPATH=src python scripts/snowball_phase1_ADUR_ARE_generate.py [--dry-run]
    --dry-run: only validate model refs without running heavy work
"""
#from __future__ import annotations

import sys
import argparse
from pathlib import Path as _Path

# If the user usually runs this script with: PYTHONPATH=src python scripts/...
# add the project's `src` directory to sys.path automatically so the script
# can be run from the debugger or directly without setting PYTHONPATH.
# This prepends the src path to sys.path only if it's not already present.
_THIS_FILE = _Path(__file__).resolve()
# Project root is one level up from scripts/ (adjust if repository layout changes)
_PROJECT_ROOT = _THIS_FILE.parent.parent
_SRC_DIR = (_PROJECT_ROOT / "src").resolve()
if str(_SRC_DIR) not in map(str, sys.path):
    sys.path.insert(0, str(_SRC_DIR))
from pathlib import Path
import importlib
import rootutils

rootutils.setup_root(__file__, indicator=".git")
from moralkg import get_logger
logger = get_logger("run_phase1_pipelines_2_and_3")
#import logging
#logger = logging.getLogger(__name__)

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

    # Two configs: model_1/model_1 and model_2/model_2
    #combos = [("model_1", "model_1"), ("model_2", "model_2")]
    #combos = [("model_1", "model_1")] # Just test model 1 for now
    combos = [("model_2", "model_2")] # Module 2 is broken. TODO: Ensure that the unexpected terms in sam_are_sciarg's config.json and taskmodule_config.json are not passed as kwargs to AutoPipeline.from_pretrained in models.py 

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
    main()
