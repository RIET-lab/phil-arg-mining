"""Thin wrapper script to run ADUR on a single file via the Phase 1 CLI.

Usage (from repo root with venv activated):
PYTHONPATH=src python scripts/run_adur.py --model /path/to/model_dir --input data/sample.txt --outdir /tmp/out --dry-run
"""
from pathlib import Path
import argparse
from moralkg.snowball.phase_1.cli import run_adur_cmd
from moralkg.config import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=False, help="Model ref (local dir path). If omitted, use config paths.models.adur.model_1 or model_2 when --use-model-2 is set")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-model-2", action="store_true", help="Use the model_2 entry from config.paths.models.adur")
    args = p.parse_args()

    # Determine model_ref: prefer explicit --model (local path), else read from config
    if args.model:
        model_ref = {"dir": args.model}
    else:
        cfg = Config.load()
        model_key = "model_2" if args.use_model_2 else "model_1"
        cfg_ref = cfg.get(f"paths.models.adur.{model_key}")
        if not cfg_ref:
            raise FileNotFoundError(f"ADUR model not configured in config.yaml at paths.models.adur.{model_key}")
        model_ref = cfg_ref

    run_adur_cmd(model_ref, Path(args.input), Path(args.outdir), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
