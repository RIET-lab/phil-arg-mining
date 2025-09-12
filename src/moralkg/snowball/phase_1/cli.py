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
from .batch.generation import run_from_texts
from moralkg.argmining.loaders.loader import Dataset
from moralkg import Config

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
        ok, details = registry.validate_instance(model_ref)
        if not ok:
            logger.error("Validation failed: %s", details)
            raise SystemExit(2)
        logger.info("Validation succeeded: %s", details)
        return

    # Create ADUR model instance (this will raise loudly on failure)
    adur = registry.get_adur_instance(model_ref)

    # Use text-mode runner which normalizes and saves outputs via checkpoints
    from .models.adapters import normalize_adur_output
    # read input text
    src_text = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    outdir_path = run_from_texts(adur, {Path(input_file).stem: src_text}, outdir, normalize_adur_output, prefix="adur")
    logger.info("ADUR file-mode run completed; outputs in %s", outdir_path)
    return outdir_path


def run_are_cmd(model_ref: Any, adur_model_ref: Any, input_file: Path, outdir: Path, dry_run: bool = False):
    if dry_run:
        ok1, d1 = registry.validate_instance(model_ref)
        ok2, d2 = registry.validate_instance(adur_model_ref)
        if not ok1 or not ok2:
            logger.error("Validation failed: ARE=%s ADUR=%s", d1 if not ok1 else "ok", d2 if not ok2 else "ok")
            raise SystemExit(2)
        logger.info("Validation succeeded for both ARE and ADUR")
        return

    are = registry.get_are_instance(model_ref, adur_model_ref)
    from .models.adapters import normalize_are_output
    src_text = Path(input_file).read_text(encoding="utf-8", errors="ignore")
    outdir_path = run_from_texts(are, {Path(input_file).stem: src_text}, outdir, normalize_are_output, prefix="are")
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

    p_p2 = sub.add_parser("run_pipeline2")
    p_p2.add_argument("--adur-model", required=False, help="Optional ADUR model ref (dir or hf). If omitted, load from config")
    p_p2.add_argument("--are-model", required=False, help="Optional ARE model ref (dir or hf). If omitted, load from config")
    p_p2.add_argument("--adur-outdir", required=False, help="Optional output directory. If omitted, derived from config and model flags")
    p_p2.add_argument("--are-outdir", required=False, help="Optional output directory. If omitted, derived from config and model flags")
    p_p2.add_argument("--dry-run", action="store_true")
    p_p2.add_argument("--use-adur-model-2", action="store_true")
    p_p2.add_argument("--use-are-model-2", action="store_true")
    p_p2.add_argument("--paper-limit", type=int, default=None, help="Limit number of papers to process (for quick smoke runs)")

    p_p3 = sub.add_parser("run_pipeline3")
    p_p3.add_argument("--adur-model", required=False, help="Optional ADUR model ref (dir or hf). If omitted, load from config")
    p_p3.add_argument("--are-model", required=False, help="Optional ARE model ref (dir or hf). If omitted, load from config")
    p_p3.add_argument("--adur-outdir", required=False, help="Optional output directory. If omitted, derived from config and model flags")
    p_p3.add_argument("--are-outdir", required=False, help="Optional output directory. If omitted, derived from config and model flags")
    p_p3.add_argument("--dry-run", action="store_true")
    p_p3.add_argument("--use-adur-model-2", action="store_true")
    p_p3.add_argument("--use-are-model-2", action="store_true")
    p_p3.add_argument("--paper-limit", type=int, default=None, help="Limit number of papers to process (for quick smoke runs)")
    p_p3.add_argument("--major-method", default="centroid", help="Major ADU selection method")

    return parser


def main(argv: list[str] | None = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    dataset = Dataset()

    def _derive_outdirs(pipeline_name: str, use_adur_model_2: bool = False, use_are_model_2: bool = False, explicit_adur: Path | None = None, explicit_are: Path | None = None) -> Path:
        """Derive ADUR and ARE output directories.

        This helper reads configured base output directories from the project
        config. If a config key is missing we raise a clear SystemExit so the
        caller gets a readable error instead of passing None into Path().
        """
        # Use the correct config paths for outputs
        adur_model_key = "model_2" if use_adur_model_2 else "model_1"
        are_model_key = "model_2" if use_are_model_2 else "model_1"

        adur_output_path = None
        are_output_path = None

        if not explicit_adur:
            cfg_key = f"paths.snowball.phase_1.outputs.adur.{adur_model_key}"
            base = cfg.get(cfg_key, None)
            if base is None:
                logger.error("Missing configuration key: %s; pass --adur-outdir or add it to config.yaml", cfg_key)
                raise SystemExit(2)
            adur_output_path = Path(base) / f"{pipeline_name}_ADUR_{adur_model_key}"
        else:
            # allow explicit_adur to be a string path as well
            adur_output_path = Path(explicit_adur)

        if not explicit_are:
            cfg_key = f"paths.snowball.phase_1.outputs.are.{are_model_key}"
            base = cfg.get(cfg_key, None)
            if base is None:
                logger.error("Missing configuration key: %s; pass --are-outdir or add it to config.yaml", cfg_key)
                raise SystemExit(2)
            are_output_path = Path(base) / f"{pipeline_name}_ARE_{are_model_key}_after_ADUR_{adur_model_key}"
        else:
            are_output_path = Path(explicit_are)

        # Use the configured paths
        return adur_output_path, are_output_path
      
    if args.cmd == "run_adur":
        run_adur_cmd(args.model, Path(args.input), Path(args.outdir), dry_run=args.dry_run)
    elif args.cmd == "run_are":
        run_are_cmd(args.model, args.adur_model, Path(args.input), Path(args.outdir), dry_run=args.dry_run)
    elif args.cmd == "run_pipeline2":
        # Build dataset and resolve annotated paper file paths
        ds = Dataset()
        paper_ids = list(ds.annotations.by_paper.keys())
        if args.paper_limit:
            paper_ids = paper_ids[: args.paper_limit]
        input_texts = {}
        for pid in paper_ids:
            paper_text = dataset.get_paper(pid)
            if paper_text is None:
                logger.debug("Skipping missing paper: %s", pid)
                continue
            input_texts[pid] = paper_text

        if not input_texts:
            logger.error("No input files found for annotated papers; aborting")
            raise SystemExit(2)

        from .pipelines.adur_are import run_pipeline2

        adur_outdir_path, are_outdir_path = _derive_outdirs("pipeline2", use_adur_model_2=args.use_adur_model_2, use_are_model_2=args.use_are_model_2, explicit_adur=args.adur_outdir, explicit_are=args.are_outdir)

        if args.adur_outdir:
            adur_outdir_path = Path(args.adur_outdir)
        if args.are_outdir:
            are_outdir_path = Path(args.are_outdir)

        out = run_pipeline2(input_texts, adur_outdir=adur_outdir_path, are_outdir=are_outdir_path, adur_model_ref=args.adur_model, are_model_ref=args.are_model, use_adur_model_2=args.use_adur_model_2, use_are_model_2=args.use_are_model_2, dry_run=args.dry_run)
        logger.info("Pipeline2 completed; outputs at: %s", out)
        return out
    
    elif args.cmd == "run_pipeline3":
        ds = Dataset()
        paper_ids = list(ds.annotations.by_paper.keys())
        if args.paper_limit:
            paper_ids = paper_ids[: args.paper_limit]

        input_texts = {}
        for pid in paper_ids:
            paper_text = dataset.get_paper(pid)
            if paper_text is None:
                logger.debug("Skipping missing paper: %s", pid)
                continue
            input_texts[pid] = paper_text

        if not input_texts:
            logger.error("No input files found for annotated papers; aborting")
            raise SystemExit(2)

        from .pipelines.adur_are import run_pipeline3


        adur_outdir_path, are_outdir_path = _derive_outdirs("pipeline3", use_adur_model_2=args.use_adur_model_2, use_are_model_2=args.use_are_model_2, explicit_adur=args.adur_outdir, explicit_are=args.are_outdir)
        
        if args.adur_outdir:
            adur_outdir_path = Path(args.adur_outdir)
        if args.are_outdir:
            are_outdir_path = Path(args.are_outdir)

        out = run_pipeline3(input_texts, adur_outdir=adur_outdir_path, are_outdir=are_outdir_path, adur_model_ref=args.adur_model, are_model_ref=args.are_model, use_adur_model_2=args.use_adur_model_2, use_are_model_2=args.use_are_model_2, major_method=args.major_method, dry_run=args.dry_run)
        logger.info("Pipeline3 completed; outputs at: %s", out)
        return out
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
