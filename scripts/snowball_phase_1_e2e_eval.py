"""CLI wrapper for Phase 1 End-to-End evaluation.

This script is a thin bridge between archived evaluation usage and the refactored
`src.moralkg.snowball.phase_1` modules. It intentionally avoids importing or
modifying any generation code so it is safe to run while generation jobs are
running in the background.

Usage examples (from repo root, with venv active and PYTHONPATH=src):

# Evaluate a single generated JSON file
PYTHONPATH=src python scripts/snowball_phase_1_e2e_eval.py --input-file datasets/interim/.../generation_outputs.json

# Evaluate all JSON files in a folder and save aggregated results
PYTHONPATH=src python scripts/snowball_phase_1_e2e_eval.py --input-dir datasets/interim/... --parsed-argmap-dir datasets/processed/... --out-file evaluation_results.json

"""
from pathlib import Path
import argparse
import sys

# Use the project's config and logging helpers
from moralkg.config import Config
from moralkg.logging import get_logger
from moralkg.argmining.loaders import Dataset

# Phase 1 evaluator
from moralkg.snowball.phase_1.evals.end2end_eval import End2EndEvaluator


def discover_input_files(input_file: Path, input_dir: Path):
    """Return a list of input files to evaluate.

    Preference order:
    - If input_file is provided, return [input_file]
    - Else if input_dir is provided, return all .json files directly under it (non-recursive)
    - Else empty list
    """
    if input_file is not None:
        return [input_file]
    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
        return files
    return []


def summarize_and_print(logger, results):
    logger.info("%s", "=" * 50)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("%s", "=" * 50)
    logger.info(f"Total papers: {getattr(results, 'total_papers', 'N/A')}")
    logger.info(f"Successful parses: {getattr(results, 'successful_parses', 'N/A')}")
    logger.info(f"Failed parses: {getattr(results, 'failed_parses', 'N/A')}")
    try:
        rate = getattr(results, 'parse_success_rate', None)
        if rate is not None:
            logger.info(f"Parse success rate: {rate:.2%}")
    except Exception:
        pass

    if getattr(results, 'aggregate_metrics', None):
        logger.info("\nAggregate Metrics:")
        for metric, value in results.aggregate_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    if getattr(results, 'strategy_breakdown', None):
        logger.info("\nStrategy Breakdown:")
        for strategy, metrics in results.strategy_breakdown.items():
            logger.info(f"  {strategy}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 1 End-to-End evaluator (thin CLI)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--input-file', type=Path, help='Single generated JSON file to evaluate')
    grp.add_argument('--input-dir', type=Path, help='Directory containing generated JSON files (non-recursive)')
    # If neither input-file nor input-dir are provided the script will fall back
    # to resolving shot-based output directories from `config.yaml` (same
    # keys used by `scripts/snowball_phase_1_generate.py`).
    p.add_argument('--shot', type=str, default='all', choices=['zero-shot', 'one-shot', 'few-shot', 'all'],
                   help="Which shot strategy to evaluate (maps to config prompts). Use 'all' to evaluate all three")
    p.add_argument('--outdir', type=Path, default=None, help='Override output directory (applies to all shot choices)')
    p.add_argument('--parsed-argmap-dir', type=Path, required=True,
                   help='Directory containing parsed argument-map artifacts used by the evaluator (processed argmap path)')
    p.add_argument('--out-file', type=Path, default=None, help='Optional path to write aggregated evaluation results (JSON). If omitted, results are saved to parsed-argmap-dir/evaluation_results.json')
    p.add_argument('--save-parsed-json', action='store_true', help='Save per-paper parsed JSON outputs alongside parsed-argmap-dir')
    p.add_argument('--use-existing-parsed', action='store_true', help='If parsed JSONs already exist, prefer them to re-parsing')
    p.add_argument('--quiet', action='store_true', help='Less verbose logging')
    args = p.parse_args(argv)

    # Load config and logger
    cfg = Config.load()
    logger = get_logger('snowball_phase_1_e2e_eval')
    if args.quiet:
        # reduce logger level to INFO if requested
        try:
            logger.setLevel('INFO')
        except Exception:
            pass

    input_files = discover_input_files(args.input_file, args.input_dir)
    # If no explicit input files/dir provided, resolve shot-based output dirs
    if not input_files:
        # Determine shot choices
        shot_choices = ['zero-shot', 'one-shot', 'few-shot'] if args.shot == 'all' else [args.shot]

        shot_out_paths = {}
        for key in shot_choices:
            out_cfg_key = f'paths.snowball.phase_1.outputs.end2end.standard.{key}'
            out_path = cfg.get(out_cfg_key)
            if not out_path and args.outdir is None:
                raise FileNotFoundError(f"Missing output path in config for shot '{key}' (key={out_cfg_key})")
            shot_out_paths[key] = Path(args.outdir) if args.outdir is not None else Path(out_path)

        # Collect JSON files from each shot output dir (non-recursive)
        collected = []
        for s, od in shot_out_paths.items():
            if not od.exists():
                logger.warning('Configured output dir for shot=%s does not exist: %s', s, od)
                continue
            files = sorted([p for p in od.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
            logger.info('Discovered %d json files in output dir for shot=%s -> %s', len(files), s, od)
            collected.extend(files)

        input_files = collected
    if not input_files:
        logger.error('No input files found to evaluate.')
        sys.exit(2)

    logger.info('Found %d input files to evaluate', len(input_files))

    # Minimal dataset import; use repository Dataset class
    dataset = Dataset()

    # create evaluator
    evaluator = End2EndEvaluator(
        dataset=dataset,
        parsed_argmap_dir=args.parsed_argmap_dir,
        save_parsed_argument_json=args.save_parsed_json,
        use_existing_parsed_if_found=args.use_existing_parsed,
    )

    # run evaluation
    logger.info('Starting evaluation...')
    try:
        results = evaluator.evaluate_output_files(input_files)
    except Exception as e:
        logger.exception('Evaluation failed with exception: %s', e)
        raise

    # Print and save summary
    summarize_and_print(logger, results)

    out_file = args.out_file if args.out_file is not None else (args.parsed_argmap_dir / 'evaluation_results.json')
    logger.info('Saving aggregated results to %s', out_file)
    try:
        evaluator.save_results(results, out_file)
    except Exception:
        logger.exception('Failed to save results to %s', out_file)
        raise

    logger.info('Evaluation complete.')


if __name__ == '__main__':
    main()
