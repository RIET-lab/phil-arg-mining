#!/usr/bin/env python3
"""CLI wrapper to evaluate outputs from pipelines 2 and 3.

This script reuses evaluators under `src.moralkg.argmining.snowball.evals` to
evaluate generated JSON outputs from pipeline2 and pipeline3. It mirrors the
CLI shape of `scripts/snowball_phase_1_e2e_eval.py` but adds a --pipeline
option to select which pipeline(s) to evaluate.

Usage (from repo root, with PYTHONPATH=src):
  PYTHONPATH=src python scripts/snowball_phase1_pipeline2_3_e2e_eval.py [options]


TODO: If the paper ID is not in the generated JSON, try to infer it from the filename. It's usually between the last '_' and the '.json' suffix. Then, add it to the file.


"""
from pathlib import Path
import argparse
import sys


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

# Reuse project config/logging and dataset helpers
from moralkg.config import Config
from moralkg.logging import get_logger
from moralkg.argmining.loaders import Dataset

# Reuse evaluators from argmining.snowball.evals (avoid duplicating logic)
from moralkg.snowball.phase_1.evals.eval import Evaluator as ModularEvaluator
from moralkg.argmining.schemas import ArgumentMap
from moralkg.argmining.parsers.parser import Parser as ModelOutputParser


def discover_input_files(input_file: Path, input_dir: Path):
    if input_file is not None:
        return [input_file]
    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
        return files
    return []


def collect_pipeline_outputs(cfg, pipeline: str, outdir_override: Path | None, logger) -> list[Path]:
    """Collect JSON files for pipeline2/3 from configured output paths.

    Mapping assumptions:
    - pipeline2 -> ARE outputs (paths.snowball.phase_1.outputs.are)
    - pipeline3 -> ADUR outputs (paths.snowball.phase_1.outputs.adur)
    - both -> collect from both ARE and ADUR
    """
    mapping = {
        'pipeline2': ['are'],
        'pipeline3': ['adur'],
        'both': ['are', 'adur'],
    }
    keys = mapping.get(pipeline, ['are'])
    collected = []
    for k in keys:
        for model_key in ('model_1', 'model_2'):
            cfg_key = f'paths.snowball.phase_1.outputs.{k}.{model_key}'
            try:
                path_val = cfg.get(cfg_key)
            except Exception:
                path_val = None
            if not path_val:
                continue
            model_dir = Path(outdir_override) if outdir_override is not None else Path(path_val)
            if not model_dir.exists():
                logger.debug('Configured model directory does not exist: %s', model_dir)
                continue

            # Look for pipeline-specific subdirectories (e.g., pipeline2_ARE_model_1...)
            desired_token = 'pipeline2' if 'pipeline2' in pipeline else ('pipeline3' if 'pipeline3' in pipeline else None)
            # If pipeline is 'both', we accept either token
            subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
            matched_files = []
            # If user requested both, accept subdirs containing 'pipeline2' or 'pipeline3'
            for d in subdirs:
                name = d.name.lower()
                if pipeline == 'both':
                    if 'pipeline2' in name or 'pipeline3' in name:
                        matched_files.extend([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
                else:
                    if desired_token and desired_token in name:
                        matched_files.extend([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == '.json'])

            # If we matched files in pipeline-specific subdirs, use them; otherwise
            # fall back to non-recursive listing of model_dir and then recursive glob
            if matched_files:
                files = sorted(matched_files)
            else:
                files = sorted([p for p in model_dir.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
                if not files:
                    files = sorted([p for p in model_dir.rglob('*.json') if p.is_file()])

            logger.info('Discovered %d json files in configured output path %s', len(files), model_dir)
            collected.extend(files)
    return collected


def infer_paper_id_from_filename(path: Path) -> str:
    """Simple heuristic: if filename contains underscores, return the last token; else return stem."""
    stem = path.stem
    if '_' in stem:
        return stem.split('_')[-1]
    return stem


def main(argv=None):
    p = argparse.ArgumentParser(description='Pipeline2 & Pipeline3 End-to-End evaluator CLI')
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument('--input-file', type=Path, help='Single generated JSON file to evaluate')
    grp.add_argument('--input-dir', type=Path, help='Directory containing generated JSON files (non-recursive)')
    p.add_argument('--pipeline', type=str, default='both', choices=['pipeline2', 'pipeline3', 'both'],
                   help='Which pipeline to evaluate')
    p.add_argument('--outdir', type=Path, default=None, help='Override configured output directory for the pipeline')
    p.add_argument('--parsed-argmap-dir', type=Path, default=None,
                   help='(Optional) Directory containing parsed argument-map artifacts used by the evaluator')
    p.add_argument('--out-file', type=Path, default=None, help='Optional path to write aggregated evaluation results (JSON)')
    p.add_argument('--save-parsed-json', action='store_true', help='Save per-paper parsed JSON outputs alongside parsed-argmap-dir')
    p.add_argument('--use-existing-parsed', action='store_true', help='If parsed JSONs already exist, prefer them to re-parsing')
    p.add_argument('--quiet', action='store_true', help='Less verbose logging')
    args = p.parse_args(argv)

    cfg = Config.load()
    logger = get_logger('snowball_phase1_pipeline2_3_eval')
    if args.quiet:
        try:
            logger.setLevel('INFO')
        except Exception:
            pass

    input_files = discover_input_files(args.input_file, args.input_dir)
    if not input_files:
        input_files = collect_pipeline_outputs(cfg, args.pipeline, args.outdir, logger)

    if not input_files:
        logger.error('No input files found to evaluate.')
        sys.exit(2)

    logger.info('Found %d input files to evaluate', len(input_files))

    # Resolve parsed_argmap_dir default from config if not provided.
    # Prefer parsed.are then parsed.adur regardless of pipeline selection because
    # both pipelines can produce outputs under both directories.
    if args.parsed_argmap_dir is None:
        parsed_val = None
        try:
            parsed_val = cfg.get('paths.snowball.phase_1.parsed.are') or cfg.get('paths.snowball.phase_1.parsed.adur')
        except Exception:
            parsed_val = None
        if parsed_val:
            args.parsed_argmap_dir = Path(parsed_val)
            logger.info('Using parsed-argmap-dir from config -> %s', args.parsed_argmap_dir)

    if (args.save_parsed_json or args.use_existing_parsed) and args.parsed_argmap_dir is None:
        p.error('--parsed-argmap-dir is required when using --save-parsed-json or --use-existing-parsed')

    dataset = Dataset()

    logger.info('Starting evaluation for pipeline=%s ...', args.pipeline)

    # For pipeline2/pipeline3 the outputs are per-paper ArgumentMap JSONs; use the
    # modular Evaluator logic in src/moralkg/snowball/phase_1/evals/eval.py.
    if args.pipeline in ("pipeline2", "pipeline3") or args.pipeline == "both":
        # Treat input_files as a list of per-paper ArgumentMap JSON files.
        parser = ModelOutputParser()
        pred_maps = []
        for p in input_files:
            try:
                # If the ArgumentMap lacks an id, try to infer from filename
                inferred = infer_paper_id_from_filename(p)
                am = parser.parse_json_file(p, map_id=inferred if inferred else None)

                pred_maps.append(am)
            except Exception as e:
                logger.warning('Failed to parse ArgumentMap from %s using Parser.parse_json_file: %s', p, e)

        if not pred_maps:
            logger.error('No valid ArgumentMap prediction files found for pipeline evaluation')
            sys.exit(2)

        # Load gold annotations from dataset. Use the dataset loader's internal
        # annotations object to find gold maps by paper id.
        try:
            annotations = getattr(dataset, 'annotations', None) or dataset._load_annotations()
        except Exception:
            annotations = dataset._load_annotations()

        gold_maps = []
        pred_maps_filtered = []
        for pm in pred_maps:
            paper_id = getattr(pm, 'id', None)
            if paper_id is None:
                logger.warning('Predicted ArgumentMap missing id, skipping: %s', getattr(pm, 'id', str(pm)))
                continue
            gold_list = getattr(annotations, 'by_paper', {}).get(paper_id) if annotations is not None else None
            if not gold_list:
                logger.warning('No gold annotation found for paper id %s, skipping', paper_id)
                continue
            # Many datasets store a list; take first
            gold_maps.append(gold_list[0])
            pred_maps_filtered.append(pm)

        if not gold_maps:
            logger.error('No matching gold annotations found for any predicted maps')
            sys.exit(2)

        # Instantiate modular evaluator. The metrics argument is unused by
        # evaluate_argument_maps_batch, so passing None is acceptable here.
        mod_eval = ModularEvaluator(dataset=dataset, metrics=None)
        try:
            batch_res = mod_eval.evaluate_argument_maps_batch(gold_maps, pred_maps_filtered)
        except Exception as e:
            logger.exception('ArgumentMap batch evaluation failed: %s', e)
            raise

        # batch_res contains 'average_metrics' and 'per_map_results'
        logger.info('Aggregate metrics: %s', batch_res.get('average_metrics'))

        # Save results
        out_file = args.out_file if args.out_file is not None else (args.parsed_argmap_dir / 'pipeline2_3_evaluation_results.json') if args.parsed_argmap_dir is not None else Path.cwd() / 'pipeline2_3_evaluation_results.json'
        try:
            import json as _json
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w', encoding='utf-8') as fh:
                _json.dump(batch_res, fh, indent=2, ensure_ascii=False)
            logger.info('Saved aggregated results to %s', out_file)
        except Exception:
            logger.exception('Failed to save results to %s', out_file)
            raise

        logger.info('Pipeline evaluation complete.')

if __name__ == '__main__':
    main()
