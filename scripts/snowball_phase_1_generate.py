"""Thin CLI to run Phase 1 End2End generation.

This script is now a lightweight wrapper that delegates almost all work to
the refactored Phase 1 modules. It expects you to activate the project's
venv before running (source ./.venv/bin/activate) and to run from the repo
root with PYTHONPATH=src.

Example usage (from repo root):
source ./.venv/bin/activate && PYTHONPATH=src python scripts/snowball_phase_1_generate.py --shot one-shot --device-index 0

TODO: Provide run info from this script to the logger creation process so that the log files are named more informatively.
TODO: Reduce redundancy between this script's configuration of shot strategies and its configuration of CoT strategies.
"""
from pathlib import Path
import argparse

from moralkg.snowball.phase_1.models.registry import create_end2end
from moralkg.snowball.phase_1.pipelines.orchestrator import Phase1Orchestrator
from moralkg.config import Config
from moralkg.logging import get_logger
from moralkg.argmining.loaders import Dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shot', type=str, default=None, choices=['zero-shot', 'one-shot', 'few-shot', 'all'],
                   help="Which shot strategy to run (maps to config prompts). Use 'all' to run all three")
    p.add_argument('--outdir', type=Path, default=None, help='Output directory (overrides config)')
    p.add_argument('--dry-run', action='store_true', help='Do not call model; validate prompts and paper coverage only')
    p.add_argument('--stream-save', action='store_true')
    p.add_argument('--checkpoint-interval', type=int, default=5)
    p.add_argument('--prompt-limit', type=int, default=None)
    p.add_argument('--paper-limit', type=int, default=None)
    p.add_argument('--name', type=str, default='run_ck')
    p.add_argument('--device-index', type=int, default=None, help='CUDA device index to forward to registry')
    p.add_argument('--rag', action='store_true', help='Enable retrieval-augmented generation (RAG) in the generator')
    p.add_argument('--cot', type=str, default=None, choices=['all_in_one', 'system_stepwise', 'user_stepwise', 'all'], 
                   help='Which chain-of-thought (CoT) strategy to use for multi-step orchestration in the generator. Use "all" to run all three.')
    args = p.parse_args()

    # Load config and resolve defaults for prompt_dir and outdir
    cfg = Config.load()

    # Configure logging via the project's helper. This will create a per-run
    # logfile (when configured) and attach a FileHandler to the root logger.
    logger = get_logger(__name__)

    # Discover where logs are being written and print the destination now so
    # the user sees it immediately (the file, or stdout when no dir).
    log_dest = None
    # Check the configured logger first, then the root logger handlers.
    # Prefer handlers that expose a 'baseFilename' attribute (FileHandler).
    for h in list(logger.handlers) + list(getattr(logger, 'root').handlers):
        if hasattr(h, 'baseFilename'):
            try:
                log_dest = Path(h.baseFilename)
            except Exception:
                log_dest = None
            break
    if log_dest:
        print('Run log writing to:', log_dest)
    else:
        print('Logs are being written to stdout (no log dir configured)')

    # Determine which shot strategies to run and resolve their prompt and output paths
    shot_choices = ['zero-shot', 'one-shot', 'few-shot'] if args.shot == 'all' else [args.shot] if args.shot else []
    shot_prompt_paths = {}
    shot_out_paths = {}

    cot_choices = ['all_in_one', 'system_stepwise', 'user_stepwise'] if args.cot == 'all' else [args.cot] if args.cot else []
    cot_prompt_paths = {}
    cot_out_paths = {}

    if shot_choices:
        for key in shot_choices:
            prompt_cfg_key = f'paths.snowball.phase_1.prompts.meta_llama_3.standard.{key}'
            out_cfg_key = f'paths.snowball.phase_1.outputs.end2end.standard.{key}'
            prompt_path = cfg.get(prompt_cfg_key)
            out_path = cfg.get(out_cfg_key)
            if not prompt_path:
                raise FileNotFoundError(f"Missing prompt path in config for shot '{key}' (key={prompt_cfg_key})")
            if not out_path and args.outdir is None:
                raise FileNotFoundError(f"Missing output path in config for shot '{key}' (key={out_cfg_key})")
            shot_prompt_paths[key] = Path(prompt_path)
            shot_out_paths[key] = Path(args.outdir) if args.outdir is not None else Path(out_path)
    
    elif args.cot:
        for key in cot_choices:
            prompt_cfg_key = f'paths.snowball.phase_1.prompts.meta_llama_3.cot.{key}'
            out_cfg_key = f'paths.snowball.phase_1.outputs.end2end.cot.{key}'
            prompt_path = cfg.get(prompt_cfg_key)
            out_path = cfg.get(out_cfg_key)
            if not prompt_path:
                raise FileNotFoundError(f"Missing prompt path in config for CoT '{key}' (key={prompt_cfg_key})")
            if not out_path and args.outdir is None:
                raise FileNotFoundError(f"Missing output path in config for CoT '{key}' (key={out_cfg_key})")
            cot_prompt_paths[key] = Path(prompt_path)
            cot_out_paths[key] = Path(args.outdir) if args.outdir is not None else Path(out_path)
    

    # Instantiate End2End via registry (may raise if models/deps are missing).
    # If doing a dry-run we do not create the heavy model.
    end2end = None
    if not args.dry_run:
        real_kwargs = {}
        if args.device_index is not None:
            real_kwargs['device_index'] = int(args.device_index)
        end2end = create_end2end(real_kwargs=real_kwargs)

    # Minimal dataset import; use the repository Dataset class
    dataset = Dataset()

    # Choose an orchestrator outdir (must be non-None). Prefer user-provided outdir
    if args.outdir is not None:
        orch_outdir = Path(args.outdir)
    else:
        orch_outdir = None

    orchestrator = Phase1Orchestrator(outdir=orch_outdir, generator_callable=end2end)

    final_paths = []
    for s in shot_choices:
        pd = shot_prompt_paths[s]
        od = shot_out_paths[s]
        logger.info("Running shot=%s prompts=%s outputs=%s", s, pd, od)
        path = orchestrator.generate_end2end(
            prompt_dir=pd,
            end2end_instance=end2end,
            dataset=dataset,
            outdir=od,
            stream_save=args.stream_save,
            checkpoint_interval=args.checkpoint_interval,
            prompt_limit=args.prompt_limit,
            paper_limit=args.paper_limit,
            dry_run=args.dry_run,
            name=f"{args.name}_{'dry_run_' if args.dry_run else ''}{s}",
        )
        final_paths.append(path)

    for c in cot_choices:
        pd = cot_prompt_paths[c]
        od = cot_out_paths[c]
        logger.info("Running CoT=%s prompts=%s outputs=%s", c, pd, od)
        path = orchestrator.generate_end2end_cot(
            prompt_dir=pd,
            strategy=c,
            end2end_instance=end2end,
            dataset=dataset,
            outdir=od,
            stream_save=args.stream_save,
            checkpoint_interval=args.checkpoint_interval,
            prompt_limit=args.prompt_limit,
            paper_limit=args.paper_limit,
            dry_run=args.dry_run,
            name=f"{args.name}_{'dry_run_' if args.dry_run else ''}{c}",
        )
        final_paths.append(path)

    print('Generation completed. Outputs written to:')
    for p in final_paths:
        print('  ', p)

if __name__ == '__main__':
    main()
