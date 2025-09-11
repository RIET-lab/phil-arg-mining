"""Thin CLI to run Phase 1 generation using the End2End pipeline.

Usage (from repo root, with venv activated):
  python scripts/snowball_phase_1_generate.py --prompt-dir <path> --outdir <path> --count 1

This script uses the models.registry factory to create a real End2End instance
and runs generation over prompt files, saving checkpoint(s) via the existing
BatchArgMapper/Phase1Orchestrator wiring.
"""
from pathlib import Path
import argparse
import sys
from importlib import util


def load_module(path: Path, mod_name: str):
    spec = util.spec_from_file_location(mod_name, str(path))
    mod = util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt-dir', type=Path, default=Path('src/moralkg/snowball/phase_1/prompts/meta_llama_3.1_8B_Instruct/standard/zero-shot'))
    p.add_argument('--outdir', type=Path, default=Path('temp/out_run'))
    p.add_argument('--count', type=int, default=1, help='Limit number of prompts to process')
    p.add_argument('--paper-limit', type=int, default=1, help='Limit number of papers to process')
    p.add_argument('--name', type=str, default='run_ck')
    args = p.parse_args()

    ROOT = Path(__file__).resolve().parents[1]

    # load registry and orchestrator modules
    reg_mod = load_module(ROOT / 'src' / 'moralkg' / 'snowball' / 'phase_1' / 'models' / 'registry.py', 'reg_mod')
    orch_mod = load_module(ROOT / 'src' / 'moralkg' / 'snowball' / 'phase_1' / 'pipelines' / 'orchestrator.py', 'orch_mod')
    loader_mod = load_module(ROOT / 'src' / 'moralkg' / 'snowball' / 'phase_1' / 'prompts' / 'loader.py', 'loader_mod')
    batch_mod = load_module(ROOT / 'src' / 'moralkg' / 'snowball' / 'phase_1' / 'batch' / 'generation.py', 'batch_mod')

    if not args.prompt_dir.exists():
        raise FileNotFoundError(f'Prompt dir not found: {args.prompt_dir}')

    # instantiate real End2End (may raise if model not available)
    end2end = reg_mod.create_end2end(allow_real=True)

    def generator_from_pipeline(cfg):
        system_text = cfg.system_text or ''
        user_text = cfg.user_text or ''
        return end2end.generate(system_text, user_text, prompt_files=None)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # load prompts and slice to count
    prompt_configs = loader_mod.load_prompts(args.prompt_dir)
    prompt_configs = prompt_configs[: args.count]

    # Load dataset and get annotated papers
    try:
        # Prefer package import (requires PYTHONPATH=src). This script assumes you run it from repo root with PYTHONPATH=src
        from moralkg.argmining.loaders import Dataset
        dataset = Dataset()
    except Exception as e:
        raise RuntimeError(
            "Could not import Dataset. Make sure to run this script from the repo root with PYTHONPATH=src and the venv activated."
        ) from e

    annotated_papers = list(dataset.annotations.by_paper.keys())
    if not annotated_papers:
        print('No annotated papers found in dataset; aborting')
        return

    if args.paper_limit is not None and args.paper_limit > 0:
        annotated_papers = annotated_papers[: args.paper_limit]

    # Prepare outputs list
    outputs = []

    for cfg in prompt_configs:
        system_text = cfg.system_text or ''
        user_template = cfg.user_text or ''
        for paper_id in annotated_papers:
            paper_text = dataset.get_paper(paper_id)
            if paper_text is None:
                continue
            # Replace placeholder
            processed_user = user_template.replace('<paper text inserted here>', paper_text)

            # Call model
            res = end2end.generate(system_text, processed_user, prompt_files=None)

            out_item = {
                'id': paper_id,
                'text': res.get('text', ''),
                'trace': res.get('trace', {}),
                'prompt_info': {
                    'shot_type': cfg.shot_type,
                    'system_file': str(cfg.system_file) if cfg.system_file else None,
                    'user_file': str(cfg.user_file),
                    'variation': cfg.variation,
                },
                'prompts_used': {
                    'system_prompt': system_text,
                    'user_prompt': processed_user,
                }
            }
            outputs.append(out_item)

    # Save full batch using checkpoint helper
    ck_path = None
    try:
        from moralkg.snowball.phase_1.io.checkpoints import save_batch
        ck_path = save_batch(outputs, outdir, args.name)
    except Exception:
        # Fallback: write JSON file
        import json
        outf = outdir / f"{args.name}.json"
        outf.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding='utf-8')
        ck_path = outf

    print('Checkpoint written to:', ck_path)


if __name__ == '__main__':
    main()
