"""Orchestrator for Phase 1 generation runs.

This is intentionally small: it loads prompts (via loader), creates a BatchArgMapper,
and runs the provided generator (which can be a real End2End instance or a test
callable). The goal is to produce a checkpoint file that can be evaluated later.

TODO: Load prompts from src/moralkg/snowball/phase_1/prompts/meta_llama_3.1_8B_Instruct/cot instead of the templates, so that we don't have to plug in the step-specific formatting instructions here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
from moralkg.snowball.phase_1.prompts.loader import load_prompts, PromptConfig, load_cot_prompts
from moralkg import Config
from moralkg.snowball.phase_1.batch.generation import BatchArgMapper
import logging
from moralkg.snowball.phase_1.prompts.loader import render_prompt
from moralkg.snowball.phase_1.io.checkpoints import save_individual, save_batch, save_checkpoint


class Phase1Orchestrator:
    def __init__(self, outdir: Path | None, generator_callable):
        self.outdir = Path(outdir) if outdir else None
        self.generator = generator_callable
        self.logger = logging.getLogger(__name__)
        self.logger.info("Phase1Orchestrator initialized: outdir=%s", self.outdir)

    def generate_from_prompt_dir(self, prompt_dir: Path, strategy: str = "default", name: str | None = None) -> Path:
        self.logger.info("Loading prompts from: %s", prompt_dir)
        prompt_configs = load_prompts(Path(prompt_dir))
        self.logger.info("Loaded %d prompt configurations", len(prompt_configs))
        mapper = BatchArgMapper(self.outdir, generator=self._wrap_generator(), checkpoint_interval=5)
        self.logger.info("Starting generation with generator: %s", getattr(self.generator, '__name__', str(self.generator)))
        path = mapper.run(prompt_configs, strategy=strategy, name=name)
        self.logger.info("Generation completed, outputs at: %s", path)
        return path

    def generate_end2end(self,
                         prompt_dir: Path,
                         end2end_instance,
                         dataset,
                         outdir: Path | None = None,
                         stream_save: bool = False,
                         checkpoint_interval: int = 50,
                         prompt_limit: int | None = None,
                         paper_limit: int | None = None,
                         dry_run: bool = False,
                         name: str | None = None) -> Path:
        """High-level End2End generation over prompt files and annotated papers.

        This method implements the per-prompt, per-paper loop that previously
        lived in the CLI script. It renders prompts, calls the End2End
        generator, and persists outputs using the checkpoints API.
        """
        outdir = Path(outdir or self.outdir)
        # if outdir was not provided at init time, it must be provided here
        if outdir is None:
            raise ValueError("Outdir must be provided either at orchestrator init or at generate_end2end time")
        # if outdir was provided at init time and also here, ensure they match and warn if not
        elif self.outdir is not None and outdir != self.outdir:
            self.logger.warning("Outdir provided at generate_end2end (%s) differs from orchestrator init (%s); using %s",
                                outdir, self.outdir, outdir)
        
        outdir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting End2End generation: prompt_dir=%s outdir=%s stream_save=%s",
                         prompt_dir, outdir, stream_save)

        # Load standard prompt configs (zero/one/few-shot)
        prompt_configs = load_prompts(Path(prompt_dir))
        if prompt_limit:
            prompt_configs = prompt_configs[:prompt_limit]
        self.logger.info("Will run %d prompt configurations", len(prompt_configs))

        # collect annotated papers from dataset
        try:
            annotated_papers = list(dataset.annotations.by_paper.keys())
        except Exception:
            # fallback: dataset may expose helper
            annotated_papers = list(getattr(dataset, 'metadata', {}).get('ids', []))

        if paper_limit:
            annotated_papers = annotated_papers[:paper_limit]

        if not annotated_papers:
            self.logger.warning("No annotated papers found in dataset; nothing to do")
            return outdir

        total_expected = len(prompt_configs) * len(annotated_papers)
        self.logger.info("Total combinations to process: %d (prompts=%d papers=%d)",
                         total_expected, len(prompt_configs), len(annotated_papers))

        produced = 0
        outputs = []

        for cfg in prompt_configs:
            self.logger.info("Processing prompt: shot=%s variation=%s user=%s",
                             cfg.shot_type, cfg.variation, cfg.user_file.name)
            for paper_id in annotated_papers:
                paper_text = dataset.get_paper(paper_id)
                if paper_text is None:
                    self.logger.debug("Skipping missing paper: %s", paper_id)
                    continue
                try:
                    system_text, user_text = render_prompt(cfg, {"paper_text": paper_text})
                except KeyError as exc:
                    self.logger.warning("Skipping prompt due to missing placeholder: %s", exc)
                    continue

                if dry_run:
                    # Do not call the model; just validate render and record an empty placeholder
                    self.logger.debug("Dry-run: would generate for paper=%s prompt=%s", paper_id, cfg.variation)
                    res = {"text": "", "trace": {"dry_run": True}}
                else:
                    try:
                        res = end2end_instance.generate(system_text, user_text, prompt_files=None)
                    except Exception as e:
                        self.logger.error("End2End.generate failed for paper %s prompt %s: %s", paper_id, cfg.variation, e)
                        continue

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
                        'user_prompt': user_text,
                    }
                }

                produced += 1
                outputs.append(out_item)

                # stream save
                if stream_save:
                    fname = f"{paper_id}__{cfg.variation}__{cfg.user_file.name}.json"
                    try:
                        save_individual(out_item, outdir / 'individual_outputs', fname)
                    except Exception as e:
                        self.logger.error("Failed to save individual output %s: %s", fname, e)

                # intermittent checkpointing
                if checkpoint_interval and produced % checkpoint_interval == 0:
                    chk_name = name or f"checkpoint_{cfg.variation}_{produced}"
                    try:
                        save_checkpoint(outputs, outdir / 'checkpoints', name=chk_name, strategy='end2end')
                        self.logger.info("Wrote intermittent checkpoint: %s (produced=%d)", chk_name, produced)
                    except Exception as e:
                        self.logger.error("Failed to write intermittent checkpoint: %s", e)

        # final save
        final_name = name or f"final_{prompt_dir.name}"
        try:
            final_path = save_batch(outputs, outdir, final_name)
            self.logger.info("Final outputs saved to: %s (total=%d)", final_path, len(outputs))
        except Exception as e:
            self.logger.error("Failed to save final batch: %s", e)
            raise

        return final_path
    
    def generate_end2end_cot(self,
                             prompt_dir: Path,
                             strategy: str,
                             end2end_instance,
                             dataset,
                             outdir: Path | None = None,
                             stream_save: bool = False,
                             checkpoint_interval: int = 5,
                             prompt_limit: int | None = None,
                             paper_limit: int | None = None,
                             dry_run: bool = False,
                             name: str | None = None) -> Path:
        """High-level End2End generation with Chain-of-Thought (CoT) strategy.
        """
        outdir = Path(outdir or self.outdir)
        if outdir is None:
            raise ValueError("Outdir must be provided either at orchestrator init or at generate_end2end_cot time")
        outdir.mkdir(parents=True, exist_ok=True)

        # Use the CoT-aware loader so that `cfg.step_files` is populated for
        # system_stepwise / user_stepwise prompt directories. The loader will
        # return PromptConfig objects that include `step_files` when present.
        # For all_in_one prompts, `step_files` will be None.
        prompt_configs = load_cot_prompts(Path(prompt_dir))
        if prompt_limit:
            prompt_configs = prompt_configs[:prompt_limit]

        # collect annotated papers
        try:
            annotated_papers = list(dataset.annotations.by_paper.keys())
        except Exception:
            annotated_papers = list(getattr(dataset, 'metadata', {}).get('ids', []))

        if paper_limit:
            annotated_papers = annotated_papers[:paper_limit]

        if not annotated_papers:
            self.logger.warning("No annotated papers found in dataset; nothing to do (CoT)")
            return outdir

        produced = 0
        outputs = []

        # Main loop similar to generate_end2end but orchestrates CoT per strategy
        for cfg in prompt_configs:
            self.logger.info("CoT Processing prompt: shot=%s variation=%s user=%s strategy=%s",
                             cfg.shot_type, cfg.variation, getattr(cfg.user_file, 'name', str(cfg.user_file)), strategy)
            for paper_id in annotated_papers:
                paper_text = dataset.get_paper(paper_id)
                if paper_text is None:
                    self.logger.debug("Skipping missing paper: %s", paper_id)
                    continue

                # Prepare defaults
                prompt_files = {}
                steps = {}

                if getattr(cfg, 'step_files', None):
                    # Read and render each step's system/user templates with the paper_text
                    for k in sorted(cfg.step_files.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0):
                        item = cfg.step_files[k]
                        sys_tpl = item.get('system')
                        usr_tpl = item.get('user')
                        sys_txt = sys_tpl.read_text(encoding='utf-8').replace('<paper text inserted here>', paper_text) if sys_tpl is not None else ''
                        usr_txt = usr_tpl.read_text(encoding='utf-8').replace('<paper text inserted here>', paper_text) if usr_tpl is not None else ''
                        steps[k] = {'system': sys_txt, 'user': usr_txt}

                    # For convenience, derive a base system/user from step_1 if needed
                    base_system = steps.get('step_1', {}).get('system', '')
                    base_user = steps.get('step_1', {}).get('user', '')

                else:
                    # Non-stepwise (all_in_one): render using existing loader/render_prompt
                    try:
                        base_system, base_user = render_prompt(cfg, {"paper_text": paper_text})
                    except KeyError as exc:
                        self.logger.warning("Skipping prompt due to missing placeholder: %s", exc)
                        continue

                # Build different behaviors depending on strategy
                strat = (strategy or '').lower().replace('-', '_')

                # all-in-one: zero-shot generation
                if strat == 'all_in_one' or (not steps):
                    system_text = base_system
                    user_text = base_user
                    if dry_run:
                        res = {"text": "", "trace": {"dry_run": True, "strategy": strategy}}
                    else:
                        try:
                            res = end2end_instance.generate(system_text, user_text, prompt_files=None)
                        except Exception as e:
                            self.logger.error("End2End.generate (all-in-one) failed for paper %s prompt %s: %s", paper_id, cfg.variation, e)
                            continue

                else:
                    # stepwise strategies: system_stepwise or user_stepwise
                    if strat not in ['system_stepwise', 'user_stepwise']:
                        self.logger.warning("Unknown CoT strategy '%s'; skipping", strategy)
                        continue
                    if not steps:
                        self.logger.warning("No step definitions found for stepwise CoT; skipping")
                        continue

                    if dry_run:
                        res = {"text": "", "trace": {"dry_run": True, "strategy": strategy}}
                    else:
                        try:
                            # End2End expects stepwise prompts under the key
                            # 'step_prompts' in `prompt_files`. Wrap the
                            # mapping built above accordingly so the CoT
                            # orchestration code (CoT.run_chat_sequence) can
                            # detect and exercise system_stepwise / user_stepwise
                            # behaviors (separate chats vs. single-session).
                            res = end2end_instance.generate(
                                system_text=base_system,
                                user_text=base_user,
                                prompt_files={"step_prompts": steps},
                                cot_strategy=strat,
                            )
                        except Exception as e:
                            self.logger.error("End2End.generate (stepwise) failed for paper %s prompt %s: %s", paper_id, cfg.variation, e)
                            continue

                out_item = {
                    'id': paper_id,
                    'text': res.get('text', ''),
                    'trace': res.get('trace', {}),
                    'prompt_info': {
                        'system_file': str(cfg.system_file) if cfg.system_file else None,
                        'user_file': str(cfg.user_file),
                        'cot_strategy': strategy,
                    },
                    'prompts_used': {
                        'system_prompt': base_system,
                        'user_prompt': base_user,
                        'step_prompts': list(steps.values()),
                    }
                }

                produced += 1
                outputs.append(out_item)

                # stream save
                if stream_save:
                    fname = f"{paper_id}__cot__{strategy}__{cfg.user_file.name}.json"
                    try:
                        save_individual(out_item, outdir / 'individual_outputs', fname)
                    except Exception as e:
                        self.logger.error("Failed to save individual output %s: %s", fname, e)

                # intermittent checkpointing
                if checkpoint_interval and produced % checkpoint_interval == 0:
                    chk_name = name or f"cot_checkpoint_{cfg.variation}_{produced}"
                    try:
                        save_checkpoint(outputs, outdir / 'checkpoints', name=chk_name, strategy='end2end_cot')
                        self.logger.info("Wrote intermittent CoT checkpoint: %s (produced=%d)", chk_name, produced)
                    except Exception as e:
                        self.logger.error("Failed to write intermittent CoT checkpoint: %s", e)

        # final save
        final_name = name or f"cot_final_{prompt_dir.name}"
        try:
            final_path = save_batch(outputs, outdir, final_name)
            self.logger.info("Final CoT outputs saved to: %s (total=%d)", final_path, len(outputs))
        except Exception as e:
            self.logger.error("Failed to save final CoT batch: %s", e)
            raise

        return final_path

    def _wrap_generator(self):
        # wrap the generator_callable to accept PromptConfig and return dict with text/trace
        def _gen(cfg: PromptConfig):
            return self.generator(cfg)

        return _gen

    # ---- Pipeline 2 & 3 helpers (ADUR/ARE) ----
    def run_pipeline2(self, input_files: Iterable[Path], outdir: Path | None = None, adur_model_ref: Any = None, are_model_ref: Any = None, **kwargs) -> Path:
        """Convenience wrapper to run Pipeline 2 (ADUR -> ARE).

        Accepts the same arguments as `pipelines.adur_are.run_pipeline2` and
        delegates to that implementation. This keeps the orchestrator as the
        single public entrypoint for Phase 1 orchestration.
        """
        outdir = Path(outdir or self.outdir)
        from moralkg.snowball.phase_1.pipelines.adur_are import run_pipeline2

        return run_pipeline2(input_files=input_files, outdir=outdir, adur_model_ref=adur_model_ref, are_model_ref=are_model_ref, **kwargs)

    def run_pipeline3(self, input_files: Iterable[Path], outdir: Path | None = None, adur_model_ref: Any = None, are_model_ref: Any = None, **kwargs) -> Path:
        """Convenience wrapper to run Pipeline 3 (ADUR -> Major ADU -> ARE).

        Delegates to `pipelines.adur_are.run_pipeline3`.
        """
        outdir = Path(outdir or self.outdir)
        from moralkg.snowball.phase_1.pipelines.adur_are import run_pipeline3

        return run_pipeline3(input_files=input_files, outdir=outdir, adur_model_ref=adur_model_ref, are_model_ref=are_model_ref, **kwargs)
