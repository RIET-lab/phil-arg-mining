"""Orchestrator for Phase 1 generation runs.

This is intentionally small: it loads prompts (via loader), creates a BatchArgMapper,
and runs the provided generator (which can be a real End2End instance or a test
callable). The goal is to produce a checkpoint file that can be evaluated later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from moralkg.snowball.phase_1.prompts.loader import load_prompts, PromptConfig
from moralkg.snowball.phase_1.batch.generation import BatchArgMapper


class Phase1Orchestrator:
    def __init__(self, outdir: Path, generator_callable):
        self.outdir = Path(outdir)
        self.generator = generator_callable

    def generate_from_prompt_dir(self, prompt_dir: Path, strategy: str = "default", name: str | None = None) -> Path:
        prompt_configs = load_prompts(Path(prompt_dir))
        mapper = BatchArgMapper(self.outdir, generator=self._wrap_generator(), checkpoint_interval=5)
        return mapper.run(prompt_configs, strategy=strategy, name=name)

    def _wrap_generator(self):
        # wrap the generator_callable to accept PromptConfig and return dict with text/trace
        def _gen(cfg: PromptConfig):
            return self.generator(cfg)

        return _gen
