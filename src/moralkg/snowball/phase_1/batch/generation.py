"""Minimal batch generation engine for prompt-based pipelines.

This implements a small BatchArgMapper that enumerates PromptConfig objects,
calls a provided generator (or real End2End pipeline) and persists outputs via
the checkpoints API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List
from moralkg.snowball.phase_1.prompts.loader import PromptConfig
from moralkg.snowball.phase_1.io.checkpoints import save_batch


class BatchArgMapper:
    def __init__(self, outdir: Path, generator: Callable[[PromptConfig], dict], checkpoint_interval: int = 10):
        self.outdir = Path(outdir)
        self.generator = generator
        self.checkpoint_interval = checkpoint_interval

    def run(self, prompt_configs: Iterable[PromptConfig], strategy: str = "default", name: str | None = None) -> Path:
        outdir = self.outdir
        outputs = []
        count = 0
        for cfg in prompt_configs:
            result = self.generator(cfg)
            # Normalize output artifact
            item = {
                "id": f"{cfg.shot_type}:{cfg.variation}:{cfg.user_file.name}",
                "text": result.get("text", ""),
                "prompt_info": {
                    "shot_type": cfg.shot_type,
                    "system_file": str(cfg.system_file) if cfg.system_file else None,
                    "user_file": str(cfg.user_file),
                    "variation": cfg.variation,
                },
                "trace": result.get("trace", {}),
            }
            outputs.append(item)
            count += 1

            if count % self.checkpoint_interval == 0:
                # write an intermittent checkpoint
                save_batch(outputs, outdir, name or f"partial_{strategy}_{count}")

        # final save
        final_path = save_batch(outputs, outdir, name or f"final_{strategy}")
        return final_path
