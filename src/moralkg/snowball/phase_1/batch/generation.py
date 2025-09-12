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
import logging


class BatchArgMapper:
    def __init__(self, outdir: Path, generator: Callable[[PromptConfig], dict], checkpoint_interval: int = 10):
        self.outdir = Path(outdir)
        self.generator = generator
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(__name__)
        self.logger.info("BatchArgMapper initialized: outdir=%s, checkpoint_interval=%d", self.outdir, self.checkpoint_interval)

    def run(self, prompt_configs: Iterable[PromptConfig], strategy: str = "default", name: str | None = None) -> Path:
        outdir = self.outdir
        outputs = []
        count = 0
        self.logger.info("Starting batch run: strategy=%s name=%s", strategy, name)
        for cfg in prompt_configs:
            self.logger.debug("Processing prompt: shot=%s variation=%s user_file=%s", cfg.shot_type, cfg.variation, cfg.user_file.name)
            try:
                result = self.generator(cfg)
            except Exception as e:
                self.logger.error("Generator failed for prompt %s/%s: %s", cfg.shot_type, cfg.variation, e)
                # continue to next prompt
                continue

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
                chk_name = name or f"partial_{strategy}_{count}"
                self.logger.info("Writing intermittent checkpoint: %s (count=%d)", chk_name, count)
                try:
                    save_batch(outputs, outdir, chk_name)
                except Exception as e:
                    self.logger.error("Failed to save intermittent checkpoint %s: %s", chk_name, e)

        # final save
        final_name = name or f"final_{strategy}"
        self.logger.info("Writing final batch file: %s (total=%d)", final_name, len(outputs))
        final_path = save_batch(outputs, outdir, final_name)
        self.logger.info("Final batch saved to: %s", final_path)
        return final_path


def run_file_mode(pipeline, input_files, outdir, normalize_fn, prefix: str = "output"):
    """Run a file-mode pipeline over a list of input files, normalize and save outputs.

    Args:
      pipeline: object with a `generate(input_path)` method that returns a raw dict
      input_files: iterable of file paths (str or Path)
      outdir: destination directory to write outputs
      normalize_fn: callable(raw_dict, source_text=None) -> normalized dict
      prefix: filename prefix for saved outputs

    Raises on first error to keep failures loud and visible for debugging.
    Returns the output directory Path on success.
    """
    logger = logging.getLogger(__name__)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for p in input_files:
        p_path = Path(p)
        logger.info("Processing file-mode input: %s", p_path)
        try:
            raw = pipeline.generate(p_path)
        except Exception as e:
            logger.error("Pipeline.generate failed for %s: %s", p_path, e)
            raise

        # try to read source text for normalization hints; ignore failures
        src_text = None
        try:
            src_text = p_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            src_text = None

        try:
            normalized = normalize_fn(raw, source_text=src_text)
        except Exception as e:
            logger.error("Normalization failed for %s: %s", p_path, e)
            raise

        filename = f"{prefix}_{p_path.stem}.json"
        try:
            from moralkg.snowball.phase_1.io.checkpoints import save_individual

            path = save_individual(normalized, outdir, filename)
            saved.append(path)
            logger.info("Saved normalized output to %s", path)
        except Exception as e:
            logger.error("Failed to save output for %s: %s", p_path, e)
            raise

    return outdir
