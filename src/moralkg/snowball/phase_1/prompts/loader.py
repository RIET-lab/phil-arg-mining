"""Prompt loader utilities for phase_1.

This module provides a small, well-documented prompt discovery and normalization
utility. It reads prompt directories produced by the project's existing
`generate_llm_prompts.py` script and yields deterministic PromptConfig objects.

The goal is to provide a stable shape for prompt metadata used by the batch
generation code in later PRs.

TODO: Log a bit less verbosely by default; add a "VERBOSE" level to the config logger options and make use of it here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging


@dataclass(frozen=True)
class PromptConfig:
    shot_type: str
    system_file: Path | None
    user_file: Path
    variation: str
    system_text: str | None
    user_text: str
    # For stepwise CoT prompts, a mapping of step -> {'system': Path|None, 'user': Path|None}
    step_files: dict | None = None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger = logging.getLogger(__name__)
        logger.debug("Prompt file not found when reading: %s", path)
        return None


def load_prompts(prompt_dir: Path) -> List[PromptConfig]:
    """Discover and load prompt files from a prompt directory.

    Expected layout (examples exist in the repo):
      <prompt_dir>/system_prompt_1.txt
      <prompt_dir>/user_prompt.txt

    The function returns a stable, sorted list of PromptConfig objects.

    Args:
      prompt_dir: path to a shot-type directory (e.g. .../meta_llama_3.1_8B_Instruct/standard/zero-shot)

    Returns:
      List[PromptConfig]
    """
    logger = logging.getLogger(__name__)
    prompt_dir = Path(prompt_dir)
    logger.info("Loading prompts from directory: %s", prompt_dir)
    if not prompt_dir.exists():
        logger.error("Prompt directory not found: %s", prompt_dir)
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    # Determine shot type from directory name
    shot_type = prompt_dir.name

    # collect system prompts (optional) and user prompts
    system_files = sorted(prompt_dir.glob("system_prompt*.txt"))
    user_files = sorted(prompt_dir.glob("user_prompt*.txt"))

    logger.debug("Found %d system prompt(s) and %d user prompt(s)",
                 len(system_files), len(user_files))

    results: List[PromptConfig] = []

    if not user_files:
        logger.error("No user prompt files found in %s", prompt_dir)
        raise ValueError(f"No user prompt found in {prompt_dir}")

    # If there are system prompts, pair each system with each user to create variations
    if system_files:
        for sysf in system_files:
            sys_text = _read_text(sysf)
            logger.debug("Loaded system prompt file: %s (len=%s)", sysf.name,
                         len(sys_text) if sys_text is not None else 0)
            for uf in user_files:
                user_text = _read_text(uf) or ""
                logger.debug("Loaded user prompt file: %s (len=%d)", uf.name, len(user_text))
                # derive a stable variation key from the system filename
                variation = sysf.stem.replace("system_prompt_", "zs")
                cfg = PromptConfig(shot_type=shot_type,
                                   system_file=sysf,
                                   user_file=uf,
                                   variation=variation,
                                   system_text=sys_text,
                                   user_text=user_text,
                                   step_files=None)
                logger.debug("Created PromptConfig: %s/%s (variation=%s)",
                             sysf.name, uf.name, variation)
                results.append(cfg)
    else:
        # No system prompt: create configs with system_file=None
        logger.debug("No system prompts found; creating default PromptConfig(s)")
        for uf in user_files:
            user_text = _read_text(uf) or ""
            cfg = PromptConfig(shot_type=shot_type,
                               system_file=None,
                               user_file=uf,
                               variation="default",
                               system_text=None,
                               user_text=user_text,
                               step_files=None)
            logger.debug("Created PromptConfig (no system): %s (variation=default)", uf.name)
            results.append(cfg)

    # Sort results deterministically by (variation, user filename)
    results.sort(key=lambda r: (r.variation, r.user_file.name))
    logger.info("Loaded %d prompt configurations from %s", len(results), prompt_dir)
    return results


def load_cot_prompts(prompt_dir: Path) -> List[PromptConfig]:
    """Load Chain-of-Thought (CoT) prompts from a meta-llama style folder.

    This function expects the prompt_dir to be one of the strategy subfolders
    (e.g. `.../cot/all_in_one`, `.../cot/system_stepwise`, `.../cot/user_stepwise`)
    and will fail loudly if required files are missing or inconsistent.
    It returns a list with a single PromptConfig containing `step_files` when
    appropriate, or the same shape as `load_prompts` for `all_in_one`.
    """
    logger = logging.getLogger(__name__)
    prompt_dir = Path(prompt_dir)
    logger.info("Loading CoT prompts from: %s", prompt_dir)
    if not prompt_dir.exists():
        logger.error("CoT prompt directory not found: %s", prompt_dir)
        raise FileNotFoundError(f"CoT prompt directory not found: {prompt_dir}")

    cot_strategy = prompt_dir.name
    system_files = sorted(prompt_dir.glob("system_prompt*.txt"))
    user_files = sorted(prompt_dir.glob("user_prompt*.txt"))

    results: List[PromptConfig] = []

    if 'all_in_one' in cot_strategy.lower():
        sysf = next((p for p in system_files), None)
        uf = next((p for p in user_files if p.stem == 'user_prompt' or p.stem.endswith('_1')), None)
        if sysf is None or uf is None:
            raise ValueError(f"all_in_one missing expected user prompt in {prompt_dir}")
        sys_text = _read_text(sysf) if sysf else None
        user_text = _read_text(uf) or ""
        # step_files = {f"step_1": {"system": sysf, "user": uf}}
        cfg = PromptConfig(cot_strategy=cot_strategy,
                           system_file=sysf,
                           user_file=uf,
                           system_text=sys_text,
                           user_text=user_text,
                           step_files=None)
        results.append(cfg)
        return results

    if 'system_stepwise' in cot_strategy.lower():
        if not system_files or not user_files:
            logger.error("system_stepwise requires both system_prompt_N and user_prompt_N in %s", prompt_dir)
            raise ValueError(f"system_stepwise incomplete: {prompt_dir}")
        steps = {}
        n_steps = min(len(system_files), len(user_files))
        for i in range(n_steps):
            steps[f'step_{i+1}'] = {'system': system_files[i], 'user': user_files[i]}
        cfg = PromptConfig(cot_strategy=cot_strategy,
                           system_file=prompt_dir,
                           user_file=prompt_dir,
                           system_text=None,
                           user_text=None,
                           step_files=steps)
        results.append(cfg)
        return results

    if 'user_stepwise' in cot_strategy.lower():
        if not system_files or not user_files:
            logger.error("user_stepwise requires a system_prompt and multiple user_prompt_N files in %s", prompt_dir)
            raise ValueError(f"user_stepwise incomplete: {prompt_dir}")
        sysf = system_files[0]
        steps = {}
        for i, uf in enumerate(user_files, start=1):
            steps[f'step_{i}'] = {'system': sysf, 'user': uf}
        cfg = PromptConfig(cot_strategy=cot_strategy,
                           system_file=sysf,
                           user_file=prompt_dir,
                           system_text=None,
                           user_text=None,
                           step_files=steps)
        results.append(cfg)
        return results

    logger.error("Unrecognized CoT prompt folder type (expected all_in_one/system_stepwise/user_stepwise): %s", prompt_dir)
    raise ValueError(f"Unrecognized CoT prompt folder: {prompt_dir}")


def render_prompt(cfg: PromptConfig, context: dict) -> tuple[str, str]:
    """Render a PromptConfig into concrete (system_text, user_text).

    - If cfg.system_text is present, it will be formatted with context.
    - The user_text will be formatted with context. If formatting fails due to
      missing keys, a KeyError is raised.

    The context may contain Path values which will be replaced by file contents.
    """
    # Simplified rendering: only support the legacy literal placeholder
    # '<paper text inserted here>' which will be replaced by the provided
    # context['paper_text'] (or empty string if missing). This avoids
    # attempting to format arbitrary files that may contain JSON or other
    # braces and produce spurious KeyErrors.
    logger = logging.getLogger(__name__)

    def _resolve_value(v):
        try:
            if isinstance(v, Path):
                txt = v.read_text(encoding='utf-8', errors='ignore')
                logger.debug("Resolved Path value for rendering: %s (len=%d)", v, len(txt))
                return txt
        except Exception as e:
            logger.warning("Could not read path while rendering prompt: %s (%s)", v, e)
        return v

    safe_ctx = {k: _resolve_value(v) for k, v in context.items()}

    sys_tpl = cfg.system_text or ""
    usr_tpl = cfg.user_text or ""

    paper_text = str(safe_ctx.get('paper_text', ''))

    try:
        sys_text = sys_tpl.replace('<paper text inserted here>', paper_text) if sys_tpl else ''
        user_text = usr_tpl.replace('<paper text inserted here>', paper_text)
        logger.debug("Rendered prompt (shot=%s, variation=%s): system_len=%d user_len=%d",
                     cfg.shot_type, cfg.variation, len(sys_text), len(user_text))
        return sys_text, user_text
    except Exception as e:
        logger.error("Error rendering prompt for %s/%s: %s", cfg.shot_type, cfg.variation, e)
        raise
