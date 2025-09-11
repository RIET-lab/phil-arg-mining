"""Prompt loader utilities for phase_1.

This module provides a small, well-documented prompt discovery and normalization
utility. It reads prompt directories produced by the project's existing
`generate_llm_prompts.py` script and yields deterministic PromptConfig objects.

The goal is to provide a stable shape for prompt metadata used by the batch
generation code in later PRs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PromptConfig:
    shot_type: str
    system_file: Path | None
    user_file: Path
    variation: str
    system_text: str | None
    user_text: str


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
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
    prompt_dir = Path(prompt_dir)
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    # Determine shot type from directory name
    shot_type = prompt_dir.name

    # collect system prompts (optional) and user prompts
    system_files = sorted(prompt_dir.glob("system_prompt*.txt"))
    user_files = sorted(prompt_dir.glob("user_prompt*.txt"))

    results: List[PromptConfig] = []

    if not user_files:
        raise ValueError(f"No user prompt found in {prompt_dir}")

    # If there are system prompts, pair each system with each user to create variations
    if system_files:
        for sysf in system_files:
            sys_text = _read_text(sysf)
            for uf in user_files:
                user_text = _read_text(uf) or ""
                # derive a stable variation key from the system filename
                variation = sysf.stem.replace("system_prompt_", "zs")
                results.append(PromptConfig(shot_type=shot_type,
                                            system_file=sysf,
                                            user_file=uf,
                                            variation=variation,
                                            system_text=sys_text,
                                            user_text=user_text))
    else:
        # No system prompt: create configs with system_file=None
        for uf in user_files:
            user_text = _read_text(uf) or ""
            results.append(PromptConfig(shot_type=shot_type,
                                        system_file=None,
                                        user_file=uf,
                                        variation="default",
                                        system_text=None,
                                        user_text=user_text))

    # Sort results deterministically by (variation, user filename)
    results.sort(key=lambda r: (r.variation, r.user_file.name))
    return results
