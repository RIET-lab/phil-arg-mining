Prompt utilities and loaders for Phase 1.

This directory hosts prompt loader code, templates, and related helpers (e.g. `loader.py`). It may also contain example templates.

API
- load_prompts(prompt_dir) -> List[PromptConfig]
- render_prompt(cfg, context) -> (system_text, user_text)
