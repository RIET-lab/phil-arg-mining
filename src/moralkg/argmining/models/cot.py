"""
CoT Class:
Orchestrate multi-step reasoning prompts for End2End with optional retrieval-at-reasoning (RA-CoT) hooks. CoT does not load models or perform generation on its own; it relies on a generator callback provided by End2End.
"""


class CoT:
    def __init__(
        self,
        *,
        steps: int = 2,
        step_prompts: dict | None = None,
        retrieval_step_positions: list[int] | None = None,
        debug: bool = False,
    ) -> None:
        from moralkg import get_logger

        self.steps = max(1, steps)
        self.step_prompts = step_prompts or {}
        self.retrieval_step_positions = set(retrieval_step_positions or [])
        self.debug = debug

        self.logger = get_logger(__name__)

    def run(
        self,
        *,
        user_prompt: str,
        generator,
        retrieve=None,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        few_shot_examples=None,
    ) -> dict:
        """
        Execute a simple multi-step reasoning loop.
        """
        trace_steps: list[dict] = []
        prior_summary = ""

        for i in range(1, self.steps + 1):
            contexts_txt = ""
            used_ids: list[str] = []

            if i in self.retrieval_step_positions and callable(retrieve):
                contexts = retrieve(user_prompt, 5)
                context_blocks = []
                for c in contexts:
                    used_ids.append(str(c.get("chunk_id")))
                    context_blocks.append(f"- [{c.get('chunk_id')}] {c.get('text','').strip()}")
                if context_blocks:
                    contexts_txt = "Context:\n" + "\n".join(context_blocks)

            step_inst = self._step_instruction(i, user_prompt)

            prompt_parts = []
            if contexts_txt:
                prompt_parts.append(contexts_txt)
            if few_shot_examples:
                prompt_parts.append(self._render_few_shot(few_shot_examples))
            if prior_summary:
                prompt_parts.append("Previous Step Summary:\n" + prior_summary)
            prompt_parts.append(step_inst)
            composed_user = "\n\n".join([p for p in prompt_parts if p])

            output = generator(composed_user)
            prior_summary = output.strip()

            trace_steps.append({
                "name": f"step_{i}",
                "prompt": composed_user if self.debug else "",
                "output": output,
                "used_context_ids": used_ids,
            })

        return {"final": prior_summary, "steps": trace_steps}

    def _step_instruction(self, i: int, user_prompt: str) -> str:
        key = f"step_{i}"
        cfg = self.step_prompts.get(key)
        if cfg and isinstance(cfg, dict):
            sys = cfg.get("system", "").strip()
            usr = cfg.get("user", "").strip()
            parts = []
            if sys:
                parts.append(sys)
            if usr:
                parts.append(usr)
            return "\n\n".join(parts)
        # Default: repeat user prompt with a simple step directive
        return f"Step {i}: {user_prompt}"
