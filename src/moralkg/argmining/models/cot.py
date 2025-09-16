"""
CoT Class:
Orchestrate multi-step reasoning prompts for End2End with optional retrieval-at-reasoning (RA-CoT) hooks. CoT does not load models or perform generation on its own; it relies on a generator callback provided by End2End.
"""
import re


class CoT:
    def __init__(
        self,
        *,
        steps: int = 2,
        step_prompts: dict | None = None,
        retrieval_step_positions: list[int] | None = None,
        logger=None,
        dry_run: bool = False,
    ) -> None:
        from moralkg import get_logger

        self.steps = max(1, steps)
        self.step_prompts = step_prompts or {}
        self.retrieval_step_positions = set(retrieval_step_positions or [])
        self.dry_run = dry_run
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_logger(__name__)

    # For experiments that require creating a fresh chat context per
    # CoT step (for example changing system instructions between steps),
    # we should provide a helper that invokes the generator as a chat API
    # (system_text, user_text) per step and optionally preserves or
    # discards prior assistant messages depending on the experiment.
    # The current `run` implementation composes a single user prompt per
    # step. Use `run_chat_sequence` to support chat-style stepwise
    # evaluation when needed.

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

            if self.dry_run:
                self.logger.info("CoT Step %d Prompt:\n%s", i, composed_user)
                output = "[dry_run]"
                prior_summary = output
                trace_steps.append({
                    "name": f"step_{i}",
                    "prompt": composed_user,
                    "output": output,
                    "used_context_ids": used_ids,
                })
                continue

            output = generator(composed_user)
            prior_summary = output.strip()

            trace_steps.append({
                "name": f"step_{i}",
                "prompt": composed_user if self.debug else "",
                "output": output,
                "used_context_ids": used_ids,
            })

        return {"final": prior_summary, "steps": trace_steps}

    def run_chat_sequence(
        self,
        *,
        initial_system: str | None,
        user_prompt: str,
        generator_chat_callable,
        retrieve=None,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ) -> dict:
        """Run CoT as a sequence of discrete chat calls.

        Each step would be invoked as a separate chat turn with its own
        system/user pair by calling `generator_chat_callable(system, user)`.
        This is a helper for 'system-stepwise' and 'user-stepwise' experiments.

        User-stepwise: preserve prior assistant outputs between steps by appending them to the user prompt.
        System-stepwise: fresh chat per step, but inject prior step outputs into the user prompt
        by replacing placeholders like <step 1 output>, <step 2 output>, etc.
        This keeps each step a fresh chat w.r.t. system message while making prior results
        available as variables for formatting.

        TODO: Make the strategy selection explicit via a parameter rather heuristic based on the number of prompts.
        """
        trace_steps: list[dict] = []
        prior_assistant = ""
        prior_outputs: dict[int, str] = {}

        # Normalize inputs
        initial_system = initial_system or ""
        user_prompt = user_prompt or ""

        # Determine strategy from provided step_prompts mapping if available.
        # If systems vary across steps -> system_stepwise (fresh chat per step).
        # If users vary across steps -> user_stepwise (preserve assistant outputs between steps).
        systems = set()
        users = set()
        for k, v in (self.step_prompts or {}).items():
            if isinstance(v, dict):
                systems.add((v.get("system") or "").strip())
                users.add((v.get("user") or "").strip())

        if len(systems) > 1:
            strategy = "system_stepwise"
        elif len(users) > 1 and len(systems) <= 1:
            strategy = "user_stepwise"
        else:
            raise ValueError("Cannot determine CoT chat strategy from step_prompts; need varying systems or users.")

        for i in range(1, self.steps + 1):
            key = f"step_{i}"
            cfg = (self.step_prompts or {}).get(key, {}) or {}

            # select system/user for this step
            system = (cfg.get("system") if isinstance(cfg, dict) else None) or initial_system
            user = (cfg.get("user") if isinstance(cfg, dict) else None) or user_prompt

            # helper to replace placeholders like <step 2 output> with prior outputs
            def _inject_outputs(text: str) -> str:
                if not text:
                    return text
                def _repl(m):
                    idx = int(m.group(1))
                    return prior_outputs.get(idx, "")
                return re.sub(r"<step\s*(\d+)\s*output>", _repl, text, flags=re.IGNORECASE)

            # retrieval hook (same semantics as run)
            contexts_txt = ""
            used_ids: list[str] = []
            if i in self.retrieval_step_positions and callable(retrieve):
                contexts = retrieve(user, 5)
                context_blocks = []
                for c in contexts:
                    used_ids.append(str(c.get("chunk_id")))
                    context_blocks.append(f"- [{c.get('chunk_id')}] {c.get('text','').strip()}")
                if context_blocks:
                    contexts_txt = "Context:\n" + "\n".join(context_blocks)

            # Compose user content depending on strategy
            if strategy == "system_stepwise":
                # system varies per step, but we still inject prior step outputs into
                # the (usually static) user prompt by replacing placeholders like
                # <step 1 output>, <step 2 output>, etc. This keeps each step a
                # fresh chat w.r.t. system message while making prior results
                # available as variables for formatting.
                injected_user = _inject_outputs(user)
                injected_system = _inject_outputs(system)
                composed_user = "\n\n".join([p for p in [contexts_txt, injected_user] if p])
                if self.dry_run:
                    output = "[dry_run]"
                else:
                    output = generator_chat_callable(injected_system, composed_user)
                # store this step's output for later injections
                prior_outputs[i] = output.strip()
                prior_assistant = output.strip()

            else:  # user_stepwise: accumulate assistant outputs in the conversation
                # For the first step, include contexts + user_prompt
                if i == 1:
                    composed_user = "\n\n".join([p for p in [contexts_txt, user] if p])
                else:
                    # Preserve assistant output from previous steps by appending it
                    composed_user = "\n\n".join([p for p in [contexts_txt, prior_assistant, user] if p])

                if self.dry_run:
                    output = "[dry_run]"
                else:
                    output = generator_chat_callable(system, composed_user)

                # accumulate assistant text for next step
                prior_assistant = (prior_assistant + "\n\n" + output).strip() if prior_assistant else output.strip()

            trace_steps.append({
                "name": key,
                "system": system if getattr(self, "debug", False) else "",
                "user": composed_user if getattr(self, "debug", False) else "",
                "output": output,
                "used_context_ids": used_ids,
            })

        return {"final": prior_assistant, "steps": trace_steps}

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
