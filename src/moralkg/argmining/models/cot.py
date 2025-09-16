"""
CoT Class:
Orchestrate multi-step reasoning prompts for End2End with optional retrieval-at-reasoning (RA-CoT) hooks. CoT does not load models or perform generation on its own; it relies on a generator callback provided by End2End.
"""
from logging import warning
import re
from venv import logger


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

    def run_system_stepwise(
        self,
        *,
        initial_system: str | None,
        user_prompt: str,
        generator_chat_callable,
        retrieve=None,
    ) -> dict:
        """Run CoT as a sequence of discrete chat calls with system message varying per step.

        This keeps each step a fresh chat w.r.t. system message while making prior
        results available as variables for formatting (placeholders like
        <step 1 output> are replaced with prior outputs).
        """
        trace_steps: list[dict] = []
        prior_assistant = ""
        prior_outputs: dict[int, str] = {}

        # Normalize inputs
        initial_system = initial_system or ""
        user_prompt = user_prompt or ""

        def _inject_outputs(text: str) -> str:
            if not text:
                return text
            def _repl(m):
                idx = int(m.group(1))
                return prior_outputs.get(idx, "")
            return re.sub(r"<step\s*(\d+)\s*output>", _repl, text, flags=re.IGNORECASE)

        for i in range(1, self.steps + 1):
            key = f"step_{i}"
            cfg = (self.step_prompts or {}).get(key, {}) or {}

            system = (cfg.get("system") if isinstance(cfg, dict) else None) or initial_system
            user = (cfg.get("user") if isinstance(cfg, dict) else None) or user_prompt

            # retrieval hook
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

            injected_user = _inject_outputs(user)
            injected_system = _inject_outputs(system)
            composed_user = "\n\n".join([p for p in [contexts_txt, injected_user] if p])

            if self.dry_run:
                output = "[dry_run]"
            else:
                output = generator_chat_callable(injected_system, composed_user)

            prior_outputs[i] = output.strip()
            prior_assistant = output.strip()

            trace_steps.append({
                "name": key,
                "step": i,
                "system": system,
                "user": composed_user,
                "output": output,
                "used_context_ids": used_ids,
            })

        return {"final": prior_assistant, "steps": trace_steps}

    def run_user_stepwise(
        self,
        *,
        initial_system: str | None,
        user_prompt: str,
        generator_chat_callable,
        retrieve=None,
    ) -> dict:
        """Run CoT as a sequence of chat calls with user message varying per step.

        This implementation uses an explicit messages list (role-based chat history)
        for clearer role semantics. It still preserves prior assistant outputs
        between steps by appending assistant messages to the messages list. It
        also preserves retrieval hooks, dry_run behavior and per-step tracing.
        The generator callable may accept either a single `messages` argument or
        the legacy `(system, user_text)` pair; we try messages first and fall
        back to the pair-callable on TypeError.
        """
        trace_steps: list[dict] = []

        # Normalize inputs
        initial_system = initial_system or ""
        user_prompt = user_prompt or ""

        # Messages list used for chat-style generation. Start with initial system if present.
        messages: list[dict] = []
        if initial_system:
            messages.append({"role": "system", "text": initial_system})

        # Keep a list of prior assistant messages to replay into fresh chats if system changes
        prior_assistant_msgs: list[str] = []

        for i in range(1, self.steps + 1):
            key = f"step_{i}"
            cfg = (self.step_prompts or {}).get(key, {}) or {}

            system = (cfg.get("system") if isinstance(cfg, dict) else None) or initial_system
            user = (cfg.get("user") if isinstance(cfg, dict) else None) or user_prompt

            # retrieval hook
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

            # Compose the user message for this step
            composed_user = "\n\n".join([p for p in [contexts_txt, user] if p])

            # If the system for this step differs from the current messages' system,
            # start a fresh messages list for a fresh chat but replay prior assistant
            # messages so the model can condition on earlier outputs.
            need_fresh_chat = False
            if system:
                # Determine current system in messages (if any)
                current_system = None
                for m in messages:
                    if m.get("role") == "system":
                        current_system = m.get("text")
                        break
                if current_system is None or (system.strip() != current_system.strip()):
                    need_fresh_chat = True

            if need_fresh_chat:
                messages = [{"role": "system", "text": system}] if system else []
                for msg in prior_assistant_msgs:
                    messages.append({"role": "assistant", "text": msg})

            # Append the current user message and call the generator
            messages.append({"role": "user", "text": composed_user})

            if self.dry_run:
                output = "[dry_run]"
            else:
                # Try messages-list callable first; fallback to (system, user) pair
                try:
                    output = generator_chat_callable(messages)
                except TypeError:
                    # Legacy callable expecting (system, user_text)
                    try:
                        output = generator_chat_callable(system, composed_user)
                    except Exception as e:
                        # Re-raise with context
                        raise

            # Append assistant output to messages and prior assistant list
            messages.append({"role": "assistant", "text": output})
            prior_assistant_msgs.append(output.strip())

            trace_steps.append({
                "name": key,
                "step": i,
                "system": system,
                "user": composed_user,
                "output": output,
                "used_context_ids": used_ids,
            })

        # Final is the concatenation of assistant messages (preserve original behavior)
        final_text = "\n\n".join(prior_assistant_msgs).strip()
        return {"final": final_text, "steps": trace_steps}

    def run_all_in_one(
        self,
        *,
        initial_system: str,
        user_prompt: str,
        generator_chat_callable,
    ) -> dict:
        """Run CoT as a single chat call with all steps contained in one user prompt.
        """
        trace_steps: list[dict] = []
        used_ids_map: dict[int, list[str]] = {}

        output = generator_chat_callable(initial_system, user_prompt)

        key = f"step_1"
        cfg = (self.step_prompts or {}).get(key, {}) or {}
        system = (cfg.get("system") if isinstance(cfg, dict) else None) or initial_system
        user = (cfg.get("user") if isinstance(cfg, dict) else None) or user_prompt
        trace_steps.append({
            "name": key,
            "step": 1,
            "system": system,
            "user": user,
            "output": output,
            "used_context_ids": used_ids_map.get(1, []),
        })

        return {"final": output.strip(), "steps": trace_steps}

    def run_chat_sequence(
        self,
        *,
        initial_system: str | None,
        user_prompt: str,
        generator_chat_callable,
        retrieve=None,
        strategy: str | None = None,
    ) -> dict:
        """Run CoT as a sequence of chat calls.

        Strategies:
        User-stepwise: preserve prior assistant outputs between steps by appending them to the user prompt.
        System-stepwise: fresh chat per step, but inject prior step outputs into the user prompt
        by replacing placeholders like <step 1 output>, <step 2 output>, etc.
        This keeps each step a fresh chat w.r.t. system message while making prior results
        available as variables for formatting.
        All-in-one: single chat call with all steps concatenated into one user prompt.

        Default strategy is "all_in_one" if not specified or unrecognized.
        """
        if strategy == "system_stepwise":
            return self.run_system_stepwise(
                initial_system=initial_system,
                user_prompt=user_prompt,
                generator_chat_callable=generator_chat_callable,
                retrieve=retrieve,
            )
        elif strategy == "user_stepwise":
            return self.run_user_stepwise(
                initial_system=initial_system,
                user_prompt=user_prompt,
                generator_chat_callable=generator_chat_callable,
                retrieve=retrieve,
            )
        elif strategy != "all_in_one":
            logger.warning(f"Unknown or missing strategy '{strategy}', defaulting to 'all_in_one'")
        return self.run_all_in_one(
            initial_system=initial_system,
            user_prompt=user_prompt,
            generator_chat_callable=generator_chat_callable,
        )
        