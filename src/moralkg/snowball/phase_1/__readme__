Purpose
- Phase 1: fit hyperparameters for AM pipelines against human annotations using a composite loss.

Key components

- evals/ (src.moralkg.snowball.phase_1.evals)
  - Purpose: run evaluation loops that compare model-generated structured outputs against gold annotations and produce scalar metrics used by HPO or analysis.
  - Main responsibilities:
    - Evaluator (class / module): load a model (base, with adapters, or a merged checkpoint), construct prompts, call the model (optionally via RAG or CoT subroutines), parse model JSON output into the canonical internal representation, and compute metrics.
      - Example public API: Evaluator(model_spec, dataset, prompt_builder, parser, metrics) -> .run(split)-> Iterator[EvalResult]
      - Inputs: model spec or registry key, dataset iterator, prompt template id, decoding/config overrides.
      - Outputs: per-example EvalResult {id, score_map, parsed_pred, raw_output, cost_meta} and aggregated metrics.
      - Error modes: parsing failures (return structured parse error in EvalResult), model timeouts (propagate or wrap as transient errors), I/O failures (raise).
    - Metrics: modular metric implementations under `evals/metrics*`.
      - Implementations: fuzzy span F1, relation macro‑F1, count RMSE, graph-edit-distance (GED), and a combined weighted cost. Each metric should accept (gold, pred) and return numeric value + optional diagnostic fields.
      - Example function signature: compute_metric(gold: GoldStruct, pred: PredStruct, cfg: dict) -> MetricResult
    - Dataset adapters: uniform iterator that yields (id, text, gold_struct, meta).
      - Purpose: isolate file-format parsing and enable reuse across datasets.
    - Prompts: lightweight prompt-builder that maps example+config -> prompt string(s). Should live in `evals/prompts` or `../prompts` and be configurable for CoT/RAG variants.
    - Cost: WeightedCost (small utility to combine metrics according to `config.yaml` weights) used by HPO objective.

- prompts/ (src.moralkg.snowball.phase_1.prompts)
  - Purpose: store prompt templates, example shots, and rendering helpers.
  - Layout: each engine or template set gets a folder (e.g. `meta_llama_3.1_8B/`) containing plain text templates, placeholder variables, and small example files.
  - Responsibilities:
    - Provide `load_template(name)` and `render_template(template, context)` helpers.
    - Keep templates data-only; heavy formatting or I/O should be triggered explicitly by functions, not on import.
    - Support variants: zero-shot, k-shot, CoT, RAG-inserted context.

- models/ (src.moralkg.snowball.phase_1.models)
  - Purpose: model registry and light adapters that encapsulate how to instantiate LLMs or adapter-wrapped models.
  - Responsibilities:
    - `registry.py` should expose a `ModelRegistry` able to return a callable `ModelClient` given a model key and config. ModelClient should present a small, stable interface, for example:
      - ModelClient.generate(prompts: Sequence[str], max_tokens: int, **kwargs) -> Sequence[ModelResponse]

- pipelines/ (src.moralkg.snowball.phase_1.pipelines)
  - Purpose: high-level orchestration of end-to-end flows: batching inputs, calling models, checkpointing results, and triggering evaluation.
  - Responsibilities:
    - `orchestrator.py` contains a `PipelineOrchestrator` that composes pieces from `batch/`, `models/`, and `io/` to run a complete job.
    - Provide functions to run a single job (sync/async), resume from checkpoints, and emit structured outputs for downstream HPO.

- batch/ (src.moralkg.snowball.phase_1.batch)
  - Purpose: utilities to group examples into requests appropriate for the target LLM (token-budget aware batching, simple list-chunking, or orchestration across workers).
  - Responsibilities:
    - Provide `create_batches(examples, max_tokens, tokenizer)` and `flatten_results(batched_results)` helpers.

- io/ (src.moralkg.snowball.phase_1.io)
  - Purpose: checkpointing and artifact I/O used by evaluation and pipelines.
  - Responsibilities:
    - Save/load checkpoints (per-job, per-batch) and provide atomic write helpers.
    - Provide `list_checkpoints(job_id)` and `load_checkpoint(path)` utilities. Keep any cloud-storage logic behind small adapters so unit tests can use local FS.

- utility scripts
  - `generate_llm_prompts.py`: a simple CLI that renders templates in `prompts/` to produce the actual prompt files used by offline LLM runs or auditing. Expected inputs: template id, dataset split, variant flags (CoT/RAG). Expected outputs: text files (one prompt per example) and a small manifest JSON.
  - `hpo.py` (placeholder): intended to wrap an HPO loop (Optuna or similar). The eventual HPO should provide:
    - An objective function that accepts a hyperparameter dict and returns a scalar (lower is better) computed from the `WeightedCost` over a held-out eval split.
    - Clear separation between trial construction and model instantiation so that expensive model loading is done once per trial process.


Config usage
- snowball.phase_1.eval.fuzzy_thr: fuzzy matching threshold.
- snowball.phase_1.eval.loss.w: weight vector [span, relation, count, ged].
- snowball.phase_1.hparams.end2end.*: decoding, rag, cot defaults (if wired by caller).

Status / TODO
- HPO loop is not yet implemented.
- For generate_llm_prompts.py, implement prompt generation for CoT, RAG, and CoT + RAG. This should integrate with the CoT and RAG orchestration in src.moralkg.argmining.models.models .

Notes
- For more details on each sub-package, see the `__readme__` files in the following subdirectories: `batch/`, `io/`, `models/`, `pipelines/`, `prompts/`.

