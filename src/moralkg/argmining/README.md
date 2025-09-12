Argument mining APIs and types. Submodules and their roles:

- loaders
  - Dataset: provides paper metadata/content and optional human annotations; supports splits.
  - Uses config: paths.philpapers.metadata, paths.philpapers.docling.cleaned, paths.workshop.annotations.{large_maps,small_maps}, workshop.annotations.use, snowball.annotations.holdout.

- models
  - End2End: LLM prompting (optional RAG, optional CoT). key method: generate(system_prompt, user_prompt, prompt_files) -> {text, trace}.
  - ADUR: span extraction (SAM/pytorch_ie). key method: generate(input_file) -> {adus, statistics}.
  - ARE: relation extraction on top of predicted spans. key method: generate(input_file) -> {adus, relations, statistics}.
  - Helpers: RAG (embeddings + FAISS), CoT (multi‑step orchestration), generator (HF+PEFT loaders and decode helpers).
  - Uses config: paths.models.end2end.{base,finetune,embedder}, paths.models.{adur,are}.model_* entries (dir/hf fallbacks).

- parsers
  - Parser: parse_json_file/parse_string/parse_dict/parse_model_response → ArgumentMap; save_to_json; extract_from_text.
  - Optionally validate/shape against argmining.schema (pass schema_path or read from config argmining.schema).

- schemas
  - Types: ArgumentMap, ADU, Relation (+ enums ADUType, RelationType in code). JSON schema: schemas/argmining.json for model I/O.

How modules interact
- loaders supplies paper text + (optional) gold maps → models produce raw JSON (or spans/relations) → parsers normalize to ArgumentMap using schemas.

Not yet implemented
- Some glue (e.g., end‑to‑end pipeline wiring) and multi‑GPU scheduling are marked TODO in code.
- loaders split utilities (iter_split) and enhanced indexing helpers are pending.
