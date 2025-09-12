Purpose
- Convert model outputs (JSON or text with embedded JSON) into `schemas.ArgumentMap` objects.

Exposed API
- Parser(schema_path: str | None = None)
  - parse_json_file(path, source_text: str | None) -> ArgumentMap
  - parse_string(json_str, map_id: str | None, source_text: str | None) -> ArgumentMap
  - parse_dict(model_output: dict, map_id: str, source_text: str | None) -> ArgumentMap
  - parse_model_response(response_text: str, map_id: str | None, source_text: str | None) -> ArgumentMap
  - extract_from_text(text: str, allow_partial=True) -> dict
  - save_to_json(argument_map, output_path)

Schema + Types
- Uses `src/moralkg/argmining/schemas/argmining.json` (if `schema_path` provided) as guidance for expected fields.
- Produces `ArgumentMap` with `ADU` and `Relation` instances.

Config usage
- argmining.schema (optional): default schema path if you want to inject schema into prompts or validate outputs.

Notes
- Robust JSON extraction from freeform text is included; validation is minimal unless you add explicit JSON schema checks.
