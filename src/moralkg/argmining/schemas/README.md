Purpose
- Single source of truth for AM data structures and model I/O schema.

Exposed types
- ADU, ADUType, SpanPosition
- Relation, RelationType
- ArgumentMap (helpers: get_adu_by_id, get_relations_for_adu, to_dict, map_statistics)

JSON schema
- argmining.json: model response structure with ADUs object and relations list.
