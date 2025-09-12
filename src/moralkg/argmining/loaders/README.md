Purpose
- Load PhilPapers metadata, associated paper text, and workshop annotations for downstream AM.

Exposed API
- Dataset: single entrypoint
  - attributes: metadata (dict[id  fields]), annotations (dict[id  parsed map])
  - methods: get_paper(id) -> str; iter_split(split="validation"|"holdout") -> iterator of (id, text, annotation?) [planned]

Config usage
- paths.philpapers.metadata: file or directory; if dir, load newest file. Builds `metadata` keyed by paper id.
- paths.philpapers.docling.cleaned: directory with `<id>.md` or `<id>.txt` used by get_paper.
- workshop.annotations.use: "large" | "both"  paths.workshop.annotations.{large_maps,small_maps}.
- snowball.annotations.holdout: float ratio for stratified split (by category).
