"""ADUR → ARE (Pipeline 2) and ADUR → Major ADU → ARE (Pipeline 3) runners.

These runners are intentionally lightweight: they validate model refs,
use the `registry` factories lazily to construct pipelines, run the
file-mode generation helper, normalize outputs via adapters, and persist
artifacts using the checkpoints API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Tuple
import logging


def _validate_refs(registry, refs: List[Any]) -> Tuple[bool, List[dict]]:
    details = []
    ok = True
    for r in refs:
        try:
            r_ok, r_det = registry.validate_pipeline(r)
        except Exception as e:
            r_ok = False
            r_det = {"error": str(e)}
        details.append(r_det)
        if not r_ok:
            ok = False
    return ok, details


def _select_major_adus(adus: list[dict], method: str = "centroid") -> list[dict]:
    """Cheap deterministic major-ADU selector used in Pipeline 3.

    Current heuristic: pick the longest ADU text labelled as a claim.
    This is intentionally simple and deterministic; it only annotates the
    normalized ADU dicts with a 'major' boolean flag and returns the
    modified list.
    """
    if not adus:
        return adus
    # filter claims
    claims = [(i, a) for i, a in enumerate(adus) if str(a.get("type") or a.get("label") or "").lower().startswith("claim")]
    if not claims:
        # fallback: mark the first ADU
        adus[0]["major"] = True
        return adus

    # pick longest text among claims
    best_idx = None
    best_len = -1
    for i, a in claims:
        t = str(a.get("text") or "")
        if len(t) > best_len:
            best_len = len(t)
            best_idx = i

    if best_idx is not None:
        adus[best_idx]["major"] = True
    return adus


def run_pipeline2(
    input_files: Iterable[Path] | Iterable[str],
    outdir: Path | str,
    adur_model_ref: Any = None,
    are_model_ref: Any = None,
    use_adur_model_2: bool = False,
    use_are_model_2: bool = False,
    dry_run: bool = False,
):
    """Run Pipeline 2 (ADUR -> ARE -> final map).

    Steps:
    - validate model refs (dry-run returns validation details)
    - instantiate ADUR pipeline and run file-mode to write normalized ADUR outputs
    - instantiate ARE pipeline (passing adur_model_ref when available) and run file-mode to produce final maps

    Returns the outdir Path on success or validation details on dry-run.
    """
    logger = logging.getLogger(__name__)
    from moralkg.snowball.phase_1.models import registry
    from moralkg.snowball.phase_1.models import adapters
    from moralkg.snowball.phase_1.batch.generation import run_file_mode
    from moralkg.snowball.phase_1.io import checkpoints

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validation step
    ok, details = _validate_refs(registry, [adur_model_ref, are_model_ref])
    if dry_run:
        return {"ok": ok, "details": details}
    if not ok:
        logger.error("Validation failed for model refs: %s", details)
        raise RuntimeError(f"Model validation failed: {details}")

    # Run ADUR in file-mode and persist normalized outputs
    adur_outdir = outdir / "adur"
    adur = registry.get_adur_pipeline(adur_model_ref, use_model_2=use_adur_model_2)
    logger.info("Running ADUR pipeline over %d files, outputs -> %s", len(list(input_files)), adur_outdir)
    run_file_mode(adur, input_files, adur_outdir, adapters.normalize_adur_output, prefix="adur")

    # Run ARE (it may run its own ADUR internally or use adur_model_ref)
    are_outdir = outdir / "are"
    are = registry.get_are_pipeline(are_model_ref, adur_model_ref=adur_model_ref, use_model_2=use_are_model_2, use_adur_model_2=use_adur_model_2)
    logger.info("Running ARE pipeline over %d files, outputs -> %s", len(list(input_files)), are_outdir)
    run_file_mode(are, input_files, are_outdir, adapters.normalize_are_output, prefix="are")

    return outdir


def run_pipeline3(
    input_files: Iterable[Path] | Iterable[str],
    outdir: Path | str,
    adur_model_ref: Any = None,
    are_model_ref: Any = None,
    use_adur_model_2: bool = False,
    use_are_model_2: bool = False,
    major_method: str = "centroid",
    dry_run: bool = False,
):
    """Run Pipeline 3 (ADUR -> Major ADU extraction -> ARE -> final map).

    This is the same as Pipeline 2 but applies a deterministic major-ADU
    selector after normalization and before invoking ARE. The selector adds
    a 'major' boolean flag on selected ADUs and persists the updated
    normalized ADU artifact.
    """
    logger = logging.getLogger(__name__)
    from moralkg.snowball.phase_1.models import registry
    from moralkg.snowball.phase_1.models import adapters
    from moralkg.snowball.phase_1.batch.generation import run_file_mode
    from moralkg.snowball.phase_1.io import checkpoints
    import json

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validation step
    ok, details = _validate_refs(registry, [adur_model_ref, are_model_ref])
    if dry_run:
        return {"ok": ok, "details": details}
    if not ok:
        logger.error("Validation failed for model refs: %s", details)
        raise RuntimeError(f"Model validation failed: {details}")

    # Run ADUR and then annotate major ADUs
    adur_outdir = outdir / "adur"
    adur = registry.get_adur_pipeline(adur_model_ref, use_model_2=use_adur_model_2)
    run_file_mode(adur, input_files, adur_outdir, adapters.normalize_adur_output, prefix="adur")

    # Annotate normalized ADU outputs with major flags
    for p in sorted(adur_outdir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            adus = data.get("adus") or []
            adus = _select_major_adus(adus, method=major_method)
            data["adus"] = adus
            # overwrite file atomically via checkpoints.save_individual
            checkpoints.save_individual(data, adur_outdir, p.name)
        except Exception as e:
            logger.warning("Failed to annotate major ADUs for %s: %s", p, e)
            continue

    # Run ARE using the (annotated) ADU checkpoint as upstream artifact
    are_outdir = outdir / "are"
    are = registry.get_are_pipeline(are_model_ref, adur_model_ref=adur_model_ref, use_model_2=use_are_model_2, use_adur_model_2=use_adur_model_2)
    run_file_mode(are, input_files, are_outdir, adapters.normalize_are_output, prefix="are")

    return outdir
