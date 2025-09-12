"""ADUR → ARE (Pipeline 2) and ADUR → Major ADU → ARE (Pipeline 3) runners.

These runners are intentionally lightweight: they validate model refs,
use the `registry` factories lazily to construct pipelines, run the
file-mode generation helper, normalize outputs via adapters, and persist
artifacts using the checkpoints API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from moralkg.snowball.phase_1.models import registry
from moralkg.snowball.phase_1.models import adapters
from moralkg.snowball.phase_1.batch.generation import run_from_texts
from moralkg.snowball.phase_1.io import checkpoints
from moralkg.config import Config
from moralkg.logging import get_logger
import json

def _select_major_adus(adus: list[dict], method: str = "centroid") -> list[dict]:
    """TODO: Implement major ADU selection strategies.

    Current simple heuristic: pick the longest Claim-type ADU and mark it as major.
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
    adur_outdir: Path | str,
    are_outdir: Path | str,
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
    cfg = Config.load()
    logger = get_logger(__name__)

    adur_outdir = Path(adur_outdir)
    adur_outdir.mkdir(parents=True, exist_ok=True)
    are_outdir = Path(are_outdir)
    are_outdir.mkdir(parents=True, exist_ok=True)

    adur_model_ref = cfg.get(f"paths.models.adur.{adur_model_ref}.dir", None) if isinstance(adur_model_ref, str) else None
    are_model_ref = cfg.get(f"paths.models.are.{are_model_ref}.dir", None) if isinstance(are_model_ref, str) else None

    # Validation step
    for model_ref in [adur_model_ref, are_model_ref]:
        if model_ref is not None:
            p = Path(model_ref)
            if not p.exists() or not p.is_dir():
                logger.error("Model path does not exist or is not a directory: %s", p)
                raise RuntimeError(f"Invalid model path: {p}")
            ok, details = registry.validate_instance(p)
            if not ok:
                logger.error("Validation failed for model refs: %s", details)
                raise RuntimeError(f"Model validation failed: {details}")

    # Run ADUR in file-mode and persist normalized outputs
    adur = registry.get_adur_instance(adur_model_ref, use_model_2=use_adur_model_2)
    # Prepare mapping of id->text by reading input files
    texts = {}
    for p in input_files:
        p_path = Path(p)
        texts[p_path.stem] = p_path.read_text(encoding="utf-8", errors="ignore")
    logger.info("Running ADUR pipeline over %d items, outputs -> %s", len(texts), adur_outdir)
    run_from_texts(adur, texts, adur_outdir, adapters.normalize_adur_output, prefix="adur")

    # Run ARE (it may run its own ADUR internally or use adur_model_ref)
    are = registry.get_are_instance(are_model_ref, adur_model_ref=adur_model_ref, use_model_2=use_are_model_2, use_adur_model_2=use_adur_model_2)
    # Reuse same texts mapping for ARE
    logger.info("Running ARE pipeline over %d items, outputs -> %s", len(texts), are_outdir)
    run_from_texts(are, texts, are_outdir, adapters.normalize_are_output, prefix="are")

    return are_outdir


def run_pipeline3(
    input_files: Iterable[Path] | Iterable[str],
    adur_outdir: Path | str,
    are_outdir: Path | str,
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
    cfg = Config.load()
    logger = get_logger(__name__)

    adur_outdir = Path(adur_outdir)
    adur_outdir.mkdir(parents=True, exist_ok=True)
    are_outdir = Path(are_outdir)
    are_outdir.mkdir(parents=True, exist_ok=True)

    adur_model_ref = cfg.get(f"paths.models.adur.{adur_model_ref}.dir", None) if isinstance(adur_model_ref, str) else None
    are_model_ref = cfg.get(f"paths.models.are.{are_model_ref}.dir", None) if isinstance(are_model_ref, str) else None

    # Validation step
    for model_ref in [adur_model_ref, are_model_ref]:
        if model_ref is not None:
            p = Path(model_ref)
            if not p.exists() or not p.is_dir():
                logger.error("Model path does not exist or is not a directory: %s", p)
                raise RuntimeError(f"Invalid model path: {p}")
            ok, details = registry.validate_instance(p)
            if not ok:
                logger.error("Validation failed for model refs: %s", details)
                raise RuntimeError(f"Model validation failed: {details}")

    # TODO: Implement major ADU selection (according to config, can either be centroid or pairwise)

    # Run ADUR and then annotate major ADUs
    adur = registry.get_adur_instance(adur_model_ref, use_model_2=use_adur_model_2)
    texts = {}
    for p in input_files:
        p_path = Path(p)
        texts[p_path.stem] = p_path.read_text(encoding="utf-8", errors="ignore")
    run_from_texts(adur, texts, adur_outdir, adapters.normalize_adur_output, prefix="adur")

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
    are = registry.get_are_instance(are_model_ref, adur_model_ref=adur_model_ref, use_model_2=use_are_model_2, use_adur_model_2=use_adur_model_2)
    run_from_texts(are, texts, are_outdir, adapters.normalize_are_output, prefix="are")

    return are_outdir
