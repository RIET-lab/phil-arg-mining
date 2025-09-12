"""Model registry helpers for Phase 1.

This module provides a thin factory to construct the real End2End pipeline.
It intentionally does not provide a mock fallback — the pipeline code should
raise if required dependencies or model files are missing.
"""
from __future__ import annotations

from typing import Any, Tuple, Dict
import logging
from pathlib import Path

try:
  # Prefer not to force heavy imports here; these are used in factory calls below
  pass
except Exception:
  pass


def create_end2end(real_kwargs: dict | None = None) -> Any:
    """Instantiate and return the real End2End model.

    Args:
      real_kwargs: forwarded to the End2End constructor.

    Raises:
      Any exception raised by the End2End constructor (no fallback performed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating End2End pipeline with kwargs: %s", real_kwargs)
    try:
        from moralkg.argmining.models.models import End2End
        inst = End2End(**(real_kwargs or {}))
        logger.info("End2End pipeline created successfully")
        return inst
    except Exception as e:
        logger.error("Failed to create End2End pipeline: %s", e)
        raise


def _build_remediation_suggestions(exc: Exception) -> str:
  """Return a short remediation message depending on the exception type/content.

  This includes suggested pip installs and environment actions to fix common issues
  encountered when loading SAM/SAM-like models via `pytorch_ie.AutoPipeline`.
  """
  msg_lines = []
  txt = str(exc)
  if "pytorch_ie" in txt or "AutoPipeline" in txt:
    msg_lines.append("Required package 'pytorch-ie' (and its optional 'pie_modules') may be missing.")
    msg_lines.append("Try: pip install 'pytorch-ie' 'pie_modules' or follow the project README for installing pie modules.")
  if "taskmodule_type" in txt or "taskmodule_config" in txt or "taskmodule" in txt:
    msg_lines.append(
      "Model snapshot appears to be missing taskmodule registration (KeyError: taskmodule_type or missing taskmodule_config.json)."
    )
    msg_lines.append(
      "Ensure you downloaded a validated SAM model snapshot that includes 'taskmodule_config.json', and that any required pie_modules are importable in your environment."
    )
    msg_lines.append(
      "If you have pie_modules locally, add it to PYTHONPATH before running: export PYTHONPATH=/path/to/pie_modules:$PYTHONPATH"
    )
  if not msg_lines:
    msg_lines.append("See the model load exception above for details; verify model path/config and required packages.")
  return "\n".join(msg_lines)


def validate_pipeline(model_ref: Any) -> Tuple[bool, Dict[str, Any]]:
  """Validate a model reference without downloading remote HF snapshots.

  Args:
    model_ref: dict|str|Path. If dict and contains 'dir' this will inspect the directory.

  Returns:
    (ok, details) where ok is bool and details contains fields like 'reason', 'path', 'found_files'.

  Note: we deliberately avoid downloading from HF in validation; use the full factory to attempt downloads.
  """
  details: Dict[str, Any] = {"model_ref": model_ref}
  p = None
  # Accept dicts like {"dir": "/abs/path"} or {"hf": "owner/repo"}
  if isinstance(model_ref, dict):
    if "dir" in model_ref and model_ref["dir"]:
      p = Path(model_ref["dir"]).expanduser().resolve()
      details["model_type"] = "local"
    elif "hf" in model_ref:
      details["model_type"] = "hf"
      details["reason"] = "HF repo reference provided; no local dir to validate. Run a dry-run after downloading or validate the downloaded snapshot."
      return False, details
    else:
      details["reason"] = "Dict model_ref provided but lacks 'dir' or 'hf' keys."
      return False, details
  else:
    # str or Path: treat as path
    try:
      p = Path(str(model_ref)).expanduser().resolve()
      details["model_type"] = "local"
    except Exception:
      details["reason"] = "Unable to interpret model_ref as a local path"
      return False, details

  if p is None:
    details["reason"] = "No path resolved for validation"
    return False, details

  details["path"] = str(p)
  if not p.exists():
    details["reason"] = f"Local path does not exist: {p}"
    return False, details
  if not p.is_dir():
    details["reason"] = f"Provided path is not a directory: {p}"
    return False, details

  # Heuristics: look for taskmodule_config.json, config.json, pytorch model files
  found = []
  for name in ("taskmodule_config.json", "taskmodule_config.yaml", "config.json", "pytorch_model.bin", "pytorch_model.pt"):
    for f in p.rglob(name):
      found.append(str(f))
  # Also detect large.bin variants
  for f in p.rglob("*.bin"):
    found.append(str(f))

  details["found_files"] = sorted(set(found))
  if not details["found_files"]:
    details["reason"] = "Model directory exists but no expected model config or binary files were found."
    return False, details

  details["reason"] = "Looks like a valid model snapshot (heuristic checks passed)."
  return True, details


def _raise_with_diagnostics(exc: Exception, model_ref: Any = None) -> None:
  logger = logging.getLogger(__name__)
  diag = _build_remediation_suggestions(exc)
  details = None
  try:
    ok, details = validate_pipeline(model_ref) if model_ref is not None else (False, None)
  except Exception:
    details = None

  msg_lines = ["Failed to create pipeline:\n", str(exc), "\n"]
  if details:
    msg_lines.append("Validation details:\n")
    for k, v in details.items():
      msg_lines.append(f"  {k}: {v}\n")
  msg_lines.append("Remediation suggestions:\n")
  msg_lines.append(diag)
  full = "\n".join(msg_lines)
  logger.error(full)
  raise RuntimeError(full) from exc


def get_adur_pipeline(model_ref: Any | None = None, *, use_model_2: bool = False, **kwargs) -> Any:
  """Factory for ADUR pipeline. Fails loudly with diagnostics when load fails.

  Args:
    model_ref: model identifier (dict with 'dir' or 'hf', or path string). If None, ADUR will read from Config.
    use_model_2: pick alternate model in config when model_ref is None.
  """
  logger = logging.getLogger(__name__)
  logger.info("Creating ADUR pipeline (use_model_2=%s) with model_ref=%s", use_model_2, model_ref)
  try:
    from moralkg.argmining.models.models import ADUR  # local heavy import

    # Ensure pie_modules (taskmodule implementations) are importable so pytorch_ie
    # can find registered TaskModule classes referenced by model snapshots.
    # This is a best-effort import sequence; failures are ignored and will be
    # surfaced when AutoPipeline.from_pretrained is called.
    try:
      import importlib
      candidates = [
        "pie_modules",
        "pie_modules.sam",
        "pie_modules.sam_adur",
        "pie_modules.sam_are",
        "pie_modules.pytorch_ie",
      ]
      for name in candidates:
        try:
          importlib.import_module(name)
          logger.info("Imported pie module: %s", name)
        except Exception:
          logger.debug("Could not import pie module: %s", name)
      # Additionally, try importing the pie_modules.taskmodules package and its children
      try:
        tm_pkg = importlib.import_module('pie_modules.taskmodules')
        logger.info('Imported pie_modules.taskmodules')
        # Import all submodules under pie_modules.taskmodules to trigger registration
        if hasattr(tm_pkg, '__path__'):
          import pkgutil
          for finder, subname, ispkg in pkgutil.iter_modules(tm_pkg.__path__):
            fullname = f"{tm_pkg.__name__}.{subname}"
            try:
              importlib.import_module(fullname)
              logger.info('Imported pie taskmodule submodule: %s', fullname)
            except Exception:
              logger.debug('Could not import pie taskmodule submodule: %s', fullname)
      except Exception:
        logger.debug('pie_modules.taskmodules not available for import')
    except Exception:
      # Non-fatal; the main ADUR instantiation will reveal missing registrations
      logger.debug("Pie modules import attempt failed")

    if model_ref is None:
      inst = ADUR(use_model_2=use_model_2, **kwargs)
    else:
      inst = ADUR(model=model_ref, use_model_2=use_model_2, **kwargs)
    logger.info("ADUR pipeline created successfully")
    return inst
  except Exception as exc:
    _raise_with_diagnostics(exc, model_ref)


def get_are_pipeline(model_ref: Any | None = None, *, adur_model_ref: Any | None = None, use_model_2: bool = False, use_adur_model_2: bool = False, **kwargs) -> Any:
  """Factory for ARE pipeline. Fails loudly with diagnostics when load fails.

  Args:
    model_ref: ARE model identifier (dict with 'dir' or 'hf', or path string). If None, ARE will read from Config.
    adur_model_ref: optional ADUR model ref to pass to ARE for preprocessing.
  """
  logger = logging.getLogger(__name__)
  logger.info("Creating ARE pipeline (use_model_2=%s) with model_ref=%s and adur_model_ref=%s", use_model_2, model_ref, adur_model_ref)
  try:
    from moralkg.argmining.models.models import ARE

    if model_ref is None and adur_model_ref is None:
      inst = ARE(use_model_2=use_model_2, use_adur_model_2=use_adur_model_2, **kwargs)
    else:
      inst = ARE(model=model_ref, adur_model=adur_model_ref, use_model_2=use_model_2, use_adur_model_2=use_adur_model_2, **kwargs)
    logger.info("ARE pipeline created successfully")
    return inst
  except Exception as exc:
    _raise_with_diagnostics(exc, model_ref or adur_model_ref)
