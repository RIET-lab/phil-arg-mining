from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import os
import time
import rootutils

from moralkg.config import Config
from moralkg.logging import get_logger

_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


@dataclass
class GenerationConfig:
    sample_csv: Optional[str] = None
    papers_dir: Optional[str] = None
    labels_dir: Optional[str] = None
    arguments_dir: Optional[str] = None
    prefer_markdown: bool = False
    # Processing controls
    number_limit: Optional[int] = None
    skip_not_found: bool = True
    # Existing file behavior
    skip_existing: bool = False
    skip_existing_labels: bool = False
    skip_existing_arguments: bool = False
    overwrite_labels: bool = False
    overwrite_arguments: bool = False
    append_labels: bool = True
    append_arguments: bool = True
    # API controls
    retries: int = 3
    min_temperature: Optional[float] = None
    max_temperature: Optional[float] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    num_combinations: int = 2


def _default_paths(cfg: Config) -> Dict[str, str]:
    # Workshop defaults
    sample_dir = cfg.get("workshop.sample.dir")
    annotations = cfg.get("workshop.annotations", {}) or {}
    use = annotations.get("use", "large")
    large_dir = annotations.get("large_maps", {}).get("dir")
    small_dir = annotations.get("small_maps", {}).get("dir")
    # Docling path
    docling_raw = cfg.get("philpapers.papers.docling.raw.dir") or cfg.get("philpapers.papers.docling.cleaned.dir")

    defaults = {
        "sample_csv": str(Path(str(sample_dir)) / "sample.csv") if sample_dir else None,
        "papers_dir": str(docling_raw) if docling_raw else None,
        "labels_dir": str(Path(large_dir if use in {"large", "both"} else small_dir) or "datasets/processed/argument_mining/workshop_annotations/large"),
        "arguments_dir": str(Path(annotations.get("large_maps", {}).get("dir", "datasets/processed/argument_mining/workshop_annotations/large")) / "arguments"),
    }
    return defaults


def _load_identifiers(sample_csv_path: Path) -> List[str]:
    import csv

    if not sample_csv_path.exists():
        raise FileNotFoundError(f"Sample CSV not found: {sample_csv_path}")
    with sample_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "identifier" not in reader.fieldnames:
            raise ValueError("sample.csv must contain an 'identifier' column")
        return [row["identifier"].strip() for row in reader if row.get("identifier")]  # type: ignore


def generate_annotations_for_sample(config: Optional[GenerationConfig] = None) -> Dict[str, Path]:
    """
    Generate argumentative annotations for each paper in the sample, matching
    the archive scripts' behavior (OpenAI + Anthropic, optional hyperparam search,
    skip/overwrite/append flags, and path defaults via Config).
    """
    # Lazy imports of API clients to avoid hard deps if unused
    try:
        import openai  # type: ignore
    except Exception:  # pragma: no cover
        openai = None  # type: ignore
    try:
        import anthropic  # type: ignore
    except Exception:  # pragma: no cover
        anthropic = None  # type: ignore

    def load_prompt_template() -> str:
        prompt_path = Path(__file__).parent / "archive" / "labels" / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def load_paper_content(papers_dir: Path, identifier: str, prefer_md: bool = False) -> Optional[str]:
        exts = [".md", ".txt"] if prefer_md else [".txt", ".md"]
        for ext in exts:
            p = papers_dir / f"{identifier}{ext}"
            if p.exists():
                return p.read_text(encoding="utf-8", errors="ignore")
        return None

    def generate_hyperparameters(min_temp: float, max_temp: float, min_tokens: int, max_tokens: int, num_combinations: int) -> List[Dict[str, float]]:
        import random
        combos: List[Dict[str, float]] = []
        for _ in range(num_combinations):
            t = random.uniform(min_temp, max_temp)
            mt = random.randint(int(min_tokens), int(max_tokens))
            combos.append({"temperature": t, "max_tokens": mt})
        return combos

    def validate_adus_in_text(parsed: Dict[str, any], source_text: str) -> Dict[str, any]:
        if "ADUs" not in parsed:
            return parsed
        validated_adus: Dict[str, Dict[str, any]] = {}
        invalid: List[Dict[str, str]] = []
        valid_rel: List[Dict[str, str]] = []
        norm_source = "".join(source_text.split()).lower()
        basic_source = " ".join(source_text.split())
        for adu_id, adu in parsed["ADUs"].items():
            text = adu.get("text", "")
            norm = "".join(text.split()).lower()
            basic = " ".join(text.split())
            ok = bool(norm) and (basic in basic_source or norm in norm_source)
            if ok:
                validated_adus[adu_id] = adu
            else:
                invalid.append({"id": adu_id, "text": text})
        for rel in parsed.get("relations", []):
            if rel.get("source") in validated_adus and rel.get("target") in validated_adus:
                valid_rel.append(rel)
        return {"ADUs": validated_adus, "relations": valid_rel}

    oai_client = None
    claude_client = None
    if openai is not None and os.getenv("OPEN_AI_API_KEY"):
        try:
            oai_client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))  # type: ignore[attr-defined]
        except Exception:
            oai_client = None
    if anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
        try:
            claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # type: ignore[attr-defined]
        except Exception:
            claude_client = None

    def call_openai_api(system_prompt: str, user_content: str, temperature: float, max_tokens: int, retries: int) -> Optional[Dict[str, any]]:
        if oai_client is None:
            return None
        for attempt in range(retries):
            try:
                response = oai_client.chat.completions.create(  # type: ignore[attr-defined]
                    model="gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                content = (response.choices[0].message.content or "").strip()
                if content.startswith("```json") and content.endswith("```"):
                    content = content[7:-3].strip()
                elif content.startswith("```") and content.endswith("```"):
                    content = content[3:-3].strip()
                try:
                    parsed = json.loads(content)
                    parsed_valid = validate_adus_in_text(parsed, user_content)
                    return {
                        "source": "gpt-4o-2024-11-20",
                        "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                        "raw_response": content,
                        "parsed_response_unvalidated": parsed,
                        "parsed_response": parsed_valid,
                    }
                except Exception as e:  # JSON error
                    if attempt == retries - 1:
                        return {
                            "source": "gpt-4o-2024-11-20",
                            "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                            "raw_response": content,
                            "parsed_response": None,
                            "error": f"Invalid JSON: {e}",
                        }
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "source": "gpt-4o-2024-11-20",
                        "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                        "raw_response": None,
                        "parsed_response": None,
                        "error": str(e),
                    }
        return None

    def call_anthropic_api(system_prompt: str, user_content: str, temperature: float, max_tokens: int, retries: int) -> Optional[Dict[str, any]]:
        if claude_client is None:
            return None
        for attempt in range(retries):
            try:
                response = claude_client.messages.create(  # type: ignore[attr-defined]
                    model="claude-3-5-sonnet-20241022",
                    system=system_prompt,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    messages=[{"role": "user", "content": user_content}],
                )
                content = ""
                for block in getattr(response, "content", []) or []:
                    if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                        content += block.text
                content = content.strip()
                if not content:
                    if attempt == retries - 1:
                        return {
                            "source": "claude-3-5-sonnet-20241022",
                            "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                            "raw_response": None,
                            "parsed_response": None,
                            "error": "Empty response content",
                        }
                    continue
                if content.startswith("```json") and content.endswith("```"):
                    content = content[7:-3].strip()
                elif content.startswith("```") and content.endswith("```"):
                    content = content[3:-3].strip()
                try:
                    parsed = json.loads(content)
                    parsed_valid = validate_adus_in_text(parsed, user_content)
                    return {
                        "source": "claude-3-5-sonnet-20241022",
                        "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                        "raw_response": content,
                        "parsed_response_unvalidated": parsed,
                        "parsed_response": parsed_valid,
                    }
                except Exception as e:
                    if attempt == retries - 1:
                        return {
                            "source": "claude-3-5-sonnet-20241022",
                            "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                            "raw_response": content,
                            "parsed_response": None,
                            "error": f"Invalid JSON: {e}",
                        }
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "source": "claude-3-3-5-sonnet-20241022",
                        "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens},
                        "raw_response": None,
                        "parsed_response": None,
                        "error": str(e),
                    }
        return None

    def save_arguments(path: Path, identifier: str, system_prompt: str, user_content: str, responses: List[Dict[str, any]], overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "identifier": identifier,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "responses": responses,
            "timestamp": time.time(),
        }
        if overwrite or not path.exists():
            output = [record]
        else:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
            output = existing + [record]
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_labels(path: Path, identifier: str, responses: List[Dict[str, any]], overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        labels: Dict[str, str] = {}
        for r in responses:
            parsed = r.get("parsed_response")
            if parsed and "ADUs" in parsed:
                source = r.get("source", "")
                hp = r.get("hyperparameters", {}) or {}
                t = hp.get("temperature", 0.1)
                mt = hp.get("max_tokens", 4000)
                source_name = f"{source} (T:{t:.2f}, MT:{mt})" if (t != 0.1 or mt != 4000) else source
                for _, adu in parsed["ADUs"].items():
                    if adu.get("label") == "claim" and adu.get("text"):
                        labels[adu["text"]] = source_name
        if not labels:
            return
        if overwrite or not path.exists():
            output = labels
        else:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    existing.update(labels)
                    output = existing
                else:
                    output = labels
            except Exception:
                output = labels
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    cfg = Config.load()
    defaults = _default_paths(cfg)
    sample_csv = (config.sample_csv if config and config.sample_csv else None) or defaults["sample_csv"]
    papers_dir = (config.papers_dir if config and config.papers_dir else None) or defaults["papers_dir"]
    labels_dir = (config.labels_dir if config and config.labels_dir else None) or defaults["labels_dir"]
    arguments_dir = (config.arguments_dir if config and config.arguments_dir else None) or defaults["arguments_dir"]
    prefer_md = bool(config.prefer_markdown) if config else False

    if not sample_csv or not papers_dir or not labels_dir or not arguments_dir:
        raise ValueError("Missing paths; ensure config.yaml has workshop and docling paths configured")

    logger = get_logger("annotations.labels")
    ids = _load_identifiers(Path(sample_csv))
    if config and config.number_limit:
        ids = ids[: int(config.number_limit)]
    logger.info(f"Processing {len(ids)} papers from {sample_csv}")

    labels_out = Path(str(labels_dir))
    args_out = Path(str(arguments_dir))
    papers_path = Path(str(papers_dir))
    if not labels_out.is_absolute():
        labels_out = _ROOT / labels_out
    if not args_out.is_absolute():
        args_out = _ROOT / args_out
    if not papers_path.is_absolute():
        papers_path = _ROOT / papers_path
    labels_out.mkdir(parents=True, exist_ok=True)
    args_out.mkdir(parents=True, exist_ok=True)

    # Determine hyperparameter grid
    enable_search = (
        config is not None
        and config.min_temperature is not None
        and config.max_temperature is not None
        and config.min_tokens is not None
        and config.max_tokens is not None
        and (
            config.min_temperature != config.max_temperature
            or config.min_tokens != config.max_tokens
        )
    )
    if enable_search:
        hp_list = generate_hyperparameters(
            float(config.min_temperature),
            float(config.max_temperature),
            int(config.min_tokens),
            int(config.max_tokens),
            int(config.num_combinations),
        )
    else:
        hp_list = [{"temperature": 0.1, "max_tokens": 4000}]

    created: Dict[str, Path] = {}
    for i, identifier in enumerate(ids, 1):
        args_path = args_out / f"{identifier}.json"
        labels_path = labels_out / f"{identifier}.json"

        # Skip existing logic
        if config:
            if config.skip_existing and args_path.exists() and labels_path.exists():
                logger.info(f"Skipping {identifier}: both outputs exist")
                continue
            if config.skip_existing_labels and labels_path.exists():
                logger.info(f"Skipping {identifier}: labels exist")
                continue
            if config.skip_existing_arguments and args_path.exists():
                logger.info(f"Skipping {identifier}: arguments exist")
                continue

        # Load paper text
        content = load_paper_content(papers_path, identifier, prefer_md)
        if not content:
            if not config or config.skip_not_found:
                logger.warning(f"Skipping {identifier}: paper content not found")
                continue
            raise FileNotFoundError(f"Paper content not found for {identifier}")

        # Call APIs per hyperparameter combo
        responses: List[Dict[str, any]] = []
        for j, hp in enumerate(hp_list, 1):
            oai = call_openai_api(load_prompt_template(), content, float(hp["temperature"]), int(hp["max_tokens"]), int(config.retries if config else 3))
            if oai:
                responses.append(oai)
        for j, hp in enumerate(hp_list, 1):
            anth = call_anthropic_api(load_prompt_template(), content, float(hp["temperature"]), int(hp["max_tokens"]), int(config.retries if config else 3))
            if anth:
                responses.append(anth)

        if not responses:
            logger.warning(f"No successful API responses for {identifier}")
            continue

        # Determine overwrite/append per output
        overwrite_args = False
        overwrite_labels = False
        if config:
            overwrite_args = bool(config.overwrite_arguments or (not config.append_arguments and (config.overwrite_labels or config.overwrite_arguments)))
            overwrite_labels = bool(config.overwrite_labels or (not config.append_labels and (config.overwrite_labels or config.overwrite_arguments)))

        save_arguments(args_path, identifier, load_prompt_template(), content, responses, overwrite_args)
        save_labels(labels_path, identifier, responses, overwrite_labels)

        created[f"arguments:{identifier}"] = args_path
        created[f"labels:{identifier}"] = labels_path

    return created


