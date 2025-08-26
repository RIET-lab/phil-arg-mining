""" 
This should first load in the metadata for philpapers by getting it from config.paths.philpapers.metadata. if the path is a directory, gets most recent file. if the path is a file, loads that. Either way, there should be a metadata attribute that links paper metadata to the paper contents. There should be metadata 'sub-attributes' that correspond with the metadata columns which make it simple to access specific metadata. metadata should be structured where the key is the paper ID and values are the other columns. 

Papers should be dynamically grabbed with get_paper() and loaded in from the path declared at config.paths.philpapers.docling.cleaned. It should look for a file like <metadata.id>.md then <metadata.id>.txt. No paper should raise an error.

Annotations should be batch loaded in during the init and tied to papers. what annotations to load in should be decided by config.workshop.annotations.use. if "large", just use config.paths.workshop.annotations.large_maps, if "both" also use config.paths.workshop.annotations.small_maps.

Maps should be parsed into an ArgumentMap (see from moralkg.argmining.schemas import ArgumentMap, ADU, Relation). Parsing should be done with from moralkg.argmining.parsers import Parser.
"""

from __future__ import annotations

import csv
import json
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import rootutils

from moralkg import Config, get_logger
from moralkg.argmining.parsers import Parser
from moralkg.argmining.schemas import ArgumentMap


_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


class _MetadataAccessor:
    """Helper to expose metadata by id and by column as attributes.

    - Index access: accessor[paper_id] -> dict of all columns for that id
    - Attribute per column: accessor.title[paper_id] -> title string
    - ids: list of all paper ids
    - columns: list of available columns
    """

    def __init__(self, rows_by_id: Dict[str, Dict[str, str]]):
        self._index = rows_by_id
        # Collect columns from any row
        columns: List[str] = []
        for row in rows_by_id.values():
            columns = list(row.keys())
            break
        self._columns = set(columns)
        self._by_column: Dict[str, Dict[str, str]] = {}
        for col in self._columns:
            self._by_column[col] = {pid: row.get(col) for pid, row in rows_by_id.items()}

    def __getitem__(self, paper_id: str) -> Optional[Dict[str, str]]:
        return self._index.get(paper_id)

    def __getattr__(self, name: str):
        if name in self._by_column:
            return self._by_column[name]
        raise AttributeError(f"Unknown metadata field: {name}")

    @property
    def ids(self) -> List[str]:
        return list(self._index.keys())

    @property
    def columns(self) -> List[str]:
        return list(self._columns)


@dataclass
class AnnotationIndex:
    """Holds parsed annotations indexed by paper id."""

    by_paper: Dict[str, List[ArgumentMap]]
    all: List[ArgumentMap]


class Dataset:
    """Dataset/Loader for PhilPapers papers + optional workshop annotations.

    Responsibilities:
    - Load metadata from config.paths.philpapers.metadata (file or newest in dir)
    - Provide convenient metadata accessors keyed by paper id and by column
    - Provide `get_paper(id)` to lazily read cleaned docling text (.md → .txt)
    - Batch-load workshop annotations and parse into ArgumentMap, tied to ids
    """

    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        self._logger = get_logger(__name__)
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = Config.load(Path(config_path))
        else:
            self._config = Config.load()

        # Resolve important paths from config
        self._metadata_path = self._resolve_path(self._config.get("paths.philpapers.metadata"))
        self._papers_dir = self._resolve_path(self._config.get("paths.philpapers.docling.cleaned"))

        # Load metadata and build accessors
        rows_by_id = self._load_metadata(self._metadata_path)
        self.metadata = _MetadataAccessor(rows_by_id)

        # Batch load annotations (workshop) and index by paper
        self.annotations = self._load_annotations()

    def get_paper(self, paper_id: str) -> Optional[str]:
        """Return paper text for the given id, if available.

        Search order inside cleaned docling dir:
        1) <id>.md
        2) <id>.txt

        """
        try:
            if not self._papers_dir:
                self._logger.warning("No papers directory configured (paths.philpapers.docling.cleaned)")
                return None

            candidates = [self._papers_dir / f"{paper_id}.md", self._papers_dir / f"{paper_id}.txt"]
            for path in candidates:
                if path.exists() and path.is_file():
                    return path.read_text(encoding="utf-8", errors="ignore")

            self._logger.error("Paper not found for id=%s in %s", paper_id, str(self._papers_dir))
            return None

        except Exception as e:
            self._logger.error("Error reading paper %s: %s", paper_id, e)
            return None


    def _resolve_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else (_ROOT / p)

    def _load_metadata(self, path: Optional[Path]) -> Dict[str, Dict[str, str]]:
        if path is None:
            self._logger.warning("Missing config paths.philpapers.metadata; metadata will be empty")
            return {}

        source: Optional[Path] = None
        if path.is_file():
            source = path
        elif path.is_dir():
            # Prefer CSVs by newest mtime; fallback to JSON if present
            csvs = sorted((p for p in path.glob("*.csv") if p.is_file()), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
            source = csvs[0] if csvs else None
        else:
            self._logger.warning("Configured metadata path does not exist: %s", str(path))

        if source is None:
            self._logger.warning("No metadata file found under %s", str(path))
            return {}

        try:
            if source.suffix.lower() == ".csv":
                rows = self._read_csv(source)
            else:
                self._logger.warning("Unsupported metadata file type: %s", source.suffix)
                rows = []
        except Exception as e:
            self._logger.error("Failed to load metadata from %s: %s", str(source), e)
            rows = []

        by_id: Dict[str, Dict[str, str]] = {}
        for row in rows:
            # Normalize id field
            pid = str(row.get("identifier") or "").strip()
            if not pid:
                continue

            row = dict(row)
            by_id[pid] = row

        self._logger.info("Loaded %d metadata records from %s", len(by_id), str(source))
        return by_id

    def _read_csv(self, path: Path) -> List[Dict[str, str]]:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [self._strip_keys_values(r) for r in reader]

    def _strip_keys_values(self, row: Dict[str, object]) -> Dict[str, str]:
        clean: Dict[str, str] = {}
        for k, v in row.items():
            key = str(k).strip()
            if isinstance(v, (str, int, float)) or v is None:
                clean[key] = "" if v is None else str(v)
            else:
                # Flatten simple lists by semicolon; else JSON-dump
                if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
                    clean[key] = ";".join(str(x) for x in v)
                else:
                    try:
                        clean[key] = json.dumps(v, ensure_ascii=False)
                    except Exception:
                        clean[key] = str(v)
        return clean

    def _glob_first(self, base_dir: Path, patterns: Iterable[str]) -> Optional[Path]:
        for pattern in patterns:
            for p in base_dir.rglob("*"):
                if p.is_file() and fnmatch.fnmatch(p.as_posix(), (base_dir / pattern).as_posix()):
                    return p
        return None

    def _load_annotations(self) -> AnnotationIndex:
        """Load and parse workshop annotations into ArgumentMap objects.

        Directories used depend on `workshop.annotations.use`:
        - "large": only `paths.workshop.annotations.large_json_maps`
        - "both": include `paths.workshop.annotations.small_json_maps` too
        """
        use_setting = (self._config.get("workshop.annotations.use") or "large").strip().lower()
        include_small = use_setting == "both"

        large_dir = self._resolve_path(self._config.get("paths.workshop.annotations.large_json_maps"))
        small_dir = self._resolve_path(self._config.get("paths.workshop.annotations.small_json_maps")) if include_small else None

        schema_path = self._resolve_path(self._config.get("argmining.schema"))
        parser = Parser(schema_path=str(schema_path) if schema_path else None)

        json_paths: List[Path] = []
        for d in [p for p in [large_dir, small_dir] if p is not None]:
            if d.exists() and d.is_dir():
                json_paths.extend(sorted(d.rglob("*.json")))
            else:
                self._logger.debug("Annotation directory missing or not a dir: %s", str(d))

        by_paper: Dict[str, List[ArgumentMap]] = {}
        all_maps: List[ArgumentMap] = []

        for jp in json_paths:
            try:
                amap = parser.parse_json_file(jp)
                paper_id = amap.id or jp.stem
                by_paper.setdefault(paper_id, []).append(amap)
                all_maps.append(amap)
            except Exception as e:
                self._logger.warning("Failed to parse annotation JSON %s: %s", str(jp), e)

        self._logger.info("Loaded %d annotation maps across %d papers", len(all_maps), len(by_paper))
        return AnnotationIndex(by_paper=by_paper, all=all_maps)
