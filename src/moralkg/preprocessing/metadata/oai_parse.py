import csv
import re
import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

import rootutils
from moralkg.config import Config
from moralkg.logging import get_logger


DEFAULT_SCHEMA: Dict = {
    "mappings": {
        "dc:title": "title",
        "dc:type": "type",
        "dc:creator": "authors",
        "dc:subject": "subjects",
        "dc:date": "year",
        "dc:identifier": "identifier-url",
        "dc:language": "language",
    },
    "special": {
        "identifier": {
            "xpath": "header/identifier",
            # Note: id root changed from philarchive.org to philpapers.org in 2025
            "pattern": r"oai:philpapers.org/rec/(?P<id>.+)$",
            "field": "identifier",
        }
    },
}


def _extract_texts(elem: ET.Element, tag: str) -> List[str]:
    return [e.text.strip() for e in elem.findall(f".//{{*}}{tag}") if e.text]


_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


@dataclass
class ParseFilters:
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    languages: Optional[List[str]] = None
    include_deleted: bool = True
    record_range: Optional[tuple] = None  # (start, end) inclusive, 1-based


class Parser:
    """Parse OAI-PMH XML repository files into a flat CSV."""

    def __init__(self, schema: Optional[Dict] = None):
        # Load schema from Config if not provided, else fallback to DEFAULT_SCHEMA
        if schema is None:
            loaded_schema: Optional[Dict] = None
            try:
                cfg = Config.load()
                loaded_schema = cfg.get("philpapers.metadata.schema")
            except Exception:
                loaded_schema = None
            self.schema = loaded_schema or DEFAULT_SCHEMA
        else:
            self.schema = schema
        self._logger = get_logger(__name__)

    def find_input_files(self, glob_pattern: Optional[str]) -> List[Path]:
        """Resolve input files for parsing.

        - If a glob pattern is provided, use it as-is
        - Else prefer directory from Config at `philpapers.metadata.dir`
        - Finally, fallback to current working directory's XML files
        """
        if glob_pattern:
            return sorted(Path(p) for p in glob.glob(glob_pattern))

        # Try config directory
        cfg_dir = None
        try:
            cfg = Config.load()
            cfg_dir = cfg.get("philpapers.metadata.dir")
        except Exception:
            cfg = None
            cfg_dir = None

        if cfg_dir:
            cfg_path = Path(str(cfg_dir))
            if not cfg_path.is_absolute():
                cfg_path = _ROOT / cfg_path
            pat = str(cfg_path / "*.xml")
            files = sorted(Path(p) for p in glob.glob(pat))
            if files:
                return files

        # Last resort: any XMLs in CWD
        return sorted(Path(p) for p in glob.glob("*.xml"))

    def parse_files(
        self,
        input_files: Iterable[Path],
        output_csv: Path,
        filters: Optional[ParseFilters] = None,
        limit: Optional[int] = None,
    ) -> int:
        filters = filters or ParseFilters()
        schema = self.schema

        mappings = schema["mappings"]
        fieldnames = ["identifier"] + list(mappings.values())

        processed = 0
        filtered_idx = 0
        stop_parsing = False

        with open(output_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for xml_file in input_files:
                if stop_parsing:
                    break
                try:
                    tree = ET.parse(xml_file)
                except Exception:
                    self._logger.warning("Failed to parse XML file %s", xml_file)
                    continue
                root = tree.getroot()

                for rec in root.findall(".//record"):
                    header = rec.find("header")
                    if header is None:
                        continue

                    if (
                        header.get("status") == "deleted"
                        and not filters.include_deleted
                    ):
                        continue

                    dt = header.findtext("datestamp")
                    if not dt:
                        continue
                    dts = datetime.fromisoformat(dt.replace("Z", ""))
                    if filters.start_date and dts < filters.start_date:
                        continue
                    if filters.end_date and dts > filters.end_date:
                        continue

                    filtered_idx += 1
                    if filters.record_range:
                        start, end = filters.record_range
                        if filtered_idx < start:
                            continue
                        if filtered_idx > end:
                            stop_parsing = True
                            break

                    metadata = rec.find("metadata")
                    if metadata is None:
                        continue

                    if filters.languages:
                        langs = _extract_texts(metadata, "language")
                        if not any(l in filters.languages for l in langs):
                            continue

                    row: Dict[str, str] = {}

                    # Identifier
                    raw_id = header.findtext("identifier", "").strip()
                    pat = (
                        schema.get("special", {})
                        .get("identifier", {})
                        .get("pattern", "")
                    )
                    m = re.match(pat, raw_id)
                    row["identifier"] = m.group("id") if m else raw_id

                    # Mappings
                    for xml_tag, col in mappings.items():
                        _, tag = xml_tag.split(":", 1)
                        vals = _extract_texts(metadata, tag)
                        if vals:
                            row[col] = ";".join(vals)

                    writer.writerow(row)
                    processed += 1

                    # Stop after writing 'limit' records if specified
                    if limit is not None and processed >= limit:
                        stop_parsing = True
                        break

        return processed


def parse_metadata(
    input_glob: Optional[str],
    output_csv: str,
    schema: Optional[Dict] = None,
    filters: Optional[ParseFilters] = None,
) -> int:
    parser = Parser(schema=schema)
    files = parser.find_input_files(input_glob)
    return parser.parse_files(files, Path(output_csv), filters)
