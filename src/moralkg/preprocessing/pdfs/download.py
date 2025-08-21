from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import requests
import time
import rootutils
from moralkg.config import Config
from moralkg.logging import get_logger


_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


@dataclass
class DownloadFilters:
    """Filters and options controlling which PDFs to download and how.

    Attributes
    ----------
    languages: Optional[List[str]]
        Language codes to include; matches substrings in the `language` column if present.
    start_date: Optional[datetime]
        Inclusive lower bound compared against a parsed `year` column.
    end_date: Optional[datetime]
        Inclusive upper bound compared against a parsed `year` column.
    limit: Optional[int]
        Maximum number of identifiers to consider.
    shuffle: bool
        Whether to shuffle identifiers before limiting/downloading.
    skip_existing: bool
        If True, do not re-download PDFs that already exist on disk.
    sleep_time: float
        Seconds to sleep between successive HTTP requests (politeness/backoff).
    """

    languages: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None
    shuffle: bool = True
    skip_existing: bool = False
    sleep_time: float = 0.0


class PaperDownloader:
    """Download PhilArchive/PhilPapers PDFs based on metadata identifiers.

    This class is configured via the repository `Config` and exposes methods to
    locate the input metadata CSV, filter identifiers, and download PDFs to the
    configured output directory.
    """

    def __init__(self, config: Config, output_dir: Optional[Path] = None) -> None:
        self._config = config
        self._output_dir = self._resolve_output_dir(output_dir)
        self._logger = get_logger(__name__)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def _resolve_output_dir(self, override: Optional[Path]) -> Path:
        if override is not None:
            out = Path(override)
        else:
            rel_dir = self._config.get("philpapers.papers.pdfs.dir")
            if not rel_dir:
                raise ValueError("Config key 'philpapers.papers.pdfs.dir' is required")
            out = Path(rel_dir)
        # Resolve relative paths against repo root
        if not out.is_absolute():
            out = (_ROOT / out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    def find_input_file(self, explicit: Optional[Path] = None) -> Path:
        """Resolve the input metadata CSV file.

        Order of precedence:
        1) Explicit path if provided
        2) `philpapers.metadata.file` override in config (if it exists)
        3) Most recent CSV in `philpapers.metadata.dir`
        """
        if explicit is not None:
            csv_path = Path(explicit)
            if not csv_path.is_absolute():
                csv_path = (_ROOT / csv_path).resolve()
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {csv_path}")
            self._logger.info("Using input CSV file: %s", csv_path)
            return csv_path

        # Config override file
        override_file = self._config.get("philpapers.metadata.file")
        if override_file:
            csv_path = Path(override_file)
            if not csv_path.is_absolute():
                csv_path = (_ROOT / csv_path).resolve()
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Configured metadata file not found: {csv_path}"
                )
            self._logger.info("Using configured metadata file: %s", csv_path)
            return csv_path

        # Most recent in metadata dir
        metadata_dir = self._config.get("philpapers.metadata.dir")
        if not metadata_dir:
            raise ValueError("Config key 'philpapers.metadata.dir' is required")
        dir_path = Path(metadata_dir)
        if not dir_path.is_absolute():
            dir_path = (_ROOT / dir_path).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Metadata directory not found: {dir_path}")
        csv_files = sorted(dir_path.glob("*.csv"), reverse=True)
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dir_path}")
        self._logger.info("Using most recent CSV file: %s", csv_files[0])
        return csv_files[0]

    def load_identifiers(self, input_file: Path, filters: DownloadFilters) -> List[str]:
        """Load and filter identifiers from the metadata CSV.

        Expects a column named `identifier`. Optionally filters by `language`
        and by a parsed date in the `year` column.
        """
        self._logger.info("Loading data from %s", input_file)
        df = pd.read_csv(input_file)
        if "identifier" not in df.columns:
            raise ValueError("CSV must contain 'identifier' column")
        self._logger.info("Loaded %d records", len(df))

        # Language filter
        if filters.languages and "language" in df.columns:
            pattern = "|".join(filters.languages)
            df = df[df["language"].astype(str).str.contains(pattern, case=False, na=False)]
            self._logger.info("After language filter: %d records", len(df))

        # Date range filter
        if (filters.start_date is not None or filters.end_date is not None) and (
            "year" in df.columns
        ):
            df["year_dt"] = pd.to_datetime(df["year"], errors="coerce")
            if filters.start_date is not None:
                df = df[df["year_dt"] >= filters.start_date]
                self._logger.info("After start date filter: %d records", len(df))
            if filters.end_date is not None:
                df = df[df["year_dt"] <= filters.end_date]
                self._logger.info("After end date filter: %d records", len(df))

        # Shuffle before limiting
        if filters.shuffle:
            df = df.sample(frac=1.0, random_state=None).reset_index(drop=True)

        # Limit
        if filters.limit is not None:
            df = df.head(filters.limit)
            self._logger.info("After limit: %d records", len(df))

        return df["identifier"].astype(str).tolist()

    def download_paper(self, identifier: str, *, sleep_time: float = 0.0, skip_existing: bool = False) -> bool:
        """Download a single paper by PhilArchive identifier.

        Returns True on successful download, False otherwise.
        """
        output_file = self.output_dir / f"{identifier}.pdf"

        if skip_existing and output_file.exists():
            self._logger.info("Skipping existing file: %s", output_file)
            return True

        url = f"https://philpapers.org/archive/{identifier}.pdf"
        try:
            response = requests.get(url)
        except Exception as ex:
            self._logger.error("Failed to get %s, %s", url, ex)
            return False

        if not response.ok:
            if response.status_code in [404]:
                # Mark missing with empty placeholder file to avoid retries
                output_file.touch()
                self._logger.warning("Marked missing document %s", url)
                return False
            self._logger.error("Error %s, status code %s", url, response.status_code)
            return False

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb") as f:
            f.write(response.content)

        if sleep_time:
            time.sleep(sleep_time)
        self._logger.info("Downloaded: %s", output_file)
        return True

    def download_many(self, identifiers: Sequence[str], filters: DownloadFilters) -> Tuple[int, int]:
        """Download multiple identifiers. Returns (num_success, total)."""
        success = 0
        total = len(identifiers)
        for index, identifier in enumerate(identifiers, start=1):
            self._logger.info("[%d/%d] Processing %s", index, total, identifier)
            if self.download_paper(
                identifier,
                sleep_time=filters.sleep_time,
                skip_existing=filters.skip_existing,
            ):
                success += 1
        return success, total


def download_pdfs(
    config: Config,
    *,
    input_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    filters: Optional[DownloadFilters] = None,
    identifiers: Optional[Iterable[str]] = None,
) -> Tuple[int, int]:
    """Convenience wrapper to download PDFs end-to-end.

    If `identifiers` is provided, the metadata CSV is not read.
    Returns (num_success, total).
    """
    filters = filters or DownloadFilters()
    downloader = PaperDownloader(config=config, output_dir=output_dir)

    if identifiers is None:
        csv_file = downloader.find_input_file(input_csv)
        ids = downloader.load_identifiers(csv_file, filters)
    else:
        ids = list(identifiers)

    logger = get_logger(__name__)
    logger.info("Will download %d papers to %s", len(ids), downloader.output_dir)
    return downloader.download_many(ids, filters)
