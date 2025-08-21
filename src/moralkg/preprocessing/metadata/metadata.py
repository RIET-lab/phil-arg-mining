from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import rootutils

from moralkg.config import Config


_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


@dataclass(frozen=True)
class MetadataSource:
    directory: Path
    file: Path


class Metadata:
    """Convenience loader and accessor for PhilPapers/PhilArchive metadata CSVs.

    - Resolves the metadata directory from Config at `philpapers.metadata.dir`
    - Locates the most recent CSV (prefers `*-en-combined-metadata.csv`),
      unless a specific filename is provided
    - Exposes a pandas DataFrame and simple retrieval/filter helpers
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        metadata_dir: Optional[str | Path] = None,
        filename: Optional[str | Path] = None,
    ) -> None:
        self._config = config or Config.load()
        self._dir = self._resolve_directory(metadata_dir)
        self._source = self._resolve_source(self._dir, filename, self._config)
        self._df = self._load_dataframe(self._source.file)

    @staticmethod
    def _resolve_directory(metadata_dir: Optional[str | Path]) -> Path:
        if metadata_dir is not None:
            p = Path(metadata_dir)
            return p if p.is_absolute() else (_ROOT / p)
        # From config.yaml; may be relative to repo root
        cfg_dir = Config.load().get("philpapers.metadata.dir")
        if not cfg_dir:
            raise ValueError("Config missing 'philpapers.metadata.dir'")
        p = Path(cfg_dir)
        return p if p.is_absolute() else (_ROOT / p)

    @staticmethod
    def _pick_latest(files: Iterable[Path]) -> Optional[Path]:
        files_sorted = sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        return files_sorted[0] if files_sorted else None

    @classmethod
    def _resolve_source(
        cls,
        directory: Path,
        filename: Optional[str | Path],
        config: Optional[Config] = None,
    ) -> MetadataSource:
        directory.mkdir(parents=True, exist_ok=True)
        # Priority 1: explicit filename argument
        if filename:
            f = Path(filename)
            f = f if f.is_absolute() else (directory / f)
            if not f.exists():
                raise FileNotFoundError(f"Metadata CSV not found: {f}")
            return MetadataSource(directory=directory, file=f)

        # Priority 2: config override (philpapers.metadata.file)
        if config is not None:
            cfg_file = config.get("philpapers.metadata.file")
            if cfg_file:
                f = Path(str(cfg_file))
                if not f.is_absolute():
                    # Try relative to configured directory first
                    f_dir = directory / f
                    if f_dir.exists():
                        return MetadataSource(directory=directory, file=f_dir)
                    # Fallback: relative to repo root
                    f = _ROOT / f
                if not f.exists():
                    raise FileNotFoundError(f"Config 'philpapers.metadata.file' not found: {f}")
                # Directory may not match configured directory; keep original directory for context
                return MetadataSource(directory=directory, file=f)

        # Prefer combined English CSVs, then English CSVs, then any CSV
        patterns = [
            "*-en-combined-metadata.csv",
            "*-en.csv",
            "*.csv",
        ]
        for pat in patterns:
            candidates = list(directory.glob(pat))
            picked = cls._pick_latest(candidates)
            if picked is not None:
                return MetadataSource(directory=directory, file=picked)

        raise FileNotFoundError(f"No CSV files found in {directory}")

    @staticmethod
    def _load_dataframe(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        # Maintain identifier index if present
        if "identifier" in df.columns:
            df.set_index("identifier", drop=False, inplace=True)
        return df

    @property
    def source(self) -> MetadataSource:
        return self._source

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def columns(self) -> List[str]:
        return list(self._df.columns)

    def identifiers(self) -> List[str]:
        if "identifier" in self._df.columns:
            return self._df["identifier"].astype(str).tolist()
        return self._df.index.astype(str).tolist()

    def get(self, identifier: str) -> Optional[pd.Series]:
        try:
            return self._df.loc[identifier]
        except KeyError:
            # Some CSVs may lack identifier index or have string/integer mismatch
            matches = self._df[self._df.get("identifier", "").astype(str) == str(identifier)]
            if not matches.empty:
                return matches.iloc[0]
            return None

    def find_by_title_contains(self, substring: str, case: bool = False) -> pd.DataFrame:
        col = "title"
        if col not in self._df.columns:
            return self._df.iloc[0:0]
        return self._df[self._df[col].astype(str).str.contains(substring, case=case, na=False)]

    def filter(
        self,
        language: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
        types: Optional[List[str]] = None,
        subjects_contains: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self._df
        if language and "language" in df.columns:
            df = df[df["language"].astype(str) == language]
        if year_range and "year" in df.columns:
            start, end = year_range
            # Some year values may be non-numeric; coerce
            years = pd.to_numeric(df["year"], errors="coerce")
            df = df[(years >= start) & (years <= end)]
        if types and "type" in df.columns:
            df = df[df["type"].astype(str).isin(types)]
        if subjects_contains and "subjects" in df.columns:
            df = df[df["subjects"].astype(str).str.contains(subjects_contains, case=False, na=False)]
        return df

    @lru_cache(maxsize=1)
    def most_recent_file(self) -> Path:
        return self._source.file

    # --------- Simple analysis helpers ---------
    def language_distribution(self) -> pd.DataFrame:
        if "language" not in self._df.columns:
            return pd.DataFrame(columns=["language", "count"]).set_index("language")
        vc = self._df["language"].astype(str).value_counts(dropna=False)
        return vc.rename_axis("language").to_frame("count")

    def type_distribution(self) -> pd.DataFrame:
        if "type" not in self._df.columns:
            return pd.DataFrame(columns=["type", "count"]).set_index("type")
        vc = self._df["type"].astype(str).value_counts(dropna=False)
        return vc.rename_axis("type").to_frame("count")

    def year_distribution(self) -> pd.DataFrame:
        if "year" not in self._df.columns:
            return pd.DataFrame(columns=["year", "count"]).set_index("year")
        years = pd.to_numeric(self._df["year"], errors="coerce").dropna().astype(int)
        vc = years.value_counts().sort_index()
        return vc.rename_axis("year").to_frame("count")

    def top_subjects(self, top_n: int = 50) -> pd.DataFrame:
        if "subjects" not in self._df.columns:
            return pd.DataFrame(columns=["subject", "count"]).set_index("subject")
        s = (
            self._df["subjects"].dropna().astype(str).str.split(";")
            .explode()
            .str.strip()
        )
        s = s[s != ""]
        vc = s.value_counts().head(top_n)
        return vc.rename_axis("subject").to_frame("count")

    def summary(self) -> dict:
        result = {
            "total_papers": int(len(self._df)),
            "columns": self.columns,
        }
        if "language" in self._df.columns:
            result["num_languages"] = int(self._df["language"].nunique(dropna=False))
        if "type" in self._df.columns:
            result["num_types"] = int(self._df["type"].nunique(dropna=False))
        if "year" in self._df.columns:
            years = pd.to_numeric(self._df["year"], errors="coerce").dropna()
            if not years.empty:
                result["year_min"] = int(years.min())
                result["year_max"] = int(years.max())
        return result


