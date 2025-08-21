from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import rootutils

from moralkg.config import Config
from moralkg.logging import get_logger


_ROOT = Path(rootutils.setup_root(__file__, indicator=".git"))


@dataclass
class SamplingConfig:
    input_file: Optional[str] = None
    output_dir: Optional[str] = None
    sample_size: int = 100
    seed: int = 42
    allow_author_repeats: bool = False
    allow_year_repeats: bool = False
    allow_category_repeats: bool = False


def _setup_logger(destination_dir: Path) -> logging.Logger:
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("annotations.sampling")
    # Root logger is already configured via get_logger(); just return a named logger
    return logger


def _is_philosophy_related(category_names: str) -> bool:
    if pd.isna(category_names):
        return False
    keywords = ["philosophy", "ethic", "moral", "value", "virtue"]
    text = str(category_names).lower()
    return any(k in text for k in keywords)


def _parse_years(year_series: pd.Series) -> pd.Series:
    def parse_year(value) -> Optional[int]:
        try:
            text = str(value)
        except Exception:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 3:
            return None
        year = int(digits[:4]) if len(digits) >= 4 else int(digits)
        if 1500 <= year <= 2035:
            return year
        return None

    return pd.Series(year_series.apply(parse_year), index=year_series.index)


def _cluster_titles(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, int]:
    """
    Cluster paper titles using TF-IDF + K-means (parity with archive script).
    Uses a fixed k to avoid long silhouette search for large datasets.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    titles = df["title"].fillna("").astype(str)
    logger.info("Computing TF-IDF vectors for titles")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles)

    optimal_k = 94
    logger.info(f"Clustering titles with KMeans (k={optimal_k})")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(tfidf_matrix)
    df_clustered = df.copy()
    df_clustered["cluster"] = clusters
    return df_clustered, optimal_k


def _stratify(df: pd.DataFrame) -> pd.DataFrame:
    """Stratify within clusters by year quantiles and primary category."""
    import numpy as np

    df = df.copy()
    df["year_clean"] = _parse_years(df["year"].copy())
    df["category_clean"] = df["category_names"].str.split(";").str[0].str.strip()

    df["year_quantile"] = np.nan
    for cluster_id in df["cluster"].unique():
        cluster_mask = df["cluster"] == cluster_id
        cluster_years = df.loc[cluster_mask, "year_clean"].dropna()
        if len(cluster_years) >= 4:
            try:
                quantiles = pd.qcut(
                    cluster_years,
                    q=4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop",
                )
                df.loc[cluster_years.index, "year_quantile"] = quantiles
            except ValueError:
                df.loc[cluster_mask, "year_quantile"] = "Q1"
        else:
            df.loc[cluster_mask, "year_quantile"] = "Q1"

    df["year_quantile"] = df["year_quantile"].fillna("Q1")
    return df


def _allocate_by_cluster(df: pd.DataFrame, sample_size: int) -> Dict[int, int]:
    counts = df["cluster"].value_counts()
    total = int(len(df)) or 1
    allocation: Dict[int, int] = {}
    for cluster_id, count in counts.items():
        allocation[int(cluster_id)] = round((int(count) / total) * sample_size)
    # Adjust rounding to match sample_size
    diff = sample_size - sum(allocation.values())
    if diff != 0 and len(allocation) > 0:
        largest = max(allocation, key=lambda k: counts[k])
        allocation[largest] = max(0, allocation[largest] + diff)
    return allocation


def _extract_authors(author_str) -> Set[str]:
    if pd.isna(author_str) or not author_str:
        return set()
    return {a.strip() for a in str(author_str).split(";") if a.strip()}


def _violates_constraints(paper, used_authors: set, used_years: set, used_categories: set, cfg: SamplingConfig) -> bool:
    if not cfg.allow_author_repeats:
        authors = _extract_authors(paper.get("authors", ""))
        if authors.intersection(used_authors):
            return True
    if not cfg.allow_year_repeats:
        year = paper.get("year_clean")
        if year is not None and not pd.isna(year) and year in used_years:
            return True
    if not cfg.allow_category_repeats:
        category = paper.get("category_clean")
        if category in used_categories:
            return True
    return False


def _sample_within_cluster(cluster_df: pd.DataFrame, max_samples: int, cfg: SamplingConfig) -> List[dict]:
    if max_samples <= 0 or len(cluster_df) == 0:
        return []
    rng = np.random.default_rng(cfg.seed)
    used_authors: Set[str] = set()
    used_years: Set[int] = set()
    used_categories: Set[str] = set()
    sampled: List[dict] = []
    indices = list(cluster_df.index)
    rng.shuffle(indices)
    for idx in indices:
        if len(sampled) >= max_samples:
            break
        row = cluster_df.loc[idx]
        paper = row.to_dict()
        if _violates_constraints(paper, used_authors, used_years, used_categories, cfg):
            continue
        sampled.append(paper)
        if not cfg.allow_author_repeats:
            used_authors.update(_extract_authors(paper.get("authors", "")))
        if not cfg.allow_year_repeats:
            year = paper.get("year_clean")
            if year is not None and not pd.isna(year):
                used_years.add(int(year))
        if not cfg.allow_category_repeats:
            used_categories.add(paper.get("category_clean"))
    return sampled


def _filter_papers(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Filter to articles with valid docling files and word count >= 1000."""
    from pathlib import Path as _P

    logger.info("Filtering papers to articles with docling files >= 1000 words")
    original_count = len(df)
    df_filtered = df[df["type"] == "article"].copy() if "type" in df.columns else df.copy()

    docling_dir_cfg = Config.load().get("philpapers.papers.docling.cleaned.dir") or Config.load().get("philpapers.papers.docling.raw.dir")
    docling_dir = _P(str(docling_dir_cfg)) if docling_dir_cfg else (_ROOT / "data" / "docling")
    if not docling_dir.is_absolute():
        docling_dir = _ROOT / docling_dir

    valid_rows: List[dict] = []
    for _, paper in df_filtered.iterrows():
        identifier = str(paper["identifier"]) if "identifier" in paper else None
        if not identifier:
            continue
        md = docling_dir / f"{identifier}.md"
        txt = docling_dir / f"{identifier}.txt"
        doc_path = md if md.exists() else (txt if txt.exists() else None)
        if not doc_path:
            continue
        try:
            content = doc_path.read_text(encoding="utf-8", errors="ignore")
            if len(content.split()) >= 1000:
                valid_rows.append(paper.to_dict())
        except Exception:
            continue

    result = pd.DataFrame(valid_rows)
    logger.info(f"Filter reduced {original_count} -> {len(result)} papers")
    return result


def _generate_suggestions(
    df: pd.DataFrame,
    cluster_id: int,
    shortage: int,
    used_authors: set,
    used_years: set,
    used_categories: set,
    cfg: SamplingConfig,
    logger: logging.Logger,
) -> List[dict]:
    """Generate suggestions to fill shortages, mirroring archive behavior."""
    suggestions: List[dict] = []

    cluster_df = df[df["cluster"] == cluster_id].copy()
    valid_from_cluster = []
    for _, paper in cluster_df.iterrows():
        if not _violates_constraints(paper, used_authors, used_years, used_categories, cfg):
            valid_from_cluster.append({**paper.to_dict(), "suggestion_reason": f"valid_from_cluster_{cluster_id}"})

    take = min(shortage * 2, len(valid_from_cluster))
    if take > 0:
        suggestions.extend(pd.DataFrame(valid_from_cluster).sample(n=take, random_state=cfg.seed).to_dict(orient="records"))

    return suggestions


def _ensure_philosophy_ratio(sampled_df: pd.DataFrame, full_df: pd.DataFrame, cfg: SamplingConfig, logger: logging.Logger) -> pd.DataFrame:
    """Ensure >= 50% philosophy-related papers (approximate parity with archive)."""
    import numpy as np

    working = sampled_df.copy()
    min_needed = int(np.ceil(len(working) * 0.5))
    current = int(working["category_names"].apply(_is_philosophy_related).sum()) if "category_names" in working.columns else 0
    if current >= min_needed:
        return working

    sampled_ids = set(working["identifier"]) if "identifier" in working.columns else set()
    available = full_df[full_df["category_names"].apply(_is_philosophy_related)] if "category_names" in full_df.columns else pd.DataFrame()
    available = available[~available["identifier"].isin(sampled_ids)] if "identifier" in available.columns else available
    if len(available) == 0:
        logger.warning("No additional philosophy papers available to adjust ratio")
        return working

    deficit = min_needed - current
    to_add = min(deficit, len(available))
    if to_add <= 0:
        return working

    # Replace first N non-philosophy with philosophy
    non_phil_mask = ~working["category_names"].apply(_is_philosophy_related) if "category_names" in working.columns else pd.Series([False] * len(working))
    to_remove = working[non_phil_mask].head(to_add)
    remaining = working[~working["identifier"].isin(set(to_remove["identifier"]))] if "identifier" in working.columns else working
    replacements = available.sample(n=to_add, random_state=cfg.seed)
    adjusted = pd.concat([remaining, replacements], ignore_index=True)
    final_count = int(adjusted["category_names"].apply(_is_philosophy_related).sum()) if "category_names" in adjusted.columns else 0
    logger.info(f"Adjusted philosophy ratio to {final_count}/{len(adjusted)}")
    return adjusted


def create_sample(config: Optional[SamplingConfig] = None) -> Path:
    """
    Create a sampled CSV at `<workshop.sample.dir>/sample.csv` with `identifier` and
    `cluster|year|category` columns, using defaults from `config.yaml` unless overridden
    by `config`.
    """
    cfg = Config.load()

    # Determine input metadata file: if file is not set in config, auto-select most recent in dir
    input_file = (config.input_file if config and config.input_file else None) or cfg.get("philpapers.metadata.file")
    if not input_file:
        meta_dir = cfg.get("philpapers.metadata.dir")
        if not meta_dir:
            raise ValueError("philpapers.metadata.dir must be configured to auto-select metadata file")
        meta_dir_path = Path(str(meta_dir))
        if not meta_dir_path.is_absolute():
            meta_dir_path = _ROOT / meta_dir_path
        csvs = sorted(meta_dir_path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in metadata dir: {meta_dir_path}")
        input_file = str(csvs[0])

    sample_dir = (config.output_dir if config and config.output_dir else None) or cfg.get("workshop.sample.dir")
    if not sample_dir:
        raise ValueError("workshop.sample.dir must be set in config.yaml or passed explicitly")
    sample_size = (config.sample_size if config else None) or int(cfg.get("snowball.phase_2.sample_size", 100))
    seed = (config.seed if config else None) or int(cfg.get("general.seed", 42))

    effective = SamplingConfig(
        input_file=str(input_file),
        output_dir=str(sample_dir),
        sample_size=int(sample_size),
        seed=int(seed),
        allow_author_repeats=bool(config.allow_author_repeats) if config else False,
        allow_year_repeats=bool(config.allow_year_repeats) if config else False,
        allow_category_repeats=bool(config.allow_category_repeats) if config else False,
    )

    root_logger = get_logger()
    root_logger.info("Loading metadata for sampling")
    input_path = Path(str(input_file))
    if not input_path.is_absolute():
        input_path = _ROOT / input_path
    df = pd.read_csv(input_path)

    # Minimal validation
    required = ["title", "year", "category_names", "authors", "identifier"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")

    # Filter papers as per archive logic
    df_filtered = _filter_papers(df, root_logger)

    # Cluster, stratify, allocate
    logger = _setup_logger(Path(str(sample_dir)))
    df_clustered, k = _cluster_titles(df_filtered, logger)
    df_strat = _stratify(df_clustered)
    allocation = _allocate_by_cluster(df_strat, effective.sample_size)

    # Draw samples
    sampled_rows: List[dict] = []
    suggestions: List[dict] = []
    used_authors: set = set()
    used_years: set = set()
    used_categories: set = set()
    for cluster_id, max_count in allocation.items():
        cluster_df = df_strat[df_strat["cluster"] == cluster_id]
        cluster_sample = _sample_within_cluster(cluster_df, max_count, effective)
        sampled_rows.extend(cluster_sample)
        # Track used dimensions for suggestion generation
        for paper in cluster_sample:
            if not effective.allow_author_repeats:
                used_authors.update(_extract_authors(paper.get("authors", "")))
            if not effective.allow_year_repeats:
                py = paper.get("year_clean")
                if py is not None and not pd.isna(py):
                    used_years.add(int(py))
            if not effective.allow_category_repeats:
                used_categories.add(paper.get("category_clean"))
        # If shortfall, generate suggestions
        if len(cluster_sample) < max_count:
            shortage = max_count - len(cluster_sample)
            suggestions.extend(
                _generate_suggestions(
                    df_strat, cluster_id, shortage, used_authors, used_years, used_categories, effective, logger
                )
            )

    if len(sampled_rows) == 0:
        raise RuntimeError("No samples could be drawn; check inputs and constraints")

    sampled_df = pd.DataFrame(sampled_rows)
    if "category_names" in df_strat.columns and not sampled_df.empty:
        sampled_df = _ensure_philosophy_ratio(sampled_df, df_strat, effective, logger)
    sampled_df["cluster|year|category"] = (
        sampled_df["cluster"].astype(str)
        + "|"
        + sampled_df["year_clean"].astype(str)
        + "|"
        + sampled_df["category_clean"].astype(str)
    )

    out_dir = Path(str(sample_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample.csv"
    sampled_df[["identifier", "cluster|year|category"]].to_csv(out_path, index=False)

    # Save stratified dataset for downstream tools
    stratified_file = out_dir / "stratified_dataset.csv"
    df_strat.to_csv(stratified_file, index=False)

    # Save suggestions if any
    if len(suggestions) > 0:
        suggestions_df = pd.DataFrame(suggestions)
        suggestions_df["cluster|year|category"] = (
            suggestions_df["cluster"].astype(str)
            + "|"
            + suggestions_df["year_quantile"].astype(str)
            + "|"
            + suggestions_df["category_clean"].astype(str)
        )
        (out_dir / "suggestions.csv").write_text("", encoding="utf-8")  # ensure file exists if dataframe fails
        suggestions_file = out_dir / "suggestions.csv"
        try:
            suggestions_df[["identifier", "cluster|year|category", "suggestion_reason"]].to_csv(
                suggestions_file, index=False
            )
        except Exception:
            # Fallback minimal columns
            suggestions_df.to_csv(suggestions_file, index=False)

    logger.info(f"Saved sample to {out_path}")
    logger.info(f"Stratified dataset saved to {stratified_file}")
    return out_path


