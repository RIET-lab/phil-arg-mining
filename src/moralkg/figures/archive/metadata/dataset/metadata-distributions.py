# metadata-distributions.py

"""
Script to compute comprehensive statistics against the metadata dataset.
Note: Cleans 'year' to extract valid 4-digit years
"""


import argparse
import glob
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from flexidate import parse as fl_parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Locate project root
ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def timer(func):
    """Simple timing decorator for performance monitoring."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute dataset summary statistics with preprocessing."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        default="data/metadata/2025-07-09-en-combined-metadata.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/figures/metadata/dataset/",
        help="Output directory for figures and summary",
    )
    parser.add_argument(
        "--docling-dir",
        "-d",
        type=str,
        default="data/docling",
        help="Directory containing docling processed files",
    )
    parser.add_argument(
        "--pdf-dir",
        "-p",
        type=str,
        default="data/pdfs",
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        default=None,
        help="Number of papers to analyze (for testing). If not specified, analyzes all papers.",
    )

    return parser.parse_args()


def parse_year(year_str):
    """Attempt to extract the year with flexidate first then with regex fallback."""
    if pd.isna(year_str) or year_str == "":
        return None

    s = str(year_str).strip()
    if not s:
        return None

    # Try flexidate first
    try:
        parsed_date = fl_parse(s)
        if parsed_date and hasattr(parsed_date, "year") and parsed_date.year:
            year = parsed_date.year
            return -int(year[1:]) if year.startswith("-") else int(year)
    except:
        pass

    # Simple regex fallback for 1-4 digit numbers
    match = re.search(r"\b(\d{1,4})\b", s)
    if match:
        year = int(match.group(1))
        # Only accept years in reasonable range
        if 1 <= year <= 2026:
            return year

    return None


def count_semicolon_separated(series):
    """Count semicolon-separated items in a series."""
    return series.fillna("").str.split(";").str.len().where(series.notna(), 0)


def vectorized_word_count(series):
    """Vectorized word count operation."""
    return series.fillna("").astype(str).str.findall(r"\w+").str.len()


@timer
def preprocess_data(df):
    """Preprocess data efficiently with vectorized operations."""
    logger.info("Preprocessing data...")

    # Clean year - vectorized operation
    if "year" in df.columns:
        logger.info("Processing years...")
        df["year_clean"] = df["year"].apply(
            parse_year
        )  # This one still needs apply due to complex logic

    # Count multi-value fields - already vectorized
    multi_cols = ["authors", "category_names"]
    for col in multi_cols:
        if col in df.columns:
            logger.info(f"Counting {col}...")
            df[f"{col}_count"] = count_semicolon_separated(df[col])

    # Copy category count (category_ids should match category_names)
    if "category_names" in df.columns and "category_ids" in df.columns:
        df["category_ids_count"] = df["category_names_count"]

    # Title length
    if "title" in df.columns:
        logger.info("Processing title lengths...")
        df["title_length"] = df["title"].str.len()
        df["title_word_count"] = vectorized_word_count(df["title"])

    # Create exploded version
    logger.info("Creating exploded dataframe...")
    exploded_df = df.copy()
    for col in multi_cols:
        if col in exploded_df.columns:
            # pre-filter
            mask = exploded_df[col].notna() & (exploded_df[col].str.strip() != "")
            exploded_df = exploded_df[mask].copy()
            exploded_df[col] = exploded_df[col].str.split(";")
            exploded_df = exploded_df.explode(col, ignore_index=True)
            # Final cleanup
            exploded_df = exploded_df[
                exploded_df[col].str.strip().ne("") & exploded_df[col].notna()
            ]

    logger.info(f"Preprocessed {len(df)} papers, exploded to {len(exploded_df)} rows")
    return df, exploded_df


def title_semantic_diversity(df):
    """Compute title semantic diversity."""
    if "title" not in df.columns:
        return {}

    titles = df["title"].dropna().str.strip()
    titles = titles[titles != ""]

    try:
        # TF-IDF and cosine distances
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", min_df=2)
        tfidf_matrix = vectorizer.fit_transform(titles)
        # Convert sparse matrix to dense array for centroid calculation
        tfidf_dense = tfidf_matrix.toarray()  # type: ignore
        centroid = np.mean(tfidf_dense, axis=0)
        distances = cosine_distances(tfidf_matrix, centroid.reshape(1, -1)).flatten()

        # Most diverse titles
        title_distances = sorted(
            zip(titles, distances), key=lambda x: x[1], reverse=True
        )

        return {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "most_diverse": [
                {"title": title, "distance": float(dist)}
                for title, dist in title_distances[:10]
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@lru_cache(maxsize=1)
def get_available_files(docling_dir, pdf_dir):
    """Pre-scan directories once and cache results."""
    logger.info("Pre-scanning directories for available files...")

    # Scan docling files
    docling_files = set()
    if docling_dir and Path(docling_dir).exists():
        docling_path = Path(docling_dir)
        for ext in ["*.txt", "*.md"]:
            for file_path in docling_path.glob(ext):
                # Extract identifier from filename (remove extension)
                identifier = file_path.stem
                docling_files.add(identifier)

        # Scan PDF files
    pdf_files = set()
    if pdf_dir and Path(pdf_dir).exists():
        pdf_path = Path(pdf_dir)
        for file_path in pdf_path.glob("*.pdf"):
            # Extract identifier from filename (remove .pdf)
            identifier = file_path.stem
            pdf_files.add(identifier)

    logger.info(
        f"Found {len(docling_files)} docling files and {len(pdf_files)} PDF files"
    )
    return docling_files, pdf_files


def read_file_content(file_path):
    """Read content from a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            word_count = len(re.findall(r"\w+", content))
            return len(content), word_count
    except Exception:
        pass
    return None, None


@timer
def analyze_paper_lengths(df, docling_dir):
    """Analyze paper content length from docling files."""
    if not docling_dir or not Path(docling_dir).exists():
        return {"error": "Docling directory not found or not specified"}

    docling_path = Path(docling_dir)
    # Scan docling files locally to avoid cache conflicts
    docling_files = set()
    if docling_path.exists():
        for ext in ["*.txt", "*.md"]:
            for file_path in docling_path.glob(ext):
                identifier = file_path.stem
                docling_files.add(identifier)

    # Filter identifiers that have docling files
    df_clean = df.dropna(subset=["identifier"]).copy()
    df_clean["identifier_str"] = df_clean["identifier"].astype(str).str.strip()

    # Check file availability
    df_clean["has_docling"] = df_clean["identifier_str"].isin(docling_files)
    papers_with_content = int(df_clean["has_docling"].sum())
    papers_without_content = len(df_clean) - papers_with_content

    if papers_with_content == 0:
        return {"error": "No content found in docling files"}

    # Get identifiers that have files
    identifiers_with_files = df_clean[df_clean["has_docling"]][
        "identifier_str"
    ].tolist()

    logger.info(f"Reading content from {len(identifiers_with_files)} docling files...")

    # Parallel file reading with batching
    char_lengths = []
    word_lengths = []
    batch_size = 500  # Process in batches to manage memory

    for i in range(0, len(identifiers_with_files), batch_size):
        batch = identifiers_with_files[i : i + batch_size]
        batch_paths = []

        # Find actual file paths for this batch
        for identifier in batch:
            file_path = None
            for ext in [".txt", ".md"]:
                candidate = docling_path / f"{identifier}{ext}"
                if candidate.exists():
                    file_path = candidate
                    break
            if file_path:
                batch_paths.append(file_path)

        # Parallel processing for this batch
        with ThreadPoolExecutor(max_workers=min(16, len(batch_paths))) as executor:
            futures = {
                executor.submit(read_file_content, path): path for path in batch_paths
            }

            for future in as_completed(futures):
                char_len, word_len = future.result()
                if char_len is not None and word_len is not None:
                    char_lengths.append(char_len)
                    word_lengths.append(word_len)

        if i % (batch_size * 10) == 0:
            logger.info(
                f"  Processed {min(i + batch_size, len(identifiers_with_files))}/{len(identifiers_with_files)} files..."
            )

    if not char_lengths:
        return {"error": "No valid content found in docling files"}

    # Efficient numpy operations for statistics
    char_array = np.array(char_lengths)
    word_array = np.array(word_lengths)

    stats = {
        "papers_with_content": papers_with_content,
        "papers_without_content": papers_without_content,
        "availability_rate": float(papers_with_content / len(df)),
        # Character length statistics
        "char_lengths": char_lengths,
        "char_mean": float(np.mean(char_array)),
        "char_median": float(np.median(char_array)),
        "char_std": float(np.std(char_array)),
        "char_min": int(np.min(char_array)),
        "char_max": int(np.max(char_array)),
        "char_q25": float(np.percentile(char_array, 25)),
        "char_q75": float(np.percentile(char_array, 75)),
        # Word length statistics
        "word_lengths": word_lengths,
        "word_mean": float(np.mean(word_array)),
        "word_median": float(np.median(word_array)),
        "word_std": float(np.std(word_array)),
        "word_min": int(np.min(word_array)),
        "word_max": int(np.max(word_array)),
        "word_q25": float(np.percentile(word_array, 25)),
        "word_q75": float(np.percentile(word_array, 75)),
    }

    return stats


def analyze_philosophy_ethics_categories(exploded_df):
    """Analyze categories containing philosophy/ethics-related terms."""
    if "category_names" not in exploded_df.columns:
        return {"error": "category_names column not found"}

    cat_series = exploded_df["category_names"].dropna()
    cat_series = cat_series[cat_series.str.strip() != ""]

    if cat_series.empty:
        return {"error": "No valid categories found"}

    # Get unique categories only (avoid counting duplicates)
    unique_categories = cat_series.unique()
    total_categories = len(unique_categories)

    # Keywords from the bash script
    phil_ethics_pattern = r"(philosophy|ethic|moral|value|virtue)"

    # Find matching categories (case-insensitive)
    matching_mask = (
        pd.Series(unique_categories)
        .str.lower()
        .str.contains(phil_ethics_pattern, case=False, na=False)
    )
    matching_categories = pd.Series(unique_categories)[matching_mask].tolist()
    matching_count = len(matching_categories)

    # Calculate percentage
    percentage = (
        (matching_count / total_categories * 100) if total_categories > 0 else 0.0
    )

    # Count papers for each group
    phil_ethics_papers = (
        exploded_df[
            exploded_df["category_names"]
            .str.lower()
            .str.contains(phil_ethics_pattern, case=False, na=False)
        ]["identifier"].nunique()
        if "identifier" in exploded_df.columns
        else 0
    )

    total_papers_with_categories = (
        exploded_df["identifier"].nunique()
        if "identifier" in exploded_df.columns
        else len(exploded_df)
    )
    other_papers = total_papers_with_categories - phil_ethics_papers

    return {
        "total_categories": int(total_categories),
        "phil_ethics_categories": int(matching_count),
        "other_categories": int(total_categories - matching_count),
        "phil_ethics_percentage": float(percentage),
        "other_percentage": float(100.0 - percentage),
        "phil_ethics_category_examples": matching_categories[
            :20
        ],  # Show up to 20 examples
        "phil_ethics_papers": int(phil_ethics_papers),
        "other_papers": int(other_papers),
        "total_papers_with_categories": int(total_papers_with_categories),
        "phil_ethics_papers_percentage": (
            float(phil_ethics_papers / total_papers_with_categories * 100)
            if total_papers_with_categories > 0
            else 0.0
        ),
    }


@timer
def analyze_file_availability(df, docling_dir, pdf_dir):
    """Analyze availability of docling and PDF files."""
    logger.info("Analyzing file availability...")

    # Pre-scan directories once
    docling_files, pdf_files = get_available_files(
        str(docling_dir) if docling_dir else None, str(pdf_dir) if pdf_dir else None
    )

    # Clean identifier column
    df_clean = df.dropna(subset=["identifier"]).copy()
    df_clean["identifier_str"] = df_clean["identifier"].astype(str).str.strip()

    # Get lists of identifiers that have and don't have files
    all_identifiers = set(df_clean["identifier_str"].tolist())

    # Missing file lists
    missing_docling = (
        list(all_identifiers - docling_files)
        if docling_files
        else list(all_identifiers)
    )
    missing_pdf = (
        list(all_identifiers - pdf_files) if pdf_files else list(all_identifiers)
    )

    # Operations for file availability
    docling_available = (
        int(df_clean["identifier_str"].isin(docling_files).sum())
        if docling_files
        else 0
    )
    pdf_available = (
        int(df_clean["identifier_str"].isin(pdf_files).sum()) if pdf_files else 0
    )

    total_papers = len(df)
    docling_missing = total_papers - docling_available
    pdf_missing = total_papers - pdf_available

    availability_stats = {
        "docling": {
            "available": int(docling_available),
            "missing": int(docling_missing),
            "rate": (
                float(docling_available / total_papers) if total_papers > 0 else 0.0
            ),
            "missing_files": missing_docling,
        },
        "pdf": {
            "available": int(pdf_available),
            "missing": int(pdf_missing),
            "rate": float(pdf_available / total_papers) if total_papers > 0 else 0.0,
            "missing_files": missing_pdf,
        },
    }

    return availability_stats


@timer
def compute_stats(paper_df, exploded_df, docling_dir=None, pdf_dir=None):
    # """Compute all statistics."""
    logger.info("Computing statistics...")
    stats = {}

    # Identifier: count of papers without any identifier label (should be 0)
    if "identifier" in paper_df.columns:
        missing = paper_df["identifier"].isna().sum()
        stats["identifier"] = {
            "missing_count": int(missing),
            "missing_percentage": float(missing / len(paper_df) * 100),
        }

    # Type:
    # - full distribution (for each type: paper count + % of total)
    # - top 10 types by paper count (may be less than 10 that exist)
    # - min/max frequent type
    # - count of papers missing a type label
    if "type" in paper_df.columns:
        type_series = paper_df["type"]
        missing_count = int(type_series.isna().sum())
        valid_series = type_series.dropna()

        if not valid_series.empty:
            counts = valid_series.value_counts()
            total = len(valid_series)
            distribution = [
                {
                    "value": val,
                    "count": int(count),
                    "percentage": float(count / total * 100),
                }
                for val, count in counts.items()
            ]

            stats["type"] = {
                "count": total,
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_count / len(paper_df) * 100),
                "distribution": distribution,
                "top_10": distribution[:10],
                "most_frequent": distribution[0] if distribution else None,
                "least_frequent": distribution[-1] if distribution else None,
            }

    # Language:
    # - full distribution (for each language: paper count + % of total)
    # - top 10 languages by paper count (may be less than 10 that exist)
    # - min/max frequent language
    # - count of papers without a language label
    if "language" in paper_df.columns:
        lang_series = paper_df["language"]
        missing_count = int(lang_series.isna().sum())
        valid_series = lang_series.dropna()

        if not valid_series.empty:
            counts = valid_series.value_counts()
            total = len(valid_series)
            distribution = [
                {
                    "value": val,
                    "count": int(count),
                    "percentage": float(count / total * 100),
                }
                for val, count in counts.items()
            ]

            stats["language"] = {
                "count": total,
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_count / len(paper_df) * 100),
                "distribution": distribution,
                "top_10": distribution[:10],
                "most_frequent": distribution[0] if distribution else None,
                "least_frequent": distribution[-1] if distribution else None,
            }

    # Year:
    # - full distribution (for each year: paper count + % of total)
    # - mean/median/sd
    # - quartiles
    # - top 10 years by paper count
    # - min/max publication year
    # - count of missing/invalid year entries
    if "year_clean" in paper_df.columns:
        year_series = paper_df["year_clean"]
        missing_count = int(year_series.isna().sum())
        valid_series = year_series.dropna()

        if not valid_series.empty:
            year_counts = valid_series.value_counts()

            stats["year"] = {
                "count": len(valid_series),
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_count / len(paper_df) * 100),
                "mean": float(valid_series.mean()),
                "median": float(valid_series.median()),
                "std": float(valid_series.std()),
                "min": float(valid_series.min()),
                "max": float(valid_series.max()),
                "q25": float(valid_series.quantile(0.25)),
                "q50": float(valid_series.quantile(0.50)),
                "q75": float(valid_series.quantile(0.75)),
                "top_10_years": [
                    {
                        "year": int(year),
                        "count": int(count),
                        "percentage": float(count / len(valid_series) * 100),
                    }
                    for year, count in year_counts.head(10).items()
                ],
            }

    # Authors:
    # - full distribution (for each author: count of papers + % of total)
    # - top 20 most-prolific authors
    # - min/max prolific author
    # - count of papers with no author listed
    if "authors" in exploded_df.columns:
        papers_no_authors = int(paper_df["authors"].isna().sum())
        author_series = exploded_df["authors"].dropna()
        author_series = author_series[author_series.str.strip() != ""]

        if not author_series.empty:
            author_counts = author_series.value_counts()
            total_authors = len(author_counts)
            total_papers_with_authors = len(paper_df) - papers_no_authors

            distribution = [
                {
                    "author": author,
                    "paper_count": int(count),
                    "percentage": float(count / total_papers_with_authors * 100),
                }
                for author, count in author_counts.items()
            ]

            stats["authors"] = {
                "count": total_authors,
                "missing_authors": papers_no_authors,
                "missing_authors_percentage": float(
                    papers_no_authors / len(paper_df) * 100
                ),
                "distribution": distribution,
                "top_20": distribution[:20],
                "most_prolific": distribution[0] if distribution else None,
                "least_prolific": distribution[-1] if distribution else None,
            }

    # Authors per paper:
    # - full distribution (for each paper: count of authors)
    # - min/max/mean/median/sd of authors per paper
    # - quartiles of authors per paper
    # - top 10 author-counts
    if "authors_count" in paper_df.columns:
        authors_per_paper = paper_df["authors_count"].dropna()

        if not authors_per_paper.empty:
            count_distribution = authors_per_paper.value_counts().sort_index()

            stats["authors_per_paper"] = {
                "count": len(authors_per_paper),
                "mean": float(authors_per_paper.mean()),
                "median": float(authors_per_paper.median()),
                "std": float(authors_per_paper.std()),
                "min": float(authors_per_paper.min()),
                "max": float(authors_per_paper.max()),
                "q25": float(authors_per_paper.quantile(0.25)),
                "q50": float(authors_per_paper.quantile(0.50)),
                "q75": float(authors_per_paper.quantile(0.75)),
                "distribution": [
                    {"author_count": int(count), "paper_count": int(papers)}
                    for count, papers in count_distribution.items()
                ],
                "top_10_counts": [
                    {"author_count": int(count), "paper_count": int(papers)}
                    for count, papers in count_distribution.head(10).items()
                ],
            }

    # Title length:
    # - min/max/mean/median/sd of title length in characters (incl. whitespace)
    # - min/max/mean/median/sd of title length in words
    # - quartiles for both
    # - count of papers missing a title
    if "title_length" in paper_df.columns:
        title_lengths_chars = paper_df["title_length"].dropna()
        missing_titles = int(paper_df["title"].isna().sum())

        if not title_lengths_chars.empty:
            char_count_distribution = title_lengths_chars.value_counts().sort_index()

            stats["title_length_chars"] = {
                "count": len(title_lengths_chars),
                "missing_count": missing_titles,
                "missing_percentage": float(missing_titles / len(paper_df) * 100),
                "mean": float(title_lengths_chars.mean()),
                "median": float(title_lengths_chars.median()),
                "std": float(title_lengths_chars.std()),
                "min": float(title_lengths_chars.min()),
                "max": float(title_lengths_chars.max()),
                "q25": float(title_lengths_chars.quantile(0.25)),
                "q50": float(title_lengths_chars.quantile(0.50)),
                "q75": float(title_lengths_chars.quantile(0.75)),
                "full_distribution": title_lengths_chars.tolist(),
            }

    if "title_word_count" in paper_df.columns:
        title_lengths_words = paper_df["title_word_count"].dropna()

        if not title_lengths_words.empty:
            word_count_distribution = title_lengths_words.value_counts().sort_index()

            stats["title_length_words"] = {
                "count": len(title_lengths_words),
                "mean": float(title_lengths_words.mean()),
                "median": float(title_lengths_words.median()),
                "std": float(title_lengths_words.std()),
                "min": float(title_lengths_words.min()),
                "max": float(title_lengths_words.max()),
                "q25": float(title_lengths_words.quantile(0.25)),
                "q50": float(title_lengths_words.quantile(0.50)),
                "q75": float(title_lengths_words.quantile(0.75)),
                "distribution": [
                    {"word_count": int(count), "paper_count": int(papers)}
                    for count, papers in word_count_distribution.items()
                ],
                "full_distribution": title_lengths_words.tolist(),
            }

    # Title semantic diversity:
    # - mean cosine distance to the centroid of title td-idf vector embeddings
    #   - (O(n) instead of O(n^2) for pairwise distance)
    # - top 10 titles with the highest avg pairwise cosine distance
    stats["title_semantic_diversity"] = title_semantic_diversity(paper_df)

    # Category names:
    # - full distribution (for each category name: count of papers + % of total)
    # - min/max frequent category name
    if "category_names" in exploded_df.columns:
        cat_series = exploded_df["category_names"].dropna()
        cat_series = cat_series[cat_series.str.strip() != ""]

        if not cat_series.empty:
            cat_counts = cat_series.value_counts()
            total_categories = len(cat_series)

            distribution = [
                {
                    "category": cat,
                    "count": int(count),
                    "percentage": float(count / total_categories * 100),
                }
                for cat, count in cat_counts.items()
            ]

            stats["category_names"] = {
                "count": total_categories,
                "unique_categories": len(cat_counts),
                "distribution": distribution,
                "top_10": distribution[:10],
                "most_frequent": distribution[0] if distribution else None,
                "least_frequent": distribution[-1] if distribution else None,
            }

    # category ids:
    # - same as category names
    if "category_ids" in exploded_df.columns:
        cat_id_series = exploded_df["category_ids"].dropna()
        cat_id_series = cat_id_series[cat_id_series.str.strip() != ""]

        if not cat_id_series.empty:
            cat_id_counts = cat_id_series.value_counts()
            total_cat_ids = len(cat_id_series)

            distribution = [
                {
                    "category_id": cat_id,
                    "count": int(count),
                    "percentage": float(count / total_cat_ids * 100),
                }
                for cat_id, count in cat_id_counts.items()
            ]

            stats["category_ids"] = {
                "count": total_cat_ids,
                "unique_category_ids": len(cat_id_counts),
                "distribution": distribution,
                "top_10": distribution[:10],
                "most_frequent": distribution[0] if distribution else None,
                "least_frequent": distribution[-1] if distribution else None,
            }

    # Number of categories:
    # - min/max/mean/median/sd of categories per paper
    # - full distribution (for each category per paper: count of paper + % of total)
    # - count of papers with no cateories
    if "category_names_count" in paper_df.columns:
        cats_per_paper = paper_df["category_names_count"].dropna()
        papers_no_categories = int((paper_df["category_names_count"] == 0).sum())

        if not cats_per_paper.empty:
            cat_count_distribution = cats_per_paper.value_counts().sort_index()

            stats["categories_per_paper"] = {
                "count": len(cats_per_paper),
                "papers_with_no_categories": papers_no_categories,
                "papers_with_no_categories_percentage": float(
                    papers_no_categories / len(paper_df) * 100
                ),
                "mean": float(cats_per_paper.mean()),
                "median": float(cats_per_paper.median()),
                "std": float(cats_per_paper.std()),
                "min": float(cats_per_paper.min()),
                "max": float(cats_per_paper.max()),
                "q25": float(cats_per_paper.quantile(0.25)),
                "q50": float(cats_per_paper.quantile(0.25)),
                "q75": float(cats_per_paper.quantile(0.75)),
                "distribution": [
                    {"category_count": int(count), "paper_count": int(papers)}
                    for count, papers in cat_count_distribution.items()
                ],
            }

    # Paper length analysis (from docling files)
    if docling_dir:
        logger.info("Analyzing paper lengths...")
        stats["paper_lengths"] = analyze_paper_lengths(paper_df, docling_dir)

    # File availability analysis
    if docling_dir or pdf_dir:
        logger.info("Analyzing file availability...")
        stats["file_availability"] = analyze_file_availability(
            paper_df, docling_dir, pdf_dir
        )

    # Philosophy/Ethics categories analysis
    logger.info("Analyzing philosophy/ethics categories...")
    stats["philosophy_ethics_categories"] = analyze_philosophy_ethics_categories(
        exploded_df
    )

    # Missing data analysis
    missing_data = {}
    for col in paper_df.columns:
        if col not in ["year_clean"]:  # Skip derived columns
            missing_count = int(paper_df[col].isna().sum())
            missing_data[col] = {
                "missing_count": missing_count,
                "missing_percentage": float(missing_count / len(paper_df) * 100),
                "present_count": len(paper_df) - missing_count,
            }

    # Add file availability to missing data analysis
    if "file_availability" in stats:
        file_avail = stats["file_availability"]
        missing_data["docling_files"] = {
            "missing_count": file_avail["docling"]["missing"],
            "missing_percentage": float((1 - file_avail["docling"]["rate"]) * 100),
            "present_count": file_avail["docling"]["available"],
        }
        missing_data["pdf_files"] = {
            "missing_count": file_avail["pdf"]["missing"],
            "missing_percentage": float((1 - file_avail["pdf"]["rate"]) * 100),
            "present_count": file_avail["pdf"]["available"],
        }

    stats["missing_data"] = missing_data

    return stats


@timer
def create_plots(paper_df, stats, out_dir):
    """Create plots from the statistics."""
    plt.style.use("default")

    # Save missing files lists as separate JSON/CSV files
    if "file_availability" in stats:
        missing_files_data = {
            "pdfs": stats["file_availability"]["pdf"]["missing_files"],
            "docling": stats["file_availability"]["docling"]["missing_files"]
        }

        # Save as JSON
        with open(out_dir / "missing_files.json", "w") as f:
            json.dump(missing_files_data, f, indent=2)

        # Save as CSV
        import csv
        with open(out_dir / "missing_files.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["file_type", "identifier"])
            for identifier in missing_files_data["pdfs"]:
                writer.writerow(["pdf", identifier])
            for identifier in missing_files_data["docling"]:
                writer.writerow(["docling", identifier])

    # Years distribution
    if "year_clean" in paper_df.columns:
        years = paper_df["year_clean"].dropna()
        if not years.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Histogram
            year_min, year_max = max(0, years.min()), min(2026, years.max())
            ax1.hist(years, bins=50, alpha=0.7, edgecolor='black', range=(year_min, year_max))
            ax1.axvline(years.mean(), color='red', linestyle='--', label=f'Mean: {years.mean():.1f}')
            ax1.set_yscale('log')
            ax1.set_ylim(1, None)
            ax1.set_xlim(0, None)
            ax1.set_title("Publication Years")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Count")
            ax1.set_xlim(year_min, year_max)
            ax1.legend()

            # Top years
            if "top_10_years" in stats.get("year", {}):
                top_years = stats["year"]["top_10_years"]
                years_list = [item["year"] for item in top_years]
                counts = [item["count"] for item in top_years]
                ax2.bar(years_list, counts)
                ax2.set_title("Top Publication Years")
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Count")

            plt.tight_layout()
            plt.savefig(out_dir / "years.png", dpi=150)
            plt.close()

    # Authors per paper
    if "authors_per_paper" in stats and "distribution" in stats["authors_per_paper"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        dist = stats["authors_per_paper"]["distribution"]
        values = [item["author_count"] for item in dist]
        counts = [item["paper_count"] for item in dist]

        ax.bar(values, counts)
        ax.set_yscale('log')
        # ax.set_ylim(1, None)
        ax.set_xlim(0, None)
        ax.set_title("Authors per Paper Distribution")
        ax.set_xlabel("Number of Authors")
        ax.set_ylabel("Number of Papers")

        plt.tight_layout()
        plt.savefig(out_dir / "authors_distribution_per_paper.png", dpi=150)
        plt.close()

    # Categories per paper
    if "categories_per_paper" in stats and "distribution" in stats["categories_per_paper"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        dist = stats["categories_per_paper"]["distribution"]
        values = [item["category_count"] for item in dist]
        counts = [item["paper_count"] for item in dist]

        ax.bar(values, counts)
        ax.set_yscale('log')
        # ax.set_ylim(1, None)
        # ax.set_xlim(0, None)
        ax.set_title("Categories per Paper Distribution")
        ax.set_xlabel("Number of Categories")
        ax.set_ylabel("Number of Papers")

        plt.tight_layout()
        plt.savefig(out_dir / "categories_distribution_per_paper.png", dpi=150)
        plt.close()

    # Title word count distribution
    if "title_length_words" in stats and "full_distribution" in stats["title_length_words"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        word_counts = stats["title_length_words"]["full_distribution"]

        # Create histogram with all values
        ax.hist(word_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)
        ax.set_title("Title Word Count Distribution")
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Number of Papers")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "title_word_count_distribution.png", dpi=150)
        plt.close()

    # NEW: Title character count distribution figure (replacing enhanced title analysis)
    if "title_length_chars" in stats and "full_distribution" in stats["title_length_chars"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        char_counts = stats["title_length_chars"]["full_distribution"]

        # Create histogram with all values
        ax.hist(char_counts, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)
        ax.set_title("Title Character Count Distribution")
        ax.set_xlabel("Number of Characters")
        ax.set_ylabel("Number of Papers")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "title_character_count_distribution.png", dpi=150)
        plt.close()

    # Top categories, language, and type plots
    for field in ["type", "language", "category_names"]:
        if field in stats and "top_10" in stats[field]:
            top_items = stats[field]["top_10"]
            if top_items:
                # Handle different key names
                if field == "category_names":
                    labels = [item["category"] for item in top_items]
                else:
                    labels = [item["value"] for item in top_items]
                counts = [item["count"] for item in top_items]

                # Top language figure
                if field == "language":
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.set_title("Language Distribution (Top 10)")

                    plt.tight_layout()
                    plt.savefig(out_dir / "language_top_10.png", dpi=150)
                    plt.close()

                # Top type figure
                elif field == "type":
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(range(len(labels)), counts)
                    ax.set_ylim(1, max(counts) * 1.1)
                    ax.set_xlim(-0.5, len(labels) - 0.5)
                    ax.set_title("Type Distribution (Top 10)")
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel("Count")

                    # Add count labels
                    for bar, count in zip(bars, counts):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               str(count), ha='center', va='bottom')

                    plt.tight_layout()
                    plt.savefig(out_dir / "type_top_10.png", dpi=150)
                    plt.close()

                # Top category names
                elif field == "category_names":
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(range(len(labels)), counts)
                    ax.set_title("Category Names Distribution (Top 10)")
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel("Count")

                    # Add count labels
                    for bar, count in zip(bars, counts):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               str(count), ha='center', va='bottom')

                    plt.tight_layout()
                    plt.savefig(out_dir / "category_names_top_10.png", dpi=150)
                    plt.close()

    # Paper length analysis plots
    if "paper_lengths" in stats and "error" not in stats["paper_lengths"]:
        paper_stats = stats["paper_lengths"]

        # Character and word length distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Paper Content Length Analysis', fontsize=16, fontweight='bold')

        # Character length histogram
        if "char_lengths" in paper_stats:
            char_data = paper_stats["char_lengths"]
            ax1.hist(char_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_yscale('log')
            ax1.set_ylim(1, None)
            ax1.set_xlim(0, None)
            ax1.set_title('Paper Length Distribution (Characters)')
            ax1.set_xlabel('Number of Characters')
            ax1.set_ylabel('Number of Papers')
            ax1.grid(True, alpha=0.3)

        # Word length histogram
        if "word_lengths" in paper_stats:
            word_data = paper_stats["word_lengths"]
            min_words = max(1, min(word_data))
            max_words = max(word_data)
            log_bins = np.logspace(np.log10(min_words), np.log10(max_words), 50)

            ax2.hist(word_data, bins=log_bins, alpha=0.7, color='red', edgecolor='black')
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_ylim(1, None)
            ax2.set_xlim(min_words, max_words)
            ax2.set_title('Paper Length Distribution (Words)')
            ax2.set_xlabel('Number of Words')
            ax2.set_ylabel('Number of Papers')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "paper_length_analysis.png", dpi=150)
        plt.close()

    # File availability analysis
    if "file_availability" in stats:
        availability_stats = stats["file_availability"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('File Availability Analysis', fontsize=16, fontweight='bold')

        # Docling availability
        if "docling" in availability_stats:
            docling_data = availability_stats["docling"]
            ax1.pie([docling_data['available'], docling_data['missing']],
                    labels=['Available', 'Missing'],
                    autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
            ax1.set_title(f'Docling Files\n(Rate: {docling_data["rate"]:.1%})')

        # PDF availability
        if "pdf" in availability_stats:
            pdf_data = availability_stats["pdf"]
            ax2.pie([pdf_data['available'], pdf_data['missing']],
                    labels=['Available', 'Missing'],
                    autopct='%1.1f%%',
                    colors=['lightblue', 'lightyellow'])
            ax2.set_title(f'PDF Files\n(Rate: {pdf_data["rate"]:.1%})')

        plt.tight_layout()
        plt.savefig(out_dir / "file_availability.png", dpi=150)
        plt.close()

    # Missing data analysis
    if "missing_data" in stats:
        missing_stats = stats["missing_data"]

        fig, ax = plt.subplots(figsize=(12, 8))

        fields = list(missing_stats.keys())
        missing_rates = [missing_stats[field]['missing_percentage'] for field in fields]

        # Create a color map
        colormap = plt.get_cmap('RdYlGn_r')
        colors = colormap(np.linspace(0.2, 0.8, len(fields)))
        bars = ax.barh(fields, missing_rates, color=colors)

        ax.set_title('Missing Data by Field', fontsize=16, fontweight='bold')
        ax.set_xlabel('Missing Rate (%)')
        ax.set_ylabel('Fields')
        ax.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        for i, (bar, rate) in enumerate(zip(bars, missing_rates)):
            ax.text(rate + 0.5, i, f'{rate:.1f}%', va='center', ha='left')

        plt.tight_layout()
        plt.savefig(out_dir / "missing_data_analysis.png", dpi=150)
        plt.close()

    # Philosophy/Ethics categories pie chart
    if (
        "philosophy_ethics_categories" in stats
        and "error" not in stats["philosophy_ethics_categories"]
    ):
        phil_stats = stats["philosophy_ethics_categories"]

        # Create pie chart showing philosophy/ethics vs other categories
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Philosophy/Ethics Categories Analysis", fontsize=16, fontweight="bold"
        )

        # Categories pie chart
        category_labels = ["Philosophy/Ethics", "Other"]
        category_counts = [
            phil_stats["phil_ethics_categories"],
            phil_stats["other_categories"],
        ]
        category_colors = ["lightcoral", "lightblue"]

        wedges1, texts1, autotexts1 = ax1.pie(
            category_counts,
            labels=category_labels,
            autopct="%1.1f%%",
            colors=category_colors,
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax1.set_title(
            f'Categories Distribution\n({phil_stats["total_categories"]} total categories)',
            fontsize=14,
            fontweight="bold",
        )

        # Papers pie chart
        papers_labels = [
            "Papers with Phil/Ethics Categories",
            "Papers with Other Categories",
        ]
        papers_counts = [phil_stats["phil_ethics_papers"], phil_stats["other_papers"]]
        papers_colors = ["lightcoral", "lightblue"]

        wedges2, texts2, autotexts2 = ax2.pie(
            papers_counts,
            labels=papers_labels,
            autopct="%1.1f%%",
            colors=papers_colors,
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax2.set_title(
            f'Papers Distribution\n({phil_stats["total_papers_with_categories"]} total papers)',
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(out_dir / "categories_moral_restriction.png", dpi=150)
        plt.close()

        # Create a detailed breakdown text file
        with open(out_dir / "categories_moral_restriction.txt", "w") as f:
            f.write("Philosophy/Ethics Categories Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total unique categories: {phil_stats['total_categories']}\n")
            f.write(
                f"Philosophy/Ethics categories: {phil_stats['phil_ethics_categories']} ({phil_stats['phil_ethics_percentage']:.2f}%)\n"
            )
            f.write(
                f"Other categories: {phil_stats['other_categories']} ({phil_stats['other_percentage']:.2f}%)\n\n"
            )
            f.write(
                f"Total papers with categories: {phil_stats['total_papers_with_categories']}\n"
            )
            f.write(
                f"Papers with Phil/Ethics categories: {phil_stats['phil_ethics_papers']} ({phil_stats['phil_ethics_papers_percentage']:.2f}%)\n"
            )
            f.write(f"Papers with other categories: {phil_stats['other_papers']}\n\n")
            f.write("Philosophy/Ethics Category Examples:\n")
            f.write("-" * 40 + "\n")
            for i, category in enumerate(
                phil_stats["phil_ethics_category_examples"], 1
            ):
                f.write(f"{i:2d}. {category}\n")
            f.write(
                f"\nSearch pattern used: (philosophy|ethic|moral|value|virtue)\n"
            )
            f.write("(case-insensitive)\n")


def main():
    args = parse_args()

    input_path = ROOT / args.input_file
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} papers")

    # Sample data if requested
    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size:,} papers for testing...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Using sample of {len(df):,} papers")
    elif args.sample_size:
        logger.info(
            f"Sample size ({args.sample_size:,}) >= dataset size ({len(df):,}), using all papers"
        )

    # Preprocess data
    paper_df, exploded_df = preprocess_data(df)

    # Compute statistics
    stats = compute_stats(
        paper_df,
        exploded_df,
        docling_dir=ROOT / args.docling_dir,
        pdf_dir=ROOT / args.pdf_dir,
    )

    # Save results
    logger.info("Saving results...")
    with open(out_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Create plots
    logger.info("Creating visualizations...")
    create_plots(paper_df, stats, out_dir)

    logger.info(f"Analysis complete! Results saved to {out_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE-OPTIMIZED ANALYSIS COMPLETE")
    print("=" * 60)
    if args.sample_size and args.sample_size < len(df):
        print(
            f"SAMPLE MODE: Analyzed {len(df):,} of total papers (sample size: {args.sample_size:,})"
        )
    else:
        print(f"Total papers analyzed: {len(df):,}")
    if "file_availability" in stats:
        docling_rate = stats["file_availability"]["docling"]["rate"]
        pdf_rate = stats["file_availability"]["pdf"]["rate"]
        print(f"Docling files available: {docling_rate:.1%}")
        print(f"PDF files available: {pdf_rate:.1%}")
    if "paper_lengths" in stats and "error" not in stats["paper_lengths"]:
        paper_stats = stats["paper_lengths"]
        print(f"Papers with content analyzed: {paper_stats['papers_with_content']:,}")
        print(
            f"Average paper length: {paper_stats['char_mean']:,.0f} characters, {paper_stats['word_mean']:,.0f} words"
        )
    if (
        "philosophy_ethics_categories" in stats
        and "error" not in stats["philosophy_ethics_categories"]
    ):
        phil_stats = stats["philosophy_ethics_categories"]
        print(
            f"Philosophy/Ethics categories: {phil_stats['phil_ethics_categories']:,} of {phil_stats['total_categories']:,} ({phil_stats['phil_ethics_percentage']:.1f}%)"
        )
        print(
            f"Papers with Phil/Ethics categories: {phil_stats['phil_ethics_papers']:,} of {phil_stats['total_papers_with_categories']:,} ({phil_stats['phil_ethics_papers_percentage']:.1f}%)"
        )
    print(f"Results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
