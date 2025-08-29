#!/usr/bin/env python3
"""
Metadata field distribution analysis for the PhilPapers dataset.
Analyzes distributions of language, type, year, authors, and titles.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from ...config import Config
from . import get_most_recent_file

logger = logging.getLogger(__name__)



def parse_year(year_str):
    """Extract year from string with various formats."""
    if pd.isna(year_str) or year_str == "":
        return None

    s = str(year_str).strip()
    if not s:
        return None

    # Simple regex for 1-4 digit numbers
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


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data with vectorized operations.
    
    Returns:
        Tuple of (processed_df, exploded_df)
    """
    logger.info("Preprocessing data...")

    # Clean year
    if "year" in df.columns:
        logger.info("Processing years...")
        df["year_clean"] = df["year"].apply(parse_year)

    # Count multi-value fields
    multi_cols = ["authors", "category_names"]
    for col in multi_cols:
        if col in df.columns:
            logger.info(f"Counting {col}...")
            df[f"{col}_count"] = count_semicolon_separated(df[col])

    # Copy category count
    if "category_names" in df.columns and "category_ids" in df.columns:
        df["category_ids_count"] = df["category_names_count"]

    # Title length
    if "title" in df.columns:
        logger.info("Processing title lengths...")
        df["title_length"] = df["title"].str.len()
        df["title_word_count"] = vectorized_word_count(df["title"])

    # Create exploded version for multi-value fields
    logger.info("Creating exploded dataframe...")
    exploded_df = df.copy()
    for col in multi_cols:
        if col in exploded_df.columns:
            mask = exploded_df[col].notna() & (exploded_df[col].str.strip() != "")
            exploded_df = exploded_df[mask].copy()
            exploded_df[col] = exploded_df[col].str.split(";")
            exploded_df = exploded_df.explode(col, ignore_index=True)
            exploded_df = exploded_df[
                exploded_df[col].str.strip().ne("") & exploded_df[col].notna()
            ]

    logger.info(f"Preprocessed {len(df)} papers, exploded to {len(exploded_df)} rows")
    return df, exploded_df


def compute_field_distributions(paper_df: pd.DataFrame, exploded_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute distributions for all metadata fields.
    
    Args:
        paper_df: Original preprocessed dataframe
        exploded_df: Exploded version for multi-value fields
        
    Returns:
        Dictionary with field statistics
    """
    stats = {}

    # Type distribution
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
                "missing_count": missing_count,
                "missing_percentage": float(missing_count / len(paper_df) * 100),
                "distribution": distribution,
                "top_10": distribution[:10],
            }

    # Language distribution
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
                "missing_count": missing_count,
                "missing_percentage": float(missing_count / len(paper_df) * 100),
                "distribution": distribution,
                "top_10": distribution[:10],
            }

    # Year statistics
    if "year_clean" in paper_df.columns:
        year_series = paper_df["year_clean"]
        missing_count = int(year_series.isna().sum())
        valid_series = year_series.dropna()
        
        if not valid_series.empty:
            year_counts = valid_series.value_counts()
            stats["year"] = {
                "count": len(valid_series),
                "missing_count": missing_count,
                "mean": float(valid_series.mean()),
                "median": float(valid_series.median()),
                "std": float(valid_series.std()),
                "min": float(valid_series.min()),
                "max": float(valid_series.max()),
                "top_10_years": [
                    {
                        "year": int(year),
                        "count": int(count),
                        "percentage": float(count / len(valid_series) * 100),
                    }
                    for year, count in year_counts.head(10).items()
                ],
            }

    # Author statistics
    if "authors" in exploded_df.columns:
        papers_no_authors = int(paper_df["authors"].isna().sum())
        author_series = exploded_df["authors"].dropna()
        author_series = author_series[author_series.str.strip() != ""]
        
        if not author_series.empty:
            author_counts = author_series.value_counts()
            total_authors = len(author_counts)
            distribution = [
                {
                    "author": author,
                    "paper_count": int(count),
                }
                for author, count in author_counts.items()
            ]
            
            stats["authors"] = {
                "count": total_authors,
                "missing_authors": papers_no_authors,
                "top_20": distribution[:20],
            }

    # Title statistics
    if "title_length" in paper_df.columns:
        title_lengths_chars = paper_df["title_length"].dropna()
        missing_titles = int(paper_df["title"].isna().sum())
        
        if not title_lengths_chars.empty:
            stats["title_length_chars"] = {
                "count": len(title_lengths_chars),
                "missing_count": missing_titles,
                "mean": float(title_lengths_chars.mean()),
                "median": float(title_lengths_chars.median()),
                "std": float(title_lengths_chars.std()),
                "min": float(title_lengths_chars.min()),
                "max": float(title_lengths_chars.max()),
            }

    if "title_word_count" in paper_df.columns:
        title_lengths_words = paper_df["title_word_count"].dropna()
        if not title_lengths_words.empty:
            stats["title_length_words"] = {
                "count": len(title_lengths_words),
                "mean": float(title_lengths_words.mean()),
                "median": float(title_lengths_words.median()),
                "std": float(title_lengths_words.std()),
                "min": float(title_lengths_words.min()),
                "max": float(title_lengths_words.max()),
            }

    # Title semantic diversity
    stats["title_semantic_diversity"] = compute_title_diversity(paper_df)

    # Categories per paper
    if "category_names_count" in paper_df.columns:
        cats_per_paper = paper_df["category_names_count"].dropna()
        papers_no_categories = int((paper_df["category_names_count"] == 0).sum())
        
        if not cats_per_paper.empty:
            stats["categories_per_paper"] = {
                "count": len(cats_per_paper),
                "papers_with_no_categories": papers_no_categories,
                "mean": float(cats_per_paper.mean()),
                "median": float(cats_per_paper.median()),
                "std": float(cats_per_paper.std()),
                "min": float(cats_per_paper.min()),
                "max": float(cats_per_paper.max()),
            }

    return stats


def compute_title_diversity(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute title semantic diversity using TF-IDF."""
    if "title" not in df.columns:
        return {}

    titles = df["title"].dropna().str.strip()
    titles = titles[titles != ""]

    if len(titles) < 2:
        return {"error": "Not enough titles for diversity analysis"}

    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", min_df=2)
        tfidf_matrix = vectorizer.fit_transform(titles)
        tfidf_dense = tfidf_matrix.toarray()
        centroid = np.mean(tfidf_dense, axis=0)
        distances = cosine_distances(tfidf_matrix, centroid.reshape(1, -1)).flatten()

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


def create_distribution_plots(paper_df: pd.DataFrame, stats: Dict[str, Any], output_dir: Path) -> None:
    """Create distribution plots for metadata fields."""
    
    # Language distribution plot
    if "language" in stats and "top_10" in stats["language"]:
        top_langs = stats["language"]["top_10"]
        if top_langs:
            # Get language names
            lang_map = {l.alpha_2: l.name for l in pycountry.languages if hasattr(l, 'alpha_2')}
            labels = []
            counts = []
            for item in top_langs:
                lang_code = item["value"]
                lang_name = lang_map.get(lang_code, lang_code)
                labels.append(lang_name)
                counts.append(item["count"])
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=labels, y=counts)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Language')
            plt.ylabel('Count')
            plt.title('Top 10 Languages')
            plt.tight_layout()
            plt.savefig(output_dir / "language_distribution.png", dpi=150)
            plt.close()

    # Type distribution plot
    if "type" in stats and "top_10" in stats["type"]:
        top_types = stats["type"]["top_10"]
        if top_types:
            labels = [item["value"] for item in top_types]
            counts = [item["count"] for item in top_types]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(labels)), counts)
            ax.set_title("Type Distribution (Top 10)")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel("Count")
            
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / "type_distribution.png", dpi=150)
            plt.close()

    # Year distribution plot
    if "year_clean" in paper_df.columns:
        years = paper_df["year_clean"].dropna()
        if not years.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            year_min, year_max = max(0, years.min()), min(2026, years.max())
            ax1.hist(years, bins=50, alpha=0.7, edgecolor='black', range=(year_min, year_max))
            ax1.axvline(years.mean(), color='red', linestyle='--', label=f'Mean: {years.mean():.1f}')
            ax1.set_yscale('log')
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
            plt.savefig(output_dir / "year_distribution.png", dpi=150)
            plt.close()


def run_metadata_distributions(
    input_file: Optional[str] = None,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Run metadata distribution analysis.
    
    Args:
        input_file: Optional path to metadata CSV file
        config: Optional Config instance
        
    Returns:
        Dictionary with all distribution statistics
    """
    if config is None:
        config = Config.load()
    
    # Setup paths
    metadata_dir = Path(config.get('paths.philpapers.metadata'))
    output_dir = Path(config.get('paths.figures.philpapers')) / 'distributions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input file
    if input_file:
        input_path = Path(input_file)
    else:
        input_path = get_most_recent_file(metadata_dir, '*metadata*.csv')
        if not input_path:
            input_path = get_most_recent_file(metadata_dir, '*.csv')
    
    if not input_path or not input_path.exists():
        logger.error(f"No input file found")
        return {"error": "No input file found"}
    
    # Load and preprocess data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} papers")
    
    paper_df, exploded_df = preprocess_data(df)
    
    # Compute distributions
    stats = compute_field_distributions(paper_df, exploded_df)
    
    # Create plots
    create_distribution_plots(paper_df, stats, output_dir)
    
    # Save results
    with open(output_dir / "field_distributions.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    return stats