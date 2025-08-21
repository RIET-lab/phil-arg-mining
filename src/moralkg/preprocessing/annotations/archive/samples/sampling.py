#!/usr/bin/env python3

"""
Hybrid clustering-stratified sampling from the Phil-Papers dataset.

This script implements a sophisticated sampling strategy that:
1. Filters papers to articles with valid docling files and mmore than 1000 words
2. Clusters papers by title semantics using TF-IDF + K-means
3. Stratifies within clusters by year quantiles and categories  
4. Allocates samples proportionally across strata
5. Enforces constraints on author/year/category repeats as specified
6. Ensures >=50% philosophy-related papers in final sample

Usage: python sampling.py -n 100 [options]
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import rootutils
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "sampling.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("sampling")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Proportional, hybrid clustering-stratified sampling"
    )

    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="Path to input CSV file",
        default="data/metadata/2025-07-09-en-combined-metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory path",
        default=None,  # To be set based on sample size
    )
    parser.add_argument(
        "--sample-size", "-n", type=int, help="Number of papers to sample", default=100
    )
    parser.add_argument(
        "--allow-author-repeats",
        "-a",
        help="Allow authors to repeat in sample (default: off)",
        action="store_true",
    )
    parser.add_argument(
        "--allow-year-repeats",
        "-y",
        help="Allow years to repeat in sample (default: off)",
        action="store_true",
    )
    parser.add_argument(
        "--allow-category-repeats",
        "-c",
        help="Allow categories to repeat in sample (default: off)",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="Random seed for reproducibility",
        default=42,
    )

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"data/annotations/samples/n{args.sample_size}"

    return args


def filter_papers(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Filter papers to articles with valid docling files over 1000 words.
    
    Args:
        df: Input DataFrame
        logger: Logger instance
    
    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering papers...")
    original_count = len(df)
    
    # Filter 1: Only articles
    df_filtered = df[df['type'] == 'article'].copy()
    logger.info(f"After type filter: {len(df_filtered)} papers (removed {original_count - len(df_filtered)})")
    
    # Filter 2: Check for docling files and word count
    valid_papers = []
    docling_dir = Path("data/docling")
    
    for _, paper in df_filtered.iterrows():
        identifier = paper['identifier']
        
        # Check for .md or .txt file (but not .doctags.txt)
        md_file = docling_dir / f"{identifier}.md"
        txt_file = docling_dir / f"{identifier}.txt"
        
        docling_file = None
        if md_file.exists():
            docling_file = md_file
        elif txt_file.exists():
            docling_file = txt_file
        
        if docling_file is None:
            continue
            
        # Check word count
        try:
            with open(docling_file, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())
                
            if word_count >= 1000:
                valid_papers.append(paper)
        except Exception as e:
            logger.warning(f"Error reading {docling_file}: {e}")
            continue
    
    if valid_papers:
        result_df = pd.DataFrame(valid_papers)
        logger.info(f"After docling + length filter: {len(result_df)} papers (removed {len(df_filtered) - len(result_df)})")
    else:
        logger.error("No papers passed filtering criteria")
        result_df = pd.DataFrame()
    
    return result_df


def is_philosophy_related(category_names: str) -> bool:
    """Check if paper contains philosophy-related keywords."""
    if pd.isna(category_names):
        return False
    
    keywords = ["philosophy", "ethic", "moral", "value", "virtue"]
    category_lower = str(category_names).lower()
    
    return any(keyword in category_lower for keyword in keywords)


def parse_years(year_series: pd.Series) -> pd.Series:
    """Extract 3-4 digit years from strings."""

    def parse_year(year_str) -> Optional[int]:
        if pd.isna(year_str) or not str(year_str).strip():
            return None

        match = re.search(r"(\d{3,4})", str(year_str))
        if match:
            year = int(match.group(1))
            if 1500 <= year <= 2030:
                return year
        return None

    return pd.Series(year_series.apply(parse_year), index=year_series.index)


def determine_optimal_k(
    tfidf_matrix, logger: logging.Logger, max_k: int = 20, min_k: int = 2
) -> int:
    """
    Determine optimal number of clusters using silhouette score.

    Args:
        tfidf_matrix: TF-IDF matrix for clustering
        logger: Logger instance for progress tracking
        max_k: Maximum number of clusters to try
        min_k: Minimum number of clusters to try

    Returns:
        Optimal number of clusters
    """
    n_samples = tfidf_matrix.shape[0]
    max_k = min(max_k, n_samples // 10)
    max_k = max(max_k, min_k)

    if max_k <= min_k:
        logger.info(f"Using minimum clusters: {min_k}")
        return min_k

    logger.info(
        f"Testing k values from {min_k} to {max_k} ({max_k - min_k + 1} values)"
    )

    best_score = -1
    best_k = min_k

    # Try different values of k
    for i, k in enumerate(range(min_k, max_k + 1), 1):
        try:
            logger.info(f"Testing k={k} ({i}/{max_k - min_k + 1})...")

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # Calculate silhouette score
            logger.info(f"  Computing silhouette score for k={k}...")
            score = silhouette_score(tfidf_matrix, cluster_labels)

            logger.info(f"  k={k}: silhouette score = {score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                logger.info(f"New best: k={k} with score {score:.4f}")

        except Exception as e:
            logger.warning(f"Failed to compute k={k}: {e}")
            continue

    logger.info(f"Optimal k selected: {best_k} with silhouette score {best_score:.4f}")
    return best_k


def cluster_titles(
    df: pd.DataFrame, logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """
    Cluster paper titles using TF-IDF + K-means.

    Args:
        df: DataFrame with 'title' column
        logger: Logger instance

    Returns:
        DataFrame with 'cluster' column added, number of clusters
    """
    logger.info("Starting title clustering...")

    # Prepare titles for TF-IDF
    titles = df["title"].fillna("").astype(str)

    # TF-IDF vectorization
    logger.info("Computing TF-IDF vectors...")
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(titles)
    logger.info(f"TF-IDF matrix shape: {getattr(tfidf_matrix, 'shape', 'unknown')}")

    # Determine optimal number of clusters
    logger.info("Determining optimal number of clusters...")
    # optimal_k = determine_optimal_k(tfidf_matrix, logger, max_k=100, min_k=2)
    optimal_k = 94

    logger.info(f"Performing K-means clustering with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    # Log cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    logger.info(
        f"Created {len(cluster_counts)} clusters with sizes: {cluster_counts.tolist()}"
    )

    return df_clustered, optimal_k


def stratify_within_clusters(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Stratify papers within each cluster by year quantiles and categories.

    Args:
        df: DataFrame with 'cluster' column
        logger: Logger instance

    Returns:
        DataFrame with 'year_quantile' and 'stratum' columns added
    """
    logger.info("Stratifying within clusters...")

    # Clean years
    logger.info("Cleaning year data...")
    df["year_clean"] = parse_years(df["year"].copy())  # type: ignore

    # Create year quantiles within each cluster
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
                # If qcut fails, use simple quartile boundaries
                df.loc[cluster_mask, "year_quantile"] = "Q1"
        else:
            # If too few years, put all in Q1
            df.loc[cluster_mask, "year_quantile"] = "Q1"

    # Fill missing year quantiles
    df["year_quantile"] = df["year_quantile"].fillna("Q1")

    # Take first/primary category if there are multiple
    df["category_clean"] = df["category_names"].str.split(";").str[0].str.strip()

    # Create stratum identifier: cluster|year_quantile|category
    df["stratum"] = (
        df["cluster"].astype(str)
        + "|"
        + df["year_quantile"].astype(str)
        + "|"
        + df["category_clean"].astype(str)
    )

    # Log stratification results
    stratum_counts = df["stratum"].value_counts()
    logger.info(
        f"Created {len(stratum_counts)} unique strata (largest: {stratum_counts.iloc[0]}, smallest: {stratum_counts.iloc[-1]})"
    )

    return df


def allocate_samples(
    df: pd.DataFrame, sample_size: int, logger: logging.Logger
) -> Dict[str, int]:
    """
    Allocate samples based on cluster proportion.

    Args:
        df: DataFrame with 'cluster' and 'stratum' columns
        sample_size: Total number of samples to allocate
        logger: Logger instance

    Returns:
        Dictionary mapping cluster_id to allocated sample count
    """
    logger.info("Allocating samples...")

    # Step 1: Allocate to clusters proportionally
    cluster_counts = df["cluster"].value_counts()
    total_papers = len(df)

    cluster_allocation = {}
    for cluster_id, count in cluster_counts.items():
        proportion = count / total_papers
        allocated = round(proportion * sample_size)
        cluster_allocation[cluster_id] = allocated

    # Adjust for rounding errors to ensure exact sample_size
    total_allocated = sum(cluster_allocation.values())
    diff = sample_size - total_allocated

    logger.info(
        f"Before adjustment: {total_allocated} samples allocated (target: {sample_size})"
    )

    if diff != 0:
        # Adjust largest cluster
        largest_cluster = max(
            cluster_allocation.keys(), key=lambda k: int(cluster_counts[k])
        )
        cluster_allocation[largest_cluster] = max(
            0, cluster_allocation[largest_cluster] + diff
        )

    final_total = sum(cluster_allocation.values())
    logger.info(
        f"After adjustment: {final_total} samples allocated (target: {sample_size})"
    )

    logger.info(f"Allocated samples across {len(cluster_allocation)} clusters")

    return cluster_allocation


def extract_authors(author_str) -> Set[str]:
    """Extract set of authors from semicolon-separated string."""
    if pd.isna(author_str) or not author_str:
        return set()
    return {author.strip() for author in str(author_str).split(";") if author.strip()}


def check_constraints(
    paper, used_authors: set, used_years: set, used_categories: set, args
) -> bool:
    """Check if paper violates any constraints."""
    # Author constraint
    if not args.allow_author_repeats:
        paper_authors = extract_authors(paper.get("authors", ""))
        if paper_authors.intersection(used_authors):
            return False

    # Year constraint
    if not args.allow_year_repeats:
        paper_year = paper.get("year_clean")
        if (
            paper_year is not None
            and not pd.isna(paper_year)
            and paper_year in used_years
        ):
            return False

    # Category constraint
    if not args.allow_category_repeats:
        paper_category = paper.get("category_clean")
        if paper_category in used_categories:
            return False

    return True


def generate_suggestions(
    df: pd.DataFrame,
    cluster_id,
    num_suggestions: int,
    used_authors: set,
    used_years: set,
    used_categories: set,
    args,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Generate suggestions for missing samples from a cluster.

    Args:
        df: Full DataFrame
        cluster_id: Cluster that needs suggestions
        num_suggestions: Number of suggestions to generate
        used_authors/years/categories: Already used values
        args: Command line arguments
        logger: Logger instance

    Returns:
        List of suggestion dictionaries
    """
    suggestions = []

    # Start with papers from the same cluster
    cluster_df = df[df["cluster"] == cluster_id].copy()

    # Remove already sampled papers (simple approach - all papers that would violate constraints)
    valid_papers = []
    for _, paper in cluster_df.iterrows():
        if check_constraints(paper, used_authors, used_years, used_categories, args):
            valid_papers.append(paper)

    # If we have valid papers from the same cluster, use them
    if valid_papers:
        valid_df = pd.DataFrame(valid_papers)
        sample_size = min(num_suggestions, len(valid_df))
        suggested_papers = valid_df.sample(n=sample_size, random_state=args.seed)

        for _, paper in suggested_papers.iterrows():
            suggestion = paper.to_dict()
            suggestion["suggestion_reason"] = f"valid_from_cluster_{cluster_id}"
            suggestions.append(suggestion)

    # If we still need more suggestions, look at other clusters
    remaining_needed = num_suggestions - len(suggestions)
    if remaining_needed > 0:
        other_clusters_df = df[df["cluster"] != cluster_id].copy()

        # Apply same constraint filtering
        other_valid_papers = []
        for _, paper in other_clusters_df.iterrows():
            if check_constraints(
                paper, used_authors, used_years, used_categories, args
            ):
                other_valid_papers.append(paper)

        if other_valid_papers:
            other_valid_df = pd.DataFrame(other_valid_papers)
            sample_size = min(remaining_needed, len(other_valid_df))
            suggested_papers = other_valid_df.sample(
                n=sample_size, random_state=args.seed
            )

            for _, paper in suggested_papers.iterrows():
                suggestion = paper.to_dict()
                suggestion["suggestion_reason"] = (
                    f'valid_from_other_cluster_{paper["cluster"]}'
                )
                suggestions.append(suggestion)

    logger.info(f"Generated {len(suggestions)} suggestions for cluster {cluster_id}")
    return suggestions


def adjust_for_philosophy_requirement(
    sampled_df: pd.DataFrame, full_df: pd.DataFrame, args, logger: logging.Logger
) -> pd.DataFrame:
    """
    Adjust sample to ensure >=50% philosophy-related papers.
    
    Args:
        sampled_df: Current sample DataFrame
        full_df: Full dataset DataFrame
        args: Command line arguments
        logger: Logger instance
    
    Returns:
        Adjusted sample DataFrame
    """
    # Work with a copy to avoid modifying the original
    sampled_copy = sampled_df.copy()
    
    target_size = len(sampled_copy)
    min_philosophy_needed = int(np.ceil(target_size * 0.5))
    
    # Count current philosophy papers
    sampled_copy['is_philosophy'] = sampled_copy['category_names'].apply(is_philosophy_related)
    current_philosophy = int(sampled_copy['is_philosophy'].sum())
    
    logger.info(f"Need {min_philosophy_needed} philosophy papers, have {current_philosophy}")
    
    if current_philosophy >= min_philosophy_needed:
        return sampled_copy.drop('is_philosophy', axis=1)
    
    # Get already sampled identifiers (convert set to list for isin)
    sampled_ids = list(sampled_copy['identifier'])
    
    # Find philosophy papers not yet sampled
    available_philosophy = full_df[
        (full_df['is_philosophy'] == True) & 
        (~full_df['identifier'].isin(sampled_ids))
    ].copy()
    
    if len(available_philosophy) == 0:
        logger.warning("No additional philosophy papers available")
        return sampled_copy.drop('is_philosophy', axis=1)
    
    # Replace non-philosophy papers with philosophy papers
    non_philosophy_mask = sampled_copy['is_philosophy'] == False
    non_philosophy_papers = sampled_copy[non_philosophy_mask]
    
    papers_to_replace = min_philosophy_needed - current_philosophy
    papers_to_replace = min(papers_to_replace, len(non_philosophy_papers), len(available_philosophy))
    
    if papers_to_replace > 0:
        # Remove some non-philosophy papers
        to_remove = non_philosophy_papers.head(papers_to_replace)
        to_remove_ids = list(to_remove['identifier'])
        remaining_sample = sampled_copy[~sampled_copy['identifier'].isin(to_remove_ids)]
        
        # Add philosophy papers
        replacement_philosophy = available_philosophy.sample(n=papers_to_replace, random_state=args.seed)
        
        # Remove temporary column before concatenating
        remaining_sample = remaining_sample.drop('is_philosophy', axis=1)
        
        # Concatenate to create adjusted sample
        adjusted_sample = pd.concat([remaining_sample, replacement_philosophy], ignore_index=True)
        
        # Verify the result
        final_philosophy = int(adjusted_sample['category_names'].apply(is_philosophy_related).sum()) # type: ignore
        final_percentage = final_philosophy / len(adjusted_sample) * 100
        logger.info(f"Adjusted sample: {final_philosophy}/{len(adjusted_sample)} philosophy papers ({final_percentage:.1f}%)")
        
        return adjusted_sample # type: ignore
    
    return sampled_copy.drop('is_philosophy', axis=1)


def draw_samples(
    df: pd.DataFrame, cluster_allocation: Dict[str, int], args, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draw samples from each cluster with internal stratification.
    Ensures >=50% philosophy-related papers in final sample.

    Args:
        df: DataFrame with stratification columns
        cluster_allocation: Dictionary mapping cluster_id to allocated sample count
        args: Command line arguments with repeat flags
        logger: Logger instance

    Returns:
        Tuple of (sampled DataFrame, suggestions DataFrame)
    """
    logger.info("Drawing samples...")

    # Set random seed
    np.random.seed(args.seed)

    # Track used values to prevent repeats
    used_authors = set()
    used_years = set()
    used_categories = set()

    all_sampled_papers = []
    all_suggestions = []

    # Mark philosophy-related papers
    df['is_philosophy'] = df['category_names'].apply(is_philosophy_related)
    philosophy_count = df['is_philosophy'].sum()
    logger.info(f"Philosophy-related papers available: {philosophy_count}/{len(df)} ({philosophy_count/len(df)*100:.1f}%)")

    for cluster_id, max_samples in cluster_allocation.items():
        if max_samples <= 0:
            continue

        logger.info(f"Sampling from cluster {cluster_id}: {max_samples} samples")

        cluster_df = df[df["cluster"] == cluster_id].copy()

        if len(cluster_df) == 0:
            logger.warning(f"No papers in cluster {cluster_id}")
            continue

        # Sample within this cluster using stratification
        cluster_samples = sample_within_cluster(
            cluster_df,  # type: ignore
            max_samples,
            used_authors,
            used_years,
            used_categories,
            args,
            logger,
        )

        all_sampled_papers.extend(cluster_samples)

        # Check for shortfall and generate suggestions
        actual_count = len(cluster_samples)
        if actual_count < max_samples:
            shortage = max_samples - actual_count
            logger.info(f"Cluster {cluster_id} shortage: {shortage} samples")

            # Generate 2 suggestions per missing sample
            suggestions = generate_suggestions(
                df,
                cluster_id,
                shortage * 2,
                used_authors,
                used_years,
                used_categories,
                args,
                logger,
            )
            all_suggestions.extend(suggestions)

        # Update used sets
        for paper in cluster_samples:
            if not args.allow_author_repeats:
                used_authors.update(extract_authors(paper.get("authors", "")))

            if not args.allow_year_repeats:
                paper_year = paper.get("year_clean")
                if paper_year is not None and not pd.isna(paper_year):
                    used_years.add(paper_year)

            if not args.allow_category_repeats:
                used_categories.add(paper.get("category_clean"))

    # Create result DataFrames
    if all_sampled_papers:
        result_df = pd.DataFrame(all_sampled_papers)
        
        # Check philosophy percentage and adjust if needed
        philosophy_sampled = sum(1 for paper in all_sampled_papers if is_philosophy_related(paper.get('category_names', '')))
        philosophy_percentage = philosophy_sampled / len(result_df) * 100
        
        logger.info(f"Philosophy papers in sample: {philosophy_sampled}/{len(result_df)} ({philosophy_percentage:.1f}%)")
        
        if philosophy_percentage < 50:
            logger.info("Adjusting sample to ensure >=50% philosophy-related papers...")
            result_df = adjust_for_philosophy_requirement(result_df, df, args, logger)
        
        logger.info(f"Successfully sampled {len(result_df)} papers")
    else:
        logger.error("No papers were sampled!")
        result_df = pd.DataFrame()

    if all_suggestions:
        suggestions_df = pd.DataFrame(all_suggestions)
        logger.info(f"Generated {len(suggestions_df)} suggestions for missing samples")
    else:
        suggestions_df = pd.DataFrame()

    return result_df, suggestions_df


def sample_within_cluster(
    cluster_df: pd.DataFrame,
    max_samples: int,
    used_authors: set,
    used_years: set,
    used_categories: set,
    args,
    logger: logging.Logger,
) -> List:
    """
    Sample within a cluster using quantile/category stratification.

    Args:
        cluster_df: DataFrame for this cluster
        max_samples: Maximum samples allowed for this cluster
        used_authors/years/categories: Sets of already used values
        args: Command line arguments
        logger: Logger instance

    Returns:
        List of sampled paper dictionaries
    """
    # Step 1: Try to allocate across quantiles proportionally
    quantile_groups = cluster_df.groupby("year_quantile")
    quantile_allocation = {}

    for quantile, group in quantile_groups:
        proportion = len(group) / len(cluster_df)
        allocated = round(proportion * max_samples)
        quantile_allocation[quantile] = allocated

    # Adjust for rounding errors
    total_allocated = sum(quantile_allocation.values())
    diff = max_samples - total_allocated

    if diff != 0:
        # Adjust largest quantile
        largest_quantile = max(
            quantile_allocation.keys(),
            key=lambda q: len(cluster_df[cluster_df["year_quantile"] == q]),
        )
        quantile_allocation[largest_quantile] += diff

    # Step 2: Sample from each quantile
    sampled_papers = []

    for quantile, target_count in quantile_allocation.items():
        if target_count <= 0:
            continue

        quantile_df = cluster_df[cluster_df["year_quantile"] == quantile]

        # Sample from this quantile with category diversity
        quantile_samples = sample_within_quantile(
            quantile_df, target_count, used_authors, used_years, used_categories, args  # type: ignore
        )

        sampled_papers.extend(quantile_samples)

    return sampled_papers


def sample_within_quantile(
    quantile_df: pd.DataFrame,
    target_count: int,
    used_authors: set,
    used_years: set,
    used_categories: set,
    args,
) -> List:
    """
    Sample within a quantile trying to diversify across categories.

    Args:
        quantile_df: DataFrame for this quantile
        target_count: Target number of samples
        used_authors/years/categories: Sets of already used values
        args: Command line arguments

    Returns:
        List of sampled paper dictionaries
    """
    if len(quantile_df) == 0:
        return []

    # Get unique categories in this quantile
    category_groups = quantile_df.groupby("category_clean")
    categories = list(category_groups.groups.keys())

    sampled_papers = []
    samples_drawn = 0
    attempts = 0
    max_attempts = target_count * 20  # Prevent infinite loops

    # Track constraint violations for diagnostics
    author_violations = 0
    year_violations = 0
    category_violations = 0

    while samples_drawn < target_count and attempts < max_attempts:
        attempts += 1

        # Try to sample from different categories cyclically for diversity
        category_idx = samples_drawn % len(categories)
        category = categories[category_idx]

        category_df = quantile_df[quantile_df["category_clean"] == category]

        if len(category_df) == 0:
            continue

        # Randomly select a paper from this category
        paper_idx = np.random.choice(len(category_df))
        paper = category_df.iloc[paper_idx]

        # Check constraints
        if not check_constraints(
            paper, used_authors, used_years, used_categories, args
        ):
            # Track violations for diagnostics
            if not args.allow_author_repeats:
                paper_authors = extract_authors(paper.get("authors", ""))
                if paper_authors.intersection(used_authors):
                    author_violations += 1
            if not args.allow_year_repeats:
                paper_year = paper.get("year_clean")
                if (
                    paper_year is not None
                    and not pd.isna(paper_year)
                    and paper_year in used_years
                ):
                    year_violations += 1
            if not args.allow_category_repeats:
                paper_category = paper.get("category_clean")
                if paper_category in used_categories:
                    category_violations += 1
            continue

        # Accept this paper
        sampled_papers.append(paper.to_dict())
        samples_drawn += 1

    # Log diagnostic information if we didn't reach the target
    if samples_drawn < target_count:
        print(
            f"Warning: Only sampled {samples_drawn}/{target_count} papers from quantile"
        )
        print(
            f"  Constraint violations - Authors: {author_violations}, Years: {year_violations}, Categories: {category_violations}"
        )
        print(f"  Attempts made: {attempts}/{max_attempts}")

    return sampled_papers


def main():
    """Main execution function."""
    # Setup
    root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
    args = parse_args()

    # Create output directory and setup logging
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info("=== DATASET SAMPLING ===")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Allow author repeats: {args.allow_author_repeats}")
    logger.info(f"Allow year repeats: {args.allow_year_repeats}")
    logger.info(f"Allow category repeats: {args.allow_category_repeats}")

    # Load data
    logger.info("Loading dataset...")
    input_path = root / args.input_file

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} papers")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Validate required columns
    required_columns = ["title", "year", "category_names", "authors", "identifier"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        sys.exit(1)

    # Filter papers
    df_filtered = filter_papers(df, logger)

    # Cluster titles
    df_clustered, n_clusters = cluster_titles(df_filtered, logger)

    # Stratify within clusters
    df_stratified = stratify_within_clusters(df_clustered, logger)

    # Allocate samples
    cluster_allocation = allocate_samples(df_stratified, args.sample_size, logger)

    # Draw samples
    sampled_df, suggestions_df = draw_samples(
        df_stratified, cluster_allocation, args, logger
    )

    if len(sampled_df) == 0:
        logger.error("Sampling failed - no papers selected")
        sys.exit(1)

    # Check if we got fewer samples than requested
    if len(sampled_df) < args.sample_size:
        logger.warning(
            f"Only {len(sampled_df)} papers sampled out of {args.sample_size} requested"
        )
        if len(suggestions_df) > 0:
            logger.warning(f"Generated {len(suggestions_df)} suggestions to fill gaps")

    # Prepare output
    output_columns = ["identifier", "cluster|year|category"]

    # Create the combined stratum column for output
    sampled_df["cluster|year|category"] = (
        sampled_df["cluster"].astype(str)
        + "|"
        + sampled_df["year_quantile"].astype(str)
        + "|"
        + sampled_df["category_clean"].astype(str)
    )

    # Select output columns
    output_df = sampled_df[output_columns].copy()

    # Save results
    output_file = output_dir / "sample.csv"
    output_df.to_csv(output_file, index=False)
    logger.info(f"Sample saved to: {output_file}")

    # Save stratified dataset for sampling-replace.py
    stratified_file = output_dir / "stratified_dataset.csv"
    df_stratified.to_csv(stratified_file, index=False)
    logger.info(f"Stratified dataset saved to: {stratified_file}")

    # Save suggestions if any
    if len(suggestions_df) > 0:
        suggestions_df["cluster|year|category"] = (
            suggestions_df["cluster"].astype(str)
            + "|"
            + suggestions_df["year_quantile"].astype(str)
            + "|"
            + suggestions_df["category_clean"].astype(str)
        )
        suggestions_output_columns = [
            "identifier",
            "cluster|year|category",
            "suggestion_reason",
        ]
        suggestions_output_df = suggestions_df[suggestions_output_columns].copy()

        suggestions_file = output_dir / "suggestions.csv"
        suggestions_output_df.to_csv(suggestions_file, index=False)
        logger.info(f"Suggestions saved to: {suggestions_file}")

    # Print summary
    logger.info("\n=== SAMPLING SUMMARY ===")
    logger.info(f"Total papers in dataset: {len(df):,}")
    logger.info(f"Papers sampled: {len(sampled_df)}")
    logger.info(f"Sampling rate: {len(sampled_df)/len(df)*100:.2f}%")
    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Random seed used: {args.seed}")

    if len(suggestions_df) > 0:
        logger.info(f"Suggestions generated: {len(suggestions_df)}")

    # Cluster distribution in sample
    if "cluster" in sampled_df.columns:
        cluster_dist = sampled_df["cluster"].value_counts()
        logger.info(f"Sample spans {len(cluster_dist)} clusters")

    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
