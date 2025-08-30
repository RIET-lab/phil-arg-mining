#!/usr/bin/env python3
"""
Paper content statistics including length and word count analysis.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ...config import Config

logger = logging.getLogger(__name__)


def analyze_paper_lengths_simple(docling_dir: Path) -> Dict[str, Any]:
    """
    Simple paper length analysis with word count thresholds.
    
    Args:
        docling_dir: Path to directory containing docling processed files
        
    Returns:
        Dictionary with word count statistics and threshold analysis
    """
    logger.info(f"Analyzing paper lengths in: {docling_dir}")
    
    if not docling_dir.exists():
        return {"error": f"Docling directory not found at {docling_dir}"}
    
    # Find all unique identifiers
    identifiers = set()
    for ext in ['*.md', '*.txt']:
        for file_path in docling_dir.glob(ext):
            if not file_path.name.endswith('.doctags.txt'):
                identifier = file_path.stem
                identifiers.add(identifier)
    
    if not identifiers:
        return {"error": "No papers found in docling directory"}
    
    # Process each paper
    word_counts = []
    paper_data = []
    
    for identifier in identifiers:
        # Look for .md file first, then .txt file
        content_file = None
        for ext in ['.md', '.txt']:
            candidate = docling_dir / f"{identifier}{ext}"
            if candidate.exists():
                content_file = candidate
                break
        
        if content_file:
            try:
                with open(content_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                word_count = len(re.findall(r'\w+', content))
                word_counts.append(word_count)
                paper_data.append((identifier, word_count))
            except Exception as e:
                logger.warning(f"Error reading {content_file}: {e}")
    
    if not word_counts:
        return {"error": "No valid content found"}
    
    # Calculate statistics
    word_array = np.array(word_counts)
    total_papers = len(word_counts)
    
    papers_over_100 = np.sum(word_array > 100)
    papers_over_500 = np.sum(word_array > 500)
    papers_over_1000 = np.sum(word_array > 1000)
    
    # Sort paper data for examples
    paper_data.sort(key=lambda x: x[1])
    
    return {
        "total_papers": total_papers,
        "thresholds": {
            "over_100": {
                "count": int(papers_over_100),
                "percentage": float(papers_over_100 / total_papers * 100)
            },
            "over_500": {
                "count": int(papers_over_500),
                "percentage": float(papers_over_500 / total_papers * 100)
            },
            "over_1000": {
                "count": int(papers_over_1000),
                "percentage": float(papers_over_1000 / total_papers * 100)
            }
        },
        "statistics": {
            "mean": float(np.mean(word_array)),
            "median": float(np.median(word_array)),
            "std": float(np.std(word_array)),
            "min": int(np.min(word_array)),
            "max": int(np.max(word_array))
        },
        "examples": {
            "under_100": [(id, wc) for id, wc in paper_data if wc <= 100][:10],
            "100_to_500": [(id, wc) for id, wc in paper_data if 100 < wc <= 500][:10],
            "500_to_1000": [(id, wc) for id, wc in paper_data if 500 < wc <= 1000][:10],
            "over_1000": [(id, wc) for id, wc in paper_data if wc > 1000][:10]
        },
        "word_counts": word_counts
    }


def read_file_content(file_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read content from a single file and return character and word counts."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            word_count = len(re.findall(r"\w+", content))
            return len(content), word_count
    except Exception:
        pass
    return None, None


def analyze_paper_content_lengths(df: pd.DataFrame, docling_dir: Path) -> Dict[str, Any]:
    """
    Detailed paper content analysis with parallel processing.
    Includes both character and word counts.
    
    Args:
        df: DataFrame with paper metadata
        docling_dir: Path to directory containing docling processed files
        
    Returns:
        Dictionary with detailed content statistics
    """
    if not docling_dir or not docling_dir.exists():
        return {"error": "Docling directory not found or not specified"}

    # Scan docling files
    docling_files = set()
    for ext in ["*.txt", "*.md"]:
        for file_path in docling_dir.glob(ext):
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
    identifiers_with_files = df_clean[df_clean["has_docling"]]["identifier_str"].tolist()

    logger.info(f"Reading content from {len(identifiers_with_files)} docling files...")

    # Parallel file reading
    char_lengths = []
    word_lengths = []
    batch_size = 500

    for i in range(0, len(identifiers_with_files), batch_size):
        batch = identifiers_with_files[i : i + batch_size]
        batch_paths = []

        # Find actual file paths for this batch
        for identifier in batch:
            file_path = None
            for ext in [".txt", ".md"]:
                candidate = docling_dir / f"{identifier}{ext}"
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

    if not char_lengths:
        return {"error": "No valid content found in docling files"}

    # Statistics
    char_array = np.array(char_lengths)
    word_array = np.array(word_lengths)

    return {
        "papers_with_content": papers_with_content,
        "papers_without_content": papers_without_content,
        "availability_rate": float(papers_with_content / len(df)),
        "char_lengths": char_lengths,
        "char_mean": float(np.mean(char_array)),
        "char_median": float(np.median(char_array)),
        "char_std": float(np.std(char_array)),
        "char_min": int(np.min(char_array)),
        "char_max": int(np.max(char_array)),
        "word_lengths": word_lengths,
        "word_mean": float(np.mean(word_array)),
        "word_median": float(np.median(word_array)),
        "word_std": float(np.std(word_array)),
        "word_min": int(np.min(word_array)),
        "word_max": int(np.max(word_array)),
    }


def create_content_plots(stats: Dict[str, Any], output_dir: Path, plot_type: str = "simple") -> None:
    """
    Create plots for paper content analysis.
    
    Args:
        stats: Dictionary with content statistics
        output_dir: Path to output directory
        plot_type: Type of plot ("simple" or "detailed")
    """
    if "error" in stats:
        logger.warning(f"Skipping content plots: {stats['error']}")
        return
    
    if plot_type == "simple" and "word_counts" in stats:
        # Simple analysis plots
        word_counts = stats['word_counts']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of word counts
        ax1.hist(word_counts, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Word Count')
        ax1.set_ylabel('Number of Papers')
        ax1.set_title('Distribution of Paper Lengths')
        ax1.axvline(stats['statistics']['mean'], color='red', linestyle='--', 
                    label=f"Mean: {stats['statistics']['mean']:.0f}")
        ax1.axvline(stats['statistics']['median'], color='green', linestyle='--',
                    label=f"Median: {stats['statistics']['median']:.0f}")
        ax1.legend()
        ax1.set_yscale('log')
        
        # Bar chart of threshold percentages
        thresholds = stats['thresholds']
        labels = ['> 100 words', '> 500 words', '> 1000 words']
        percentages = [
            thresholds['over_100']['percentage'],
            thresholds['over_500']['percentage'],
            thresholds['over_1000']['percentage']
        ]
        
        ax2.bar(labels, percentages)
        ax2.set_ylabel('Percentage of Papers (%)')
        ax2.set_title('Papers Meeting Word Count Thresholds')
        for i, (label, pct) in enumerate(zip(labels, percentages)):
            ax2.text(i, pct + 1, f'{pct:.1f}%', ha='center')
        
        plt.tight_layout()
        output_path = output_dir / 'paper-length-analysis.png'
        
    elif plot_type == "detailed" and "char_lengths" in stats:
        # Detailed analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Paper Content Length Analysis', fontsize=16, fontweight='bold')
        
        # Character length histogram
        char_data = stats["char_lengths"]
        ax1.hist(char_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_yscale('log')
        ax1.set_title('Paper Length Distribution (Characters)')
        ax1.set_xlabel('Number of Characters')
        ax1.set_ylabel('Number of Papers')
        ax1.grid(True, alpha=0.3)
        
        # Word length histogram
        word_data = stats["word_lengths"]
        ax2.hist(word_data, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_yscale('log')
        ax2.set_title('Paper Length Distribution (Words)')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Number of Papers')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'paper-content-detailed.png'
    
    else:
        logger.warning("Invalid plot type or missing data for content plots")
        return
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Content plots saved to: {output_path}")