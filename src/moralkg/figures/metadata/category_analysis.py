#!/usr/bin/env python3
"""
Category analysis for philosophy and ethics terms in the PhilPapers dataset.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ...config import Config

logger = logging.getLogger(__name__)


def analyze_categories_from_json(json_path: Path) -> Dict[str, Any]:
    """
    Analyze categories containing philosophy/ethics-related terms from JSON file.
    
    Args:
        json_path: Path to categories JSON file
        
    Returns:
        Dictionary with category statistics
    """
    logger.info(f"Analyzing categories in: {json_path}")
    
    if not json_path.exists():
        return {"error": f"JSON file not found at {json_path}"}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract category names
    category_names = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) > 0:
                category_names.append(item[0])
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                category_names.append(value)
            elif isinstance(value, list) and len(value) > 0:
                category_names.append(value[0])
    
    if not category_names:
        return {"error": "No categories found in JSON file"}
    
    total_categories = len(category_names)
    
    # Pattern for morality/ethics terms
    pattern = r"(ethic|moral|virtue|deontolog)"
    matching_categories = [
        cat for cat in category_names 
        if re.search(pattern, cat, re.IGNORECASE)
    ]
    matching_count = len(matching_categories)
    
    percentage = (matching_count / total_categories * 100) if total_categories > 0 else 0.0
    
    return {
        "total_categories": total_categories,
        "matching_categories": matching_count,
        "percentage": percentage,
        "examples": matching_categories[:1000],
        "pattern": pattern
    }


def analyze_categories_from_dataframe(exploded_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze categories containing philosophy/ethics-related terms from exploded dataframe.
    
    Args:
        exploded_df: DataFrame with exploded category_names column
        
    Returns:
        Dictionary with detailed category and paper statistics
    """
    if "category_names" not in exploded_df.columns:
        return {"error": "category_names column not found"}

    cat_series = exploded_df["category_names"].dropna()
    cat_series = cat_series[cat_series.str.strip() != ""]

    if cat_series.empty:
        return {"error": "No valid categories found"}

    unique_categories = cat_series.unique()
    total_categories = len(unique_categories)

    # Philosophy/ethics pattern
    phil_ethics_pattern = r"(philosophy|ethic|moral|value|virtue)"

    # Find matching categories
    matching_mask = (
        pd.Series(unique_categories)
        .str.lower()
        .str.contains(phil_ethics_pattern, case=False, na=False)
    )
    matching_categories = pd.Series(unique_categories)[matching_mask].tolist()
    matching_count = len(matching_categories)

    percentage = (matching_count / total_categories * 100) if total_categories > 0 else 0.0

    # Count papers for each group
    phil_ethics_papers = 0
    if "identifier" in exploded_df.columns:
        phil_ethics_papers = exploded_df[
            exploded_df["category_names"]
            .str.lower()
            .str.contains(phil_ethics_pattern, case=False, na=False)
        ]["identifier"].nunique()

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
        "phil_ethics_category_examples": matching_categories[:20],
        "phil_ethics_papers": int(phil_ethics_papers),
        "other_papers": int(other_papers),
        "total_papers_with_categories": int(total_papers_with_categories),
    }


def create_category_plots(stats: Dict[str, Any], output_dir: Path) -> None:
    """Create visualizations for philosophy/ethics category analysis."""
    
    if "error" in stats:
        logger.warning(f"Skipping category plots: {stats['error']}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Philosophy/Ethics Categories Analysis", fontsize=16, fontweight="bold")
    
    # Categories pie chart
    if "phil_ethics_categories" in stats:
        category_labels = ["Philosophy/Ethics", "Other"]
        category_counts = [
            stats["phil_ethics_categories"],
            stats["other_categories"],
        ]
        
        ax1.pie(category_counts, labels=category_labels, autopct="%1.1f%%",
                colors=["lightcoral", "lightblue"], startangle=90)
        ax1.set_title(f'Categories Distribution\n({stats["total_categories"]} total)')
    
    # Papers pie chart
    if "phil_ethics_papers" in stats:
        papers_labels = ["Papers with Phil/Ethics", "Papers with Other"]
        papers_counts = [stats["phil_ethics_papers"], stats["other_papers"]]
        
        ax2.pie(papers_counts, labels=papers_labels, autopct="%1.1f%%",
                colors=["lightcoral", "lightblue"], startangle=90)
        ax2.set_title(f'Papers Distribution\n({stats["total_papers_with_categories"]} total)')
    
    plt.tight_layout()
    output_path = output_dir / "philosophy_ethics_categories.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Category plots saved to: {output_path}")
    
    # Save text summary
    text_path = output_dir / "philosophy_ethics_categories.txt"
    with open(text_path, "w") as f:
        f.write("Philosophy/Ethics Categories Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        if "total_categories" in stats:
            f.write(f"Total unique categories: {stats['total_categories']}\n")
        
        if "phil_ethics_categories" in stats:
            f.write(f"Philosophy/Ethics categories: {stats['phil_ethics_categories']} ")
            f.write(f"({stats.get('phil_ethics_percentage', 0):.2f}%)\n")
            f.write(f"Other categories: {stats['other_categories']}\n\n")
        
        if "phil_ethics_papers" in stats:
            f.write(f"Total papers with categories: {stats['total_papers_with_categories']}\n")
            f.write(f"Papers with Phil/Ethics categories: {stats['phil_ethics_papers']}\n")
            f.write(f"Papers with other categories: {stats['other_papers']}\n\n")
        
        if "phil_ethics_category_examples" in stats:
            f.write("Example Philosophy/Ethics Categories:\n")
            f.write("-" * 40 + "\n")
            for i, category in enumerate(stats["phil_ethics_category_examples"], 1):
                f.write(f"{i:2d}. {category}\n")