#!/usr/bin/env python3
"""
Category analysis for philosophy and ethics terms in the PhilPapers dataset.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from ...config import Config
from . import get_most_recent_file

logger = logging.getLogger(__name__)

MORALITY_ETHICS_TERMS = ["ethic", "moral", "virtue", "deontolog", "utilitar", "justice"]
# Single-pass pattern with capture groups for efficient term extraction
MORALITY_ETHICS_PATTERN = r"(ethic)|(moral)|(virtue)|(deontolog)|(utilitar)|(justice)"

def extract_matched_terms(text: str) -> List[str]:
    """
    Extract which specific terms from the pattern match in the given text using single-pass regex.
    
    Args:
        text: Text to search in
        
    Returns:
        List of matched terms (unique, preserving order of appearance)
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    matches = []
    text_lower = text.lower()
    
    # Single pass with finditer to catch all occurrences - O(1) regex compilation + O(text_length)
    for match in re.finditer(MORALITY_ETHICS_PATTERN, text_lower):
        for i, group in enumerate(match.groups()):
            if group and MORALITY_ETHICS_TERMS[i] not in matches:
                matches.append(MORALITY_ETHICS_TERMS[i])
    
    return matches


def analyze_term_combinations(matched_terms_list: List[List[str]]) -> Dict[str, int]:
    """
    Analyze combinations of matched terms.
    
    Args:
        matched_terms_list: List of lists, each containing matched terms for a category
        
    Returns:
        Dictionary with term combinations and their counts
    """
    # Count individual terms
    individual_counts = Counter()
    for terms in matched_terms_list:
        for term in terms:
            individual_counts[term] += 1
    
    # Count combinations (for categories that match multiple terms)
    combination_counts = Counter()
    for terms in matched_terms_list:
        if len(terms) > 1:
            # Sort terms for consistent combination representation
            combo = " + ".join(sorted(terms))
            combination_counts[combo] += 1
    
    # Combine results
    results = dict(individual_counts)
    results.update({f"Combo: {k}": v for k, v in combination_counts.items()})
    
    return results


def analyze_categories_from_json(json_path: Path) -> Dict[str, Any]:
    """
    Analyze categories containing morality/ethics-related terms from JSON file.
    
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
    pattern = MORALITY_ETHICS_PATTERN
    matching_categories = [
        cat for cat in category_names 
        if re.search(pattern, cat, re.IGNORECASE)
    ]
    matching_count = len(matching_categories)
    
    # Extract matched terms for each matching category
    matched_terms_list = [extract_matched_terms(cat) for cat in matching_categories]
    term_analysis = analyze_term_combinations(matched_terms_list)
    
    percentage = (matching_count / total_categories * 100) if total_categories > 0 else 0.0
    
    return {
        "total_categories": total_categories,
        "matching_categories": matching_count,
        "percentage": percentage,
        "examples": matching_categories[:1000],
        "pattern": pattern,
        "matched_terms_analysis": term_analysis,
        "matched_terms_list": matched_terms_list
    }


def analyze_categories_from_dataframe(exploded_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze categories containing morality/ethics-related terms from exploded dataframe.
    
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

    # Find matching categories
    matching_mask = (
        pd.Series(unique_categories)
        .str.lower()
        .str.contains(MORALITY_ETHICS_PATTERN, case=False, na=False)
    )
    matching_categories = pd.Series(unique_categories)[matching_mask].tolist()
    matching_count = len(matching_categories)

    # Extract matched terms for each matching category
    matched_terms_list = [extract_matched_terms(cat) for cat in matching_categories]
    term_analysis = analyze_term_combinations(matched_terms_list)

    percentage = (matching_count / total_categories * 100) if total_categories > 0 else 0.0

    # Count papers for each group
    morality_ethics_papers = 0
    if "identifier" in exploded_df.columns:
        morality_ethics_papers = exploded_df[
            exploded_df["category_names"]
            .str.lower()
            .str.contains(MORALITY_ETHICS_PATTERN, case=False, na=False)
        ]["identifier"].nunique()

    total_papers_with_categories = (
        exploded_df["identifier"].nunique()
        if "identifier" in exploded_df.columns
        else len(exploded_df)
    )
    other_papers = total_papers_with_categories - morality_ethics_papers

    # Analyze term frequency across all category instances (not just unique)
    all_matching_categories = exploded_df[
        exploded_df["category_names"]
        .str.lower()
        .str.contains(MORALITY_ETHICS_PATTERN, case=False, na=False)
    ]["category_names"].tolist()
    
    all_matched_terms_list = [extract_matched_terms(cat) for cat in all_matching_categories]
    all_term_analysis = analyze_term_combinations(all_matched_terms_list)

    return {
        "total_categories": int(total_categories),
        "morality_ethics_categories": int(matching_count),
        "other_categories": int(total_categories - matching_count),
        "morality_ethics_percentage": float(percentage),
        "morality_ethics_category_examples": matching_categories[:20],
        "morality_ethics_papers": int(morality_ethics_papers),
        "other_papers": int(other_papers),
        "total_papers_with_categories": int(total_papers_with_categories),
        "matched_terms_analysis": term_analysis,
        "all_matched_terms_list": all_matched_terms_list,
        "all_matched_terms_analysis": all_term_analysis,
        "matched_terms_list": matched_terms_list
    }


def create_term_frequency_plot(term_analysis: Dict[str, int], output_dir: Path, 
                              title_suffix: str = "") -> None:
    """
    Create a bar plot showing frequency of matched morality/ethics terms.
    
    Args:
        term_analysis: Dictionary with term counts
        output_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    if not term_analysis:
        logger.warning("No term analysis data available for plotting")
        return
    
    # Separate individual terms from combinations
    individual_terms = {k: v for k, v in term_analysis.items() if not k.startswith("Combo:")}
    combinations = {k.replace("Combo: ", ""): v for k, v in term_analysis.items() if k.startswith("Combo:")}
    
    fig, axes = plt.subplots(1, 2 if combinations else 1, figsize=(15 if combinations else 10, 6))
    if not combinations:
        axes = [axes]
    
    fig.suptitle(f"Morality/Ethics Terms Analysis{title_suffix}", fontsize=16, fontweight="bold")
    
    # Plot individual terms
    if individual_terms:
        terms = list(individual_terms.keys())
        counts = list(individual_terms.values())

        # Sort by frequency
        sorted_pairs = sorted(zip(terms, counts), key=lambda x: x[1], reverse=True)
        terms, counts = zip(*sorted_pairs)
        
        bars = axes[0].bar(terms, counts, color='lightcoral', alpha=0.7)
        axes[0].set_title('Individual Terms Frequency')
        axes[0].set_xlabel('Terms')
        axes[0].set_ylabel('Frequency')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    # Plot combinations if they exist
    if combinations:
        combo_terms = list(combinations.keys())
        combo_counts = list(combinations.values())
        
        # Sort by frequency - ensure we have lists before zipping
        sorted_combo_pairs = sorted(list(zip(combo_terms, combo_counts)), key=lambda x: x[1], reverse=True)
        combo_terms, combo_counts = zip(*sorted_combo_pairs)
        
        bars = axes[1].bar(range(len(combo_terms)), combo_counts, color='lightblue', alpha=0.7)
        axes[1].set_title('Term Combinations Frequency')
        axes[1].set_xlabel('Combinations')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xticks(range(len(combo_terms)))
        axes[1].set_xticklabels(combo_terms, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / f"morality_ethics_terms_frequency{title_suffix.replace(' ', '_').lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Term frequency plot saved to: {output_path}")


def create_category_plots(stats: Dict[str, Any], output_dir: Path) -> None:
    """Create visualizations for morality/ethics category analysis.
    TODO: Make this work even if only JSON analysis is done (no dataframe).
    """
    
    if "error" in stats:
        logger.warning(f"Skipping category plots: {stats['error']}")
        return
    
    # Create the original pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Morality/Ethics Categories Analysis", fontsize=16, fontweight="bold")
    
    # Categories pie chart
    if "morality_ethics_categories" in stats:
        category_labels = ["Morality/Ethics", "Other"]
        category_counts = [
            stats["morality_ethics_categories"],
            stats["other_categories"],
        ]
        
        ax1.pie(category_counts, labels=category_labels, autopct="%1.1f%%",
                colors=["lightcoral", "lightblue"], startangle=90)
        ax1.set_title(f'Categories Distribution\n({stats["total_categories"]} total)')
    
    # Papers pie chart
    if "morality_ethics_papers" in stats:
        papers_labels = ["Papers with Morality/Ethics", "Papers with Other"]
        papers_counts = [stats["morality_ethics_papers"], stats["other_papers"]]
        
        ax2.pie(papers_counts, labels=papers_labels, autopct="%1.1f%%",
                colors=["lightcoral", "lightblue"], startangle=90)
        ax2.set_title(f'Papers Distribution\n({stats["total_papers_with_categories"]} total)')
    
    plt.tight_layout()
    output_path = output_dir / "morality_ethics_categories.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Category plots saved to: {output_path}")
    
    # Create term frequency plots
    if "matched_terms_analysis" in stats:
        create_term_frequency_plot(stats["matched_terms_analysis"], output_dir, " (Unique Categories)")
    
    if "all_matched_terms_analysis" in stats:
        create_term_frequency_plot(stats["all_matched_terms_analysis"], output_dir, " (All Instances)")
    
    # Save text summary
    text_path = output_dir / "morality_ethics_categories.txt"
    with open(text_path, "w") as f:
        f.write("Morality/Ethics Categories Analysis\n")
        f.write("=" * 50 + "\n\n")

        # Print the pattern used to identify morality/ethics papers
        f.write(f"Morality/Ethics search pattern: {MORALITY_ETHICS_PATTERN}\n")
        f.write(f"Individual terms searched: {', '.join(MORALITY_ETHICS_TERMS)}\n\n")

        if "total_categories" in stats:
            f.write(f"Total unique categories: {stats['total_categories']}\n")
        
        if "morality_ethics_categories" in stats:
            f.write(f"Morality/Ethics categories: {stats['morality_ethics_categories']} ")
            f.write(f"({stats.get('morality_ethics_percentage', 0):.2f}%)\n")
            f.write(f"Other categories: {stats['other_categories']}\n\n")
        
        if "morality_ethics_papers" in stats:
            f.write(f"Total papers with categories: {stats['total_papers_with_categories']}\n")
            f.write(f"Papers with Morality/Ethics categories: {stats['morality_ethics_papers']}\n")
            f.write(f"Papers with other categories: {stats['other_papers']}\n\n")
        
        # Add term frequency analysis
        if "matched_terms_analysis" in stats:
            f.write("Term Frequency Analysis (Unique Categories)\n")
            f.write("-" * 45 + "\n")
            term_analysis = stats["matched_terms_analysis"]
            
            # Individual terms
            individual_terms = {k: v for k, v in term_analysis.items() if not k.startswith("Combo:")}
            if individual_terms:
                f.write("Individual Terms:\n")
                for term, count in sorted(individual_terms.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {term}: {count}\n")
                f.write("\n")
            
            # Combinations
            combinations = {k.replace("Combo: ", ""): v for k, v in term_analysis.items() if k.startswith("Combo:")}
            if combinations:
                f.write("Term Combinations:\n")
                for combo, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {combo}: {count}\n")
                f.write("\n")
        
        if "morality_ethics_category_examples" in stats:
            f.write("Example Morality/Ethics Categories:\n")
            f.write("-" * 40 + "\n")
            for i, category in enumerate(stats["morality_ethics_category_examples"], 1):
                # Show which terms matched for this category
                matched_terms = extract_matched_terms(category)
                terms_str = f" [{', '.join(matched_terms)}]" if matched_terms else ""
                f.write(f"{i:2d}. {category}{terms_str}\n")


def run_category_analysis(config=None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run category analysis using config to find category JSON file.
    TODO: Make this import or create a dataframe of preprocessed data so that the plots work correctly.
    
    Args:
        config: Optional Config instance (creates default if not provided)
        output_dir: Optional output directory (uses config default if not provided)
    
    Returns:
        Dictionary with analysis results
    """
    if config is None:
        config = Config.load()
    
    # Get paths from config
    metadata_dir = Path(config.get('paths.philpapers.metadata'))
    
    if output_dir is None:
        output_dir = Path(config.get('paths.figures.philpapers', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the json file
    try:
        categories_file = get_most_recent_file(metadata_dir, "*categories*.json")
        if categories_file:
            stats = analyze_categories_from_json(categories_file)
        else:
            logger.error("No categories JSON file found in metadata directory")
            return {"error": "No categories JSON file found"}
    except Exception as e:
        logger.error(f"Failed to find categories file: {e}")
        return {"error": f"Failed to find categories file: {e}"}

    if "error" not in stats:
        # Create visualizations
        create_category_plots(stats, output_dir)
        logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return stats


def main():
    """
    Main function to run the category analysis.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze categories for mentions of morality and ethics")
    parser.add_argument("--json", type=str, help="Path to categories JSON file (overrides config)")
    parser.add_argument("--output", type=str, 
                       help="Output directory for plots and reports (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    config = Config.load()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    elif config:
        output_dir = Path(config.get('paths.figures.philpapers', './output'))
    else:
        output_dir = Path('./output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze based on input type
    if args.json:
        # Use explicit JSON path
        logger.info(f"Analyzing categories from JSON file: {args.json}")
        stats = analyze_categories_from_json(Path(args.json))
    else:
        # Use config to find categories file
        logger.info("Using config to find categories JSON file")
        stats = run_category_analysis(config, output_dir)
    
    # Check for analysis errors
    if "error" in stats:
        logger.error(f"Analysis failed: {stats['error']}")
        sys.exit(1)
    
    # Print summary to console
    print("\n" + "="*60)
    print("MORALITY/ETHICS CATEGORIES ANALYSIS RESULTS")
    print("="*60)
    
    if "total_categories" in stats:
        print(f"Total categories analyzed: {stats['total_categories']:,}")
    
    if "morality_ethics_categories" in stats:
        print(f"Morality/Ethics categories: {stats['morality_ethics_categories']:,} "
              f"({stats.get('morality_ethics_percentage', 0):.1f}%)")
    
    if "morality_ethics_papers" in stats:
        print(f"Papers with M/E categories: {stats['morality_ethics_papers']:,}")
        print(f"Total papers: {stats['total_papers_with_categories']:,}")
    
    # Show term frequencies
    if "matched_terms_analysis" in stats:
        print(f"\nTop matched terms:")
        term_analysis = stats["matched_terms_analysis"]
        individual_terms = {k: v for k, v in term_analysis.items() if not k.startswith("Combo:")}
        for term, count in sorted(individual_terms.items(), key=lambda x: x[1], reverse=True):
            print(f"  {term}: {count}")
    
    # Create visualizations if we haven't already
    if args.json:
        logger.info("Creating visualizations and reports...")
        create_category_plots(stats, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    print("Generated files:")
    print("  - morality_ethics_categories.png (pie charts)")
    print("  - morality_ethics_terms_frequency_*.png (term analysis)")
    print("  - philosophy_ethics_categories.txt (detailed report)")
    print("="*60)


if __name__ == "__main__":
    main()