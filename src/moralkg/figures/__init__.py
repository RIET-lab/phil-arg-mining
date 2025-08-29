"""
Figure generation module for MoralKG project.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Import main analysis functions from metadata submodule
from .metadata import (
    analyze_categories_from_json,
    analyze_categories_from_dataframe,
    run_category_analysis,
    analyze_paper_lengths_simple,
    analyze_paper_content_lengths,
    run_metadata_distributions,
    get_most_recent_file,
    preprocess_data
)


def generate_all_figures(
    config=None, 
    sample_size: Optional[int] = None,
    include_content_analysis: bool = True
) -> Dict[str, Any]:
    """
    Generate all figures for the PhilPapers dataset.
    
    Args:
        config: Optional Config instance (creates default if not provided)
        sample_size: Optional sample size for testing
        include_content_analysis: Whether to include detailed content analysis (slower)
    
    Returns:
        Dictionary containing results from all analyses
    """
    from src.moralkg.config import Config
    import pandas as pd
    
    if config is None:
        config = Config.load()
    
    results = {}
    
    # Setup paths
    metadata_dir = Path(config.get('paths.philpapers.metadata'))
    docling_dir = Path(config.get('paths.philpapers.docling.cleaned'))
    output_dir = Path(config.get('paths.figures.philpapers'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and load metadata
    metadata_path = get_most_recent_file(metadata_dir, '*metadata*.csv')
    if not metadata_path:
        metadata_path = get_most_recent_file(metadata_dir, '*.csv')
    
    if metadata_path and metadata_path.exists():
        df = pd.read_csv(metadata_path)
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Preprocess data
        paper_df, exploded_df = preprocess_data(df)
        
        # Category analysis
        from .metadata import create_category_plots
        cat_stats = analyze_categories_from_dataframe(exploded_df)
        create_category_plots(cat_stats, output_dir)
        results['categories'] = cat_stats

        # Simple paper length analysis
        if docling_dir.exists():
            from .metadata import create_content_plots
            length_stats = analyze_paper_lengths_simple(docling_dir)
            create_content_plots(length_stats, output_dir, plot_type="simple")
            results['paper_lengths'] = length_stats
            
            # Detailed content analysis if requested
            if include_content_analysis:
                detailed_stats = analyze_paper_content_lengths(paper_df, docling_dir)
                create_content_plots(detailed_stats, output_dir, plot_type="detailed")
                results['content_detailed'] = detailed_stats
    
    # Run metadata distributions
    dist_results = run_metadata_distributions(config=config)
    results['distributions'] = dist_results
    
    return results


def quick_category_check(config=None) -> Dict[str, Any]:
    """
    Quick check of philosophy/ethics categories.
    
    Returns:
        Dictionary with basic category statistics
    """
    from src.moralkg.config import Config
    
    if config is None:
        config = Config.load()
    
    metadata_dir = Path(config.get('paths.philpapers.metadata'))
    
    # Look for categories JSON
    categories_path = get_most_recent_file(metadata_dir, '*categories*.json')
    
    if categories_path and categories_path.exists():
        return analyze_categories_from_json(categories_path)
    
    return {"error": "No categories file found"}


__all__ = [
    # Main functions
    'generate_all_figures',
    'quick_category_check',
    
    # Core analysis functions
    'analyze_categories_from_json',
    'analyze_categories_from_dataframe',
    'analyze_paper_lengths_simple',
    'analyze_paper_content_lengths',
    'run_metadata_distributions',
    'get_most_recent_file',
    'preprocess_data'
]