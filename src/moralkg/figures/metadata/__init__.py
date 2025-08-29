"""
Metadata analysis submodule for figure generation.
"""

from pathlib import Path
from typing import Optional

def get_most_recent_file(directory: Path, pattern: str = "*") -> Optional[Path]:
    """Find the most recent file matching pattern in directory."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

from .metadata_distributions import (
    preprocess_data,
    compute_field_distributions,
    compute_title_diversity,
    create_distribution_plots,
    run_metadata_distributions,
)

from .category_analysis import (
    analyze_categories_from_json,
    analyze_categories_from_dataframe,
    create_category_plots,
    create_term_frequency_plot,
    run_category_analysis
)

from .content_stats import (
    analyze_paper_lengths_simple,
    analyze_paper_content_lengths,
    create_content_plots
)


__all__ = [
    # File utilities
    'get_most_recent_file',

    # Category analysis
    'analyze_categories_from_json',
    'analyze_categories_from_dataframe',
    'create_term_frequency_plot',
    'create_category_plots',
    'run_category_analysis',

    # Content statistics
    'analyze_paper_lengths_simple',
    'analyze_paper_content_lengths',
    'create_content_plots',
    
    # Metadata distributions
    'preprocess_data',
    'compute_field_distributions',
    'compute_title_diversity',
    'create_distribution_plots',
    'run_metadata_distributions',
    'get_most_recent_file'
]