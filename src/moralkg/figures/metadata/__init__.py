"""
Metadata analysis submodule for figure generation.
"""

from .category_analysis import (
    analyze_categories_from_json,
    analyze_categories_from_dataframe,
    create_category_plots
)

from .content_stats import (
    analyze_paper_lengths_simple,
    analyze_paper_content_lengths,
    create_content_plots
)

from .metadata_distributions import (
    preprocess_data,
    compute_field_distributions,
    compute_title_diversity,
    create_distribution_plots,
    run_metadata_distributions,
    get_most_recent_file
)

__all__ = [
    # Category analysis
    'analyze_categories_from_json',
    'analyze_categories_from_dataframe',
    'create_category_plots',
    
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