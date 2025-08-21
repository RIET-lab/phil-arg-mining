#!/usr/bin/env python3
"""
Shared evaluation utilities for clustering and sampling methods.
Follows KISS principle to reduce code duplication across scripts.
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def setup_logging(output_dir: Path, script_name: str) -> logging.Logger:
    """Set up logging for evaluation metrics and timing."""
    log_file = output_dir / f"{script_name}_evaluation.log"
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def time_operation(logger: logging.Logger, operation_name: str):
    """Context manager for timing operations."""
    class Timer:
        def __enter__(self):
            self.start_time = time.time()
            logger.info(f"Starting {operation_name}...")
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start_time
            logger.info(f"Completed {operation_name} in {elapsed:.2f} seconds")
    
    return Timer()

def evaluate_clustering_quality(X, labels, logger: logging.Logger) -> Dict[str, float]:
    """Evaluate clustering quality using multiple metrics."""
    logger.info("Computing clustering quality metrics...")
    
    metrics = {}
    
    try:
        # Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(X, labels)
        metrics['davies_bouldin'] = db_score
        logger.info(f"Davies-Bouldin Index: {db_score:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute Davies-Bouldin score: {e}")
    
    try:
        # Calinski-Harabasz Index (higher is better)
        ch_score = calinski_harabasz_score(X, labels)
        metrics['calinski_harabasz'] = ch_score
        logger.info(f"Calinski-Harabasz Index: {ch_score:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute Calinski-Harabasz score: {e}")
    
    try:
        # Silhouette Score (higher is better, range [-1, 1])
        sil_score = silhouette_score(X, labels)
        metrics['silhouette'] = sil_score
        logger.info(f"Silhouette Score: {sil_score:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute Silhouette score: {e}")
    
    return metrics

def evaluate_representativeness(original_df: pd.DataFrame, sample_df: pd.DataFrame, 
                               logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate how representative the sample is of the original dataset."""
    logger.info("Computing representativeness metrics...")
    
    results = {}
    
    # Get numeric and categorical columns
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns
    
    # Test numeric columns with KS test and KL divergence
    for col in numeric_cols:
        if col in sample_df.columns:
            try:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = stats.ks_2samp(original_df[col].dropna(), 
                                                 sample_df[col].dropna())
                
                # Simple distribution comparison (avoid complex KL divergence)
                orig_mean, orig_std = original_df[col].mean(), original_df[col].std()
                samp_mean, samp_std = sample_df[col].mean(), sample_df[col].std()
                
                results[f'{col}_ks_stat'] = ks_stat
                results[f'{col}_ks_pval'] = ks_pval
                results[f'{col}_mean_diff'] = abs(orig_mean - samp_mean) / orig_std if orig_std > 0 else 0
                
                logger.info(f"{col} - KS test p-value: {ks_pval:.4f}, Mean diff: {results[f'{col}_mean_diff']:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not test {col}: {e}")
    
    # Test categorical columns with Chi-squared test
    for col in categorical_cols:
        if col in sample_df.columns:
            try:
                # Get value counts
                orig_counts = original_df[col].value_counts()
                samp_counts = sample_df[col].value_counts()
                
                # Align indices and compute expected frequencies
                common_values = orig_counts.index.intersection(samp_counts.index)
                if len(common_values) > 1:
                    orig_freq = orig_counts[common_values]
                    samp_freq = samp_counts[common_values]
                    
                    # Expected frequencies based on original proportions
                    total_sample = samp_freq.sum()
                    expected = orig_freq / orig_freq.sum() * total_sample
                    
                    # Chi-squared test
                    chi2_stat, chi2_pval = stats.chisquare(samp_freq, expected)
                    
                    results[f'{col}_chi2_stat'] = chi2_stat
                    results[f'{col}_chi2_pval'] = chi2_pval
                    
                    logger.info(f"{col} - Chi-squared test p-value: {chi2_pval:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not test categorical {col}: {e}")
    
    return results

def create_enhanced_mds_plot(mds_coords: np.ndarray, clusters: np.ndarray, 
                           sample_indices: np.ndarray, output_path: Path, 
                           title: str, logger: logging.Logger) -> None:
    """Create MDS plot highlighting the sampled points."""
    logger.info("Creating enhanced MDS visualization with samples highlighted...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot all points
    scatter_all = plt.scatter(mds_coords[:, 0], mds_coords[:, 1], 
                             c=clusters, cmap='tab10', alpha=0.3, s=20, 
                             label='All points')
    
    # Highlight sampled points
    sample_coords = mds_coords[sample_indices]
    sample_clusters = clusters[sample_indices]
    scatter_sample = plt.scatter(sample_coords[:, 0], sample_coords[:, 1], 
                               c=sample_clusters, cmap='tab10', alpha=0.9, s=80,
                               edgecolors='black', linewidths=1, label='Sampled points')
    
    plt.colorbar(scatter_all, label='Cluster')
    plt.title(f'{title} - MDS Visualization with Samples')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Enhanced MDS plot saved to {output_path}")

def save_evaluation_summary(clustering_metrics: Dict[str, float], 
                          representativeness_metrics: Dict[str, Any],
                          output_dir: Path, script_name: str,
                          logger: logging.Logger) -> None:
    """Save all evaluation metrics to a summary file."""
    summary_file = output_dir / f"{script_name}_evaluation_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary for {script_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CLUSTERING QUALITY METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in clustering_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nREPRESENTATIVENESS METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in representativeness_metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
    
    logger.info(f"Evaluation summary saved to {summary_file}")

def get_sample_indices(df_with_clusters: pd.DataFrame, original_df: pd.DataFrame) -> np.ndarray:
    """Get indices of sampled points in the original dataset."""
    # Simple approach: assume sample is a subset and find matching indices
    # This works when sampling preserves original row data
    sample_indices = []
    
    for _, sample_row in df_with_clusters.iterrows():
        # Find matching row in original data (could be improved with better matching logic)
        matches = original_df.index[
            (original_df == sample_row.drop(['cluster'], errors='ignore')).all(axis=1)
        ].tolist()
        if matches:
            sample_indices.append(matches[0])
    
    return np.array(sample_indices) 