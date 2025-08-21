#!/usr/bin/env python3
"""
Gower clustering with pre-sampling for scalability.
"""

import gower_multiprocessing as gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.manifold import TSNE
from utils import extract_year_column
from evaluation_utils import (
    setup_logging, time_operation, evaluate_clustering_quality,
    evaluate_representativeness, create_enhanced_mds_plot, 
    save_evaluation_summary, get_sample_indices
)

def main():        
    # Set up paths
    root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)
    input_file = root / 'data' / 'metadata' / '2025-07-09-en-combined-metadata.csv'
    output_dir = root / 'data' / 'annotations' / 'samples' / 'test-gower-sampled'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, 'gower_subset_sampled')
    
    # Load Dataset
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Original shape: {df.shape}")
    
    # Select a subset of columns
    cols = ['title', 'authors', 'year', 'num_categories', 'category_names']
    df_subset = df[cols].dropna().copy()
    logger.info(f"Working with {df_subset.shape} subset: {cols}")
    
    # Parse year
    df_subset: pd.DataFrame = df_subset.copy()
    df_subset['year'] = extract_year_column(df_subset, 'year')
    df_subset = df_subset.dropna()
    logger.info(f"After cleaning: {len(df_subset)} records")
    
    # Store original data for evaluation
    original_data = df_subset.copy()
    
    # PRE-SAMPLE FOR SCALABILITY
    # Use stratified sampling based on year and categories to maintain diversity
    sample_size = 15000  # Manageable size that should complete in ~5 minutes
    logger.info(f"Pre-sampling to {sample_size} records for Gower computation...")
    
    # Stratified sampling by year decade and number of categories
    df_subset['year_decade'] = (df_subset['year'] // 10) * 10
    df_subset['cat_bins'] = pd.cut(df_subset['num_categories'], bins=5, labels=['very_low', 'low', 'med', 'high', 'very_high'])
    
    # Sample proportionally from each stratum
    sampled_df = df_subset.groupby(['year_decade', 'cat_bins'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df_subset)))), random_state=42)
    ).reset_index(drop=True)
    
    # If we don't have enough, fill up with random sampling
    if len(sampled_df) < sample_size:
        remaining = df_subset[~df_subset.index.isin(sampled_df.index)]
        additional = remaining.sample(min(len(remaining), sample_size - len(sampled_df)), random_state=42)
        sampled_df = pd.concat([sampled_df, additional]).reset_index(drop=True)
    
    # Clean up temporary columns
    sampled_df = sampled_df[cols + ['year']].copy()
    
    logger.info(f"Sampled dataset size: {len(sampled_df)} records")
    
    # Compute Gower matrix on the sampled data
    with time_operation(logger, "Gower distance matrix computation"):
        distance_matrix = gower.gower_matrix(sampled_df)
    
    with time_operation(logger, "hierarchical clustering"):
        linkage_matrix = linkage(distance_matrix, method='ward')
        clusters = fcluster(linkage_matrix, t=5, criterion='maxclust')
    
    # Add clusters to dataframe
    sampled_df['cluster'] = clusters
    
    # Create t-SNE visualization based on Gower distances
    with time_operation(logger, "t-SNE computation"):
        tsne = TSNE(
            n_components=2, 
            metric='precomputed', 
            random_state=42,
            perplexity=min(30, len(sampled_df)-1),
            init='random'
        )
        mds_coords = tsne.fit_transform(distance_matrix)
        
        logger.info("t-SNE visualization completed successfully")
    
    # Save results
    sampled_df.to_csv(output_dir / 'gower_clusters_subset_sampled.csv', index=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Found {len(np.unique(clusters))} clusters:")
    for i in range(1, len(np.unique(clusters)) + 1):
        count = np.sum(clusters == i)
        logger.info(f"  Cluster {i}: {count} items")
    
    # Sample 100 papers equally from each cluster
    n_clusters = len(np.unique(clusters))
    target_sample_size = 100
    per_cluster = target_sample_size // n_clusters
    remainder = target_sample_size % n_clusters
    
    logger.info(f"Sampling {target_sample_size} papers ({per_cluster} per cluster, +{remainder} extra)...")
    
    sampled_dfs = []
    for i, cluster_id in enumerate(sorted(np.unique(clusters))):
        cluster_data = sampled_df.loc[sampled_df['cluster'] == cluster_id].copy()
        
        # Add extra samples to first clusters if there's a remainder
        sample_size = per_cluster + (1 if i < remainder else 0)
        
        if len(cluster_data) >= sample_size:
            sampled = cluster_data.sample(n=sample_size, random_state=42)
        else:
            # If cluster is smaller than needed, take all samples
            sampled = cluster_data.copy()
            logger.warning(f"Cluster {cluster_id} only has {len(cluster_data)} items, taking all")
        
        sampled_dfs.append(sampled)
        logger.info(f"  Sampled {len(sampled)} from cluster {cluster_id}")
    
    # Combine all sampled data
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    # Save the stratified sample
    sample_output = output_dir / 'gower_subset_sampled_n100.csv'
    final_sample.to_csv(sample_output, index=False)
    
    logger.info(f"Stratified sample of {len(final_sample)} papers saved to {sample_output}")
    logger.info("Sample distribution by cluster:")
    for cluster_id in sorted(final_sample['cluster'].unique()):
        count = len(final_sample[final_sample['cluster'] == cluster_id])
        logger.info(f"  Cluster {cluster_id}: {count} papers")
    
    # EVALUATION SECTION
    logger.info("Starting comprehensive evaluation...")
    
    # For hierarchical clustering, we need to create a feature matrix for evaluation
    try:
        clustering_metrics = evaluate_clustering_quality(distance_matrix, clusters, logger)
    except Exception as e:
        logger.warning(f"Could not compute clustering metrics with distance matrix: {e}")
        clustering_metrics = {}
    
    # Evaluate representativeness against the original full dataset
    representativeness_metrics = evaluate_representativeness(original_data, final_sample, logger)
    
    # Get sample indices for enhanced MDS plot
    try:
        sample_indices = get_sample_indices(final_sample, sampled_df)
        
        # Create enhanced t-SNE plot with samples highlighted
        create_enhanced_mds_plot(
            mds_coords, clusters, sample_indices,
            output_dir / 'gower_subset_sampled_tsne_with_samples.png',
            'Gower Hierarchical (Subset Features, Pre-sampled) - t-SNE Visualization', logger
        )
    except Exception as e:
        logger.warning(f"Could not create enhanced t-SNE plot: {e}")
        # Fall back to basic plot
        plt.figure(figsize=(10, 8))
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title('Gower Hierarchical Clustering - t-SNE Visualization (Subset Features, Pre-sampled)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(output_dir / 'gower_subset_sampled_tsne_basic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save evaluation summary
    save_evaluation_summary(clustering_metrics, representativeness_metrics, output_dir, 'gower_subset_sampled', logger)
    
    logger.info("Evaluation completed!")
    logger.info(f"Note: This analysis used {len(sampled_df)} sampled records from the original {len(original_data)} for computational feasibility.")
    
    return final_sample

if __name__ == '__main__':
    main() 