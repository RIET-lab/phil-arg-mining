#!/usr/bin/env python3
"""
Simple Gower clustering test on a subset of the phil-papers metadata features.
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
    output_dir = root / 'data' / 'annotations' / 'samples' / 'test-gower'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, 'gower_subset')
    
    # Load Dataset
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Original shape: {df.shape}")
    
    # Select a subset of columns and sample for testing
    cols = ['title', 'authors', 'year', 'num_categories', 'category_names']
    df_subset = df[cols].dropna().copy()
    logger.info(f"Working with {df_subset.shape} subset: {cols}")
    
    # Parse year
    df_subset = df_subset.copy()
    df_subset['year'] = extract_year_column(df_subset, 'year')
    df_subset = df_subset.dropna()
    logger.info(f"After cleaning: {len(df_subset)} records")
    
    # Randomly sample a smaller subset for performance
    sample_size = min(10000, len(df_subset) // 4)  # Cap at 10k for performance
    df_subset = df_subset.sample(n=sample_size, random_state=42).copy()
    logger.info(f"Randomly sampled subset: {len(df_subset)} records")
    
    # Store original data for evaluation
    original_data = df_subset.copy()
    
    # Compute Gower matrix
    with time_operation(logger, "Gower distance matrix computation"):
        distance_matrix = gower.gower_matrix(df_subset)
    
    with time_operation(logger, "hierarchical clustering"):
        linkage_matrix = linkage(distance_matrix, method='ward')
        clusters = fcluster(linkage_matrix, t=5, criterion='maxclust')
    
    # Add clusters to dataframe
    df_subset['cluster'] = clusters
    
    # Create t-SNE visualization based on Gower distances
    with time_operation(logger, "t-SNE computation"):
        tsne = TSNE(
            n_components=2, 
            metric='precomputed', 
            random_state=42,
            perplexity=min(30, len(df_subset)-1),
            init='random'
        )
        mds_coords = tsne.fit_transform(distance_matrix)
        
        logger.info("t-SNE visualization completed successfully")
    
    # Save results
    df_subset.to_csv(output_dir / 'gower_clusters_subset.csv', index=False)
    
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
        cluster_data = df_subset.loc[df_subset['cluster'] == cluster_id].copy()
        
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
    sample_output = output_dir / 'gower_subset_n100.csv'
    final_sample.to_csv(sample_output, index=False)
    
    logger.info(f"Stratified sample of {len(final_sample)} papers saved to {sample_output}")
    logger.info("Sample distribution by cluster:")
    for cluster_id in sorted(final_sample['cluster'].unique()):
        count = len(final_sample[final_sample['cluster'] == cluster_id])
        logger.info(f"  Cluster {cluster_id}: {count} papers")
    
    # EVALUATION SECTION
    logger.info("Starting comprehensive evaluation...")
    
    # For hierarchical clustering, we need to create a feature matrix for evaluation
    # Use the Gower distance matrix as features (not ideal but workable)
    try:
        clustering_metrics = evaluate_clustering_quality(distance_matrix, clusters, logger)
    except Exception as e:
        logger.warning(f"Could not compute clustering metrics with distance matrix: {e}")
        clustering_metrics = {}
    
    # Evaluate representativeness
    original_data_df: pd.DataFrame = original_data
    final_sample_df: pd.DataFrame = final_sample
    representativeness_metrics = evaluate_representativeness(original_data_df, final_sample_df, logger)
    
    # Get sample indices for enhanced MDS plot
    try:
        sample_indices = get_sample_indices(final_sample_df, df_subset)
        
        # Create enhanced t-SNE plot with samples highlighted
        create_enhanced_mds_plot(
            mds_coords, clusters, sample_indices,
            output_dir / 'gower_subset_tsne_with_samples.png',
            'Gower Hierarchical (Subset Features) - t-SNE Visualization', logger
        )
    except Exception as e:
        logger.warning(f"Could not create enhanced t-SNE plot: {e}")
        # Fall back to basic plot
        plt.figure(figsize=(10, 8))
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title('Gower Hierarchical Clustering - t-SNE Visualization (Subset Features)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(output_dir / 'gower_subset_tsne_basic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save evaluation summary
    save_evaluation_summary(clustering_metrics, representativeness_metrics, output_dir, 'gower_subset', logger)
    
    logger.info("Evaluation completed!")
    
    return final_sample


if __name__ == '__main__':
    main()
