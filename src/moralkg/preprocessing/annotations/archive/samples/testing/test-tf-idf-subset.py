#!/usr/bin/env python3
"""
Simple TF-IDF Vectorizer with K-means Clustering on subset of Phil Papers Dataset features.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from scipy.sparse import save_npz
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from evaluation_utils import (
    create_enhanced_mds_plot,
    evaluate_clustering_quality,
    evaluate_representativeness,
    get_sample_indices,
    save_evaluation_summary,
    setup_logging,
    time_operation,
)
from utils import extract_year_column

def main():
    # Set up paths
    root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)
    input_file = root / 'data' / 'metadata' / '2025-07-09-en-combined-metadata.csv'
    output_dir = root / 'data' / 'annotations' / 'samples' / 'test-tf-idf'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, 'tfidf_subset')
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Original shape: {df.shape}")
    
    # Select a subset of columns and sample for testing (matching gower subset)
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
    
    # Combine text fields into single blob (only subset columns)
    logger.info("Creating text blobs from subset columns...")
    def create_text_blob(row):
        parts = []
        for col in cols:
            if pd.notna(row[col]):
                parts.append(str(row[col]))
        return ' '.join(parts)
    
    df_subset['text_blob'] = df_subset.apply(create_text_blob, axis=1)
    
    # Filter out empty texts
    df_subset: pd.DataFrame = df_subset.loc[df_subset['text_blob'].str.len() > 0].copy()
    logger.info(f"After filtering: {len(df_subset)} records")
    
    # TF-IDF vectorization
    with time_operation(logger, "TF-IDF vectorization"):
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(df_subset['text_blob'])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # K-means clustering
    with time_operation(logger, "K-means clustering"):
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Add clusters to dataframe
    df_subset['cluster'] = clusters
    

    
    # Create 2D visualization using t-SNE with Barnes-Hut approximation 
    with time_operation(logger, "t-SNE computation with Barnes-Hut approximation"):
        tsne = TSNE(
            n_components=2, 
            random_state=42,
            method='barnes_hut', 
            perplexity=min(30, len(df_subset)//4),
            verbose=1 if len(df_subset) > 10000 else 0
        )
        tfidf_dense_for_tsne = tfidf_matrix.toarray()
        mds_coords = tsne.fit_transform(tfidf_dense_for_tsne)
        
        # Ensure we have valid coordinates
        if mds_coords is None or mds_coords.shape[0] == 0:
            raise ValueError("t-SNE transformation failed")
        
        logger.info(f"t-SNE completed successfully with perplexity={tsne.perplexity}")
    
    # Save clustering artifacts
    logger.info("Saving clustering results...")
    
    # Save TF-IDF matrix (sparse format)
    save_npz(output_dir / "tfidf_matrix_subset.npz", tfidf_matrix)
    
    # Save vectorizer for feature names
    with open(output_dir / "tfidf_vectorizer_subset.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save KMeans model
    with open(output_dir / "kmeans_model_subset.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    # Save full dataset with cluster assignments
    df_subset.to_csv(output_dir / "tfidf_clusters_subset.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Found {len(np.unique(clusters))} clusters:")
    for i in range(len(np.unique(clusters))):
        count = np.sum(clusters == i)
        logger.info(f"  Cluster {i}: {count} items")

    # Sample 100 papers equally from each cluster
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
    sample_output = output_dir / 'tfidf_subset_n100.csv'
    final_sample.to_csv(sample_output, index=False)
    
    logger.info(f"Stratified sample of {len(final_sample)} papers saved to {sample_output}")
    logger.info("Sample distribution by cluster:")
    for cluster_id in sorted(final_sample['cluster'].unique()):
        count = len(final_sample[final_sample['cluster'] == cluster_id])
        logger.info(f"  Cluster {cluster_id}: {count} papers")
    
    # EVALUATION SECTION
    logger.info("Starting comprehensive evaluation...")
    
    # Evaluate clustering quality
    tfidf_dense = tfidf_matrix.toarray()
    clustering_metrics = evaluate_clustering_quality(tfidf_dense, clusters, logger)
    
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
            output_dir / 'tfidf_subset_tsne_with_samples.png',
            'TF-IDF K-means (Subset Features) - t-SNE Visualization', logger
        )
    except Exception as e:
        logger.warning(f"Could not create enhanced t-SNE plot: {e}")
        # Fall back to basic plot
        plt.figure(figsize=(10, 8))
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title('TF-IDF K-means Clustering - t-SNE Visualization (Subset Features)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(output_dir / 'tfidf_subset_tsne_basic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save evaluation summary
    save_evaluation_summary(clustering_metrics, representativeness_metrics, output_dir, 'tfidf_subset', logger)
    
    logger.info("Evaluation completed!")
    
    return final_sample

if __name__ == "__main__":
    result = main()
