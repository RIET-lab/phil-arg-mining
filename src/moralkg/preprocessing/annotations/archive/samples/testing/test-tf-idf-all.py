#!/usr/bin/env python3
"""
Simple TF-IDF Vectorizer with K-means Clustering for Phil Papers Dataset
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import rootutils
from scipy.sparse import save_npz
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
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
    output_dir = root / 'data' / 'annotations' / 'samples' / 'test-tf-idf'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir, 'tfidf_all')
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")
    
    # Parse year
    df['year'] = extract_year_column(df, 'year')
    df = df.dropna()
    logger.info(f"After cleaning: {len(df)} records")
    
    # Store original data for evaluation
    original_data = df.copy()
    
    # Combine text fields into single blob
    logger.info("Creating text blobs...")
    def create_text_blob(row):
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                parts.append(str(row[col]))
        return ' '.join(parts)
    
    df['text_blob'] = df.apply(create_text_blob, axis=1)
    
    # Filter out empty texts  
    df = df.loc[df['text_blob'].str.len() > 0].copy()
    logger.info(f"After filtering: {len(df)} records")
    
    # TF-IDF vectorization
    with time_operation(logger, "TF-IDF vectorization"):
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(df['text_blob'])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # K-means clustering
    with time_operation(logger, "K-means clustering"):
        n_clusters = 5 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Add clusters to dataframe
    df['cluster'] = clusters
    

    
    # Create 2D visualization using t-SNE 
    with time_operation(logger, "t-SNE computation with Barnes-Hut approximation"):
        tsne = TSNE(
            n_components=2, 
            random_state=42,
            method='barnes_hut',
            perplexity=min(30, len(df)//4),
            verbose=1 if len(df) > 10000 else 0
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
    save_npz(output_dir / "tfidf_matrix.npz", tfidf_matrix)
    
    # Save vectorizer for feature names
    with open(output_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save KMeans model
    with open(output_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    # Save full dataset with cluster assignments
    df.to_csv(output_dir / "tfidf_clusters.csv", index=False)
    
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
        cluster_data = df.loc[df['cluster'] == cluster_id].copy()
        
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
    sample_output = output_dir / 'tfidf_all_n100.csv'
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
    representativeness_metrics = evaluate_representativeness(original_data, final_sample, logger)
    
    # Get sample indices for enhanced MDS plot
    try:
        sample_indices = get_sample_indices(final_sample, df)
        
        # Create enhanced t-SNE plot with samples highlighted
        create_enhanced_mds_plot(
            mds_coords, clusters, sample_indices,
            output_dir / 'tfidf_all_tsne_with_samples.png',
            'TF-IDF K-means (All Features) - t-SNE Visualization', logger
        )
    except Exception as e:
        logger.warning(f"Could not create enhanced t-SNE plot: {e}")
        # Fall back to basic plot
        plt.figure(figsize=(10, 8))
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title('TF-IDF K-means Clustering - t-SNE Visualization (All Features)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(output_dir / 'tfidf_all_tsne_basic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save evaluation summary
    save_evaluation_summary(clustering_metrics, representativeness_metrics, output_dir, 'tfidf_all', logger)
    
    logger.info("Evaluation completed!")
    
    return final_sample

if __name__ == "__main__":
    result = main()
