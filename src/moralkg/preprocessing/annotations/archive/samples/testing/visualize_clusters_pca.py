#!/usr/bin/env python3
"""
PCA Visualization of TF-IDF Clusters for Phil Papers Dataset
"""

import rootutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.sparse import load_npz
import pickle

ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)

def load_clustering_data():
    """Load the saved clustering artifacts"""
    data_dir = ROOT / "data/annotations/samples/pure-if-idf-text-blob"
    
    # Load TF-IDF matrix
    tfidf_matrix = load_npz(data_dir / "tfidf_matrix.npz")
    
    # Load vectorizer
    with open(data_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    # Load KMeans model
    with open(data_dir / "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    # Load dataset with clusters
    df = pd.read_csv(data_dir / "full_dataset_with_clusters.csv")
    
    return tfidf_matrix, vectorizer, kmeans, df

def visualize_clusters_pca(tfidf_matrix, df, n_components=2, sample_size=1000):
    """Visualize clusters using PCA"""
    
    # If dataset is large, sample for visualization
    if len(df) > sample_size:
        print(f"Sampling {sample_size} documents for visualization...")
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        df_sample = df.iloc[sample_indices].reset_index(drop=True)
        tfidf_sample = tfidf_matrix[sample_indices]
    else:
        df_sample = df
        tfidf_sample = tfidf_matrix
    
    # Perform PCA
    print("Performing PCA...")
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(tfidf_sample.toarray())
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=df_sample['cluster'], 
                         cmap='tab20', 
                         alpha=0.6, 
                         s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
    plt.title('TF-IDF Clusters Visualization (PCA)')
    plt.colorbar(scatter, label='Cluster')
    
    # Add cluster centers if available
    if hasattr(pca, 'transform'):
        try:
            # Transform cluster centers to PCA space
            with open(ROOT / "data/annotations/samples/pure-if-idf-text-blob/kmeans_model.pkl", "rb") as f:
                kmeans = pickle.load(f)
            
            centers_pca = pca.transform(kmeans.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, 
                       label='Cluster Centers')
            plt.legend()
        except:
            pass
    
    plt.tight_layout()
    
    # Save plot
    output_path = ROOT / "data/annotations/samples/pure-if-idf-text-blob/cluster_pca_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA visualization to: {output_path}")
    
    plt.show()
    
    return pca_result, pca

def analyze_cluster_characteristics(df, vectorizer, top_n=10):
    """Analyze the characteristics of each cluster"""
    print("\nCluster Analysis:")
    print("=" * 50)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Load TF-IDF matrix and KMeans model
    tfidf_matrix = load_npz(ROOT / "data/annotations/samples/pure-if-idf-text-blob/tfidf_matrix.npz")
    with open(ROOT / "data/annotations/samples/pure-if-idf-text-blob/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    # Analyze each cluster
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_docs = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
        
        # Get top terms for this cluster from cluster center
        cluster_center = kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-top_n:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        top_scores = [cluster_center[i] for i in top_indices]
        
        print("Top terms:", ", ".join(f"{term}({score:.3f})" for term, score in zip(top_terms, top_scores)))
        
        # Show some example titles
        if 'title' in cluster_docs.columns:
            example_titles = cluster_docs['title'].dropna().head(3).tolist()
            print("Example titles:")
            for title in example_titles:
                print(f"  - {title}")

def main():
    # Load data
    print("Loading clustering data...")
    tfidf_matrix, vectorizer, kmeans, df = load_clustering_data()
    
    print(f"Loaded {len(df)} documents with {len(df['cluster'].unique())} clusters")
    
    # Visualize with PCA
    pca_result, pca = visualize_clusters_pca(tfidf_matrix, df)
    
    # Analyze cluster characteristics
    analyze_cluster_characteristics(df, vectorizer)
    
    # Save PCA results
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['cluster'] = df['cluster'] if len(df) <= 1000 else df.iloc[np.random.choice(len(df), 1000, replace=False)]['cluster'].values
    pca_df.to_csv(ROOT / "data/annotations/samples/pure-if-idf-text-blob/pca_results.csv", index=False)
    
    print(f"\nSaved PCA results to: data/annotations/samples/pure-if-idf-text-blob/pca_results.csv")

if __name__ == "__main__":
    main() 