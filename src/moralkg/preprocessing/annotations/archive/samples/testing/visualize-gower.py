#!/usr/bin/env python3
"""
Simple visualization of Gower clustering results.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import rootutils

def main():
    # Set up paths
    root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)
    input_file = root / 'data' / 'annotations' / 'samples' / 'gower-test' / 'gower_clusters.csv'
    output_dir = root / 'data' / 'annotations' / 'samples' / 'gower-test'
    
    print(f"Loading clustering results from {input_file}")
    df = pd.read_csv(input_file)
    
    # Prepare numeric data for PCA
    numeric_cols = ['year', 'num_categories']
    X = df[numeric_cols].values
    
    # Handle categorical data (simple encoding)
    type_encoded = pd.get_dummies(df['type'], prefix='type')
    X_full = pd.concat([df[numeric_cols], type_encoded], axis=1)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: PCA scatter plot colored by cluster
    colors = plt.cm.Set1(range(len(df['cluster'].unique())))
    for i, cluster in enumerate(sorted(df['cluster'].unique())):
        mask = df['cluster'] == cluster
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Clusters in PCA Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster sizes bar chart
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Papers')
    ax2.set_title('Cluster Sizes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'cluster_visualization.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {plot_file}")
    
    # Print summary
    print(f"\nClustering Summary:")
    print(f"Total papers: {len(df)}")
    print(f"Number of clusters: {len(df['cluster'].unique())}")
    print(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of variance")
    
    plt.show()

if __name__ == '__main__':
    main() 