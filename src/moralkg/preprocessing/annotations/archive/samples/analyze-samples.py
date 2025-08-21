#!/usr/bin/env python3

"""
Analyze samples from data/annotations/samples and compare them to the original dataset.
Creates comprehensive visualizations for each nX sample folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from utils import extract_year_column
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SampleAnalyzer:
    def __init__(self, original_dataset_path="data/metadata/2025-07-09-en-combined-metadata.csv"):
        self.original_dataset_path = original_dataset_path
        self.original_df = self._load_and_prepare_original()
    
    def _load_and_prepare_original(self):
        """Load and prepare original dataset."""
        print(f"Loading original dataset...")
        df = pd.read_csv(self.original_dataset_path)
        df['year'] = extract_year_column(df, 'year')
        df = df.dropna(subset=['year'])
        df['num_authors'] = df['authors'].str.split(',').str.len()
        print(f"Dataset loaded: {len(df)} records")
        return df
    
    def _load_sample_data(self, sample_folder):
        """Load sample data and merge with original dataset."""
        for filename in ['sample.csv', 'sample_original.csv']:
            file_path = os.path.join(sample_folder, filename)
            if os.path.exists(file_path):
                sample_df = pd.read_csv(file_path)
                sample_identifiers = sample_df['identifier'].tolist()
                sample_data = self.original_df[
                    self.original_df['identifier'].isin(sample_identifiers)
                ].copy()
                sample_data['year'] = pd.to_numeric(sample_data['year'], errors='coerce')
                sample_data = sample_data.dropna(subset=['year']) # type: ignore
                sample_data['num_authors'] = sample_data['authors'].str.split(',').str.len()
                return sample_data
        raise FileNotFoundError(f"No sample file found in {sample_folder}")
    
    def _create_comparison_plot(self, original_data, sample_data, column, title, output_path, 
                               log_scale=False, bins_orig=50, bins_sample=30):
        """Create side-by-side comparison histogram plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original dataset histogram
        ax1.hist(original_data, bins=bins_orig, alpha=0.8, color='blue')
        ax1.set_xlabel(column.title())
        ax1.set_ylabel('Number of Papers')
        if log_scale:
            ax1.set_yscale('log')
        ax1.set_title(f'Original Dataset - {title}')
        ax1.grid(True, alpha=0.3)
        
        # Sample dataset histogram
        ax2.hist(sample_data, bins=bins_sample, alpha=0.8, color='red')
        ax2.set_xlabel(column.title())
        ax2.set_ylabel('Number of Papers')
        if log_scale:
            ax2.set_yscale('log')
        ax2.set_title(f'Sample - {title}')
        ax2.grid(True, alpha=0.3)
        
        # Match x-axis ranges
        x_min = min(original_data.min(), sample_data.min())
        x_max = max(original_data.max(), sample_data.max())
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_bar_comparison(self, original_counts, sample_counts, labels, title, output_path, 
                              log_scale=False, top_n=15):
        """Create side-by-side bar comparison plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Get top N from original
        if hasattr(original_counts, 'most_common'):
            top_items = dict(original_counts.most_common(top_n))
        else:
            top_items = dict(original_counts.head(top_n))
        
        items = list(top_items.keys())
        original_values = [top_items[item] for item in items]
        sample_values = [sample_counts.get(item, 0) for item in items]
        
        x = np.arange(len(items))
        
        # Original dataset bar chart
        ax1.bar(x, original_values, alpha=0.8, color='blue')
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
        if log_scale:
            ax1.set_yscale('log')
        ax1.set_title(f'Original Dataset - {title}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(items, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Sample dataset bar chart
        ax2.bar(x, sample_values, alpha=0.8, color='red')
        ax2.set_xlabel(labels[0])
        ax2.set_ylabel(labels[1])
        if log_scale:
            ax2.set_yscale('log')
        ax2.set_title(f'Sample - {title}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(items, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_category_counts(self, df):
        """Extract and count categories from dataframe."""
        categories = []
        for cat_str in df['category_names'].dropna():
            categories.extend(cat_str.split(';'))
        return Counter(categories)
    
    def _get_author_counts(self, df):
        """Extract and count authors from dataframe."""
        authors = []
        for author_str in df['authors'].dropna():
            authors.extend([a.strip() for a in author_str.split(',')])
        return Counter(authors)
    
    def create_all_plots(self, sample_df, output_dir, sample_name):
        """Create all comparison plots."""
        # Year distribution
        self._create_comparison_plot(
            self.original_df['year'], sample_df['year'], 
            'year', 'Year', 
            os.path.join(output_dir, 'year_distribution.png'), 
            log_scale=True
        )
        
        # Authors per paper
        self._create_comparison_plot(
            self.original_df['num_authors'], sample_df['num_authors'],
            'num_authors', 'Authors per Paper',
            os.path.join(output_dir, 'authors_per_paper.png'),
            log_scale=True, bins_orig=20, bins_sample=10
        )
        
        # Categories per paper
        self._create_comparison_plot(
            self.original_df['num_categories'], sample_df['num_categories'],
            'num_categories', 'Categories per Paper',
            os.path.join(output_dir, 'categories_per_paper.png'),
            bins_orig=20, bins_sample=10
        )
        
        # Top categories
        orig_cat_counts = self._get_category_counts(self.original_df)
        sample_cat_counts = self._get_category_counts(sample_df)
        self._create_bar_comparison(
            orig_cat_counts, sample_cat_counts,
            ['Category', 'Papers'], 'Top Categories',
            os.path.join(output_dir, 'top_categories.png'),
            log_scale=True, top_n=20
        )
        
        # Top authors
        orig_author_counts = self._get_author_counts(self.original_df)
        sample_author_counts = self._get_author_counts(sample_df)
        self._create_bar_comparison(
            orig_author_counts, sample_author_counts,
            ['Author', 'Papers'], 'Top Authors',
            os.path.join(output_dir, 'top_authors.png'),
            log_scale=True, top_n=15
        )
        
        # Top years
        orig_year_counts = self.original_df['year'].value_counts()
        sample_year_counts = sample_df['year'].value_counts()
        self._create_bar_comparison(
            orig_year_counts, sample_year_counts,
            ['Year', 'Papers'], 'Top Publication Years',
            os.path.join(output_dir, 'top_years.png'),
            top_n=10
        )
        
        # TF-IDF PCA of titles
        self._create_tfidf_pca_plot(sample_df, output_dir, sample_name)
        
        # Summary statistics
        self._create_summary_stats(sample_df, output_dir, sample_name)
    
    def _create_tfidf_pca_plot(self, sample_df, output_dir, sample_name):
        """Create TF-IDF PCA plot."""
        try:
            original_titles = self.original_df['title'].dropna().tolist()
            sample_titles = sample_df['title'].dropna().tolist()
            all_titles = original_titles + sample_titles
            
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                       min_df=2, max_df=0.8)
            tfidf_matrix = vectorizer.fit_transform(all_titles)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(tfidf_matrix.toarray()) # type: ignore
            
            _, ax = plt.subplots(figsize=(12, 8))
            
            original_pca = pca_result[:len(original_titles)]
            sample_pca = pca_result[len(original_titles):]
            
            ax.scatter(original_pca[:, 0], original_pca[:, 1], 
                      alpha=0.6, s=20, label='Original', c='blue')
            ax.scatter(sample_pca[:, 0], sample_pca[:, 1], 
                      alpha=0.8, s=50, label='Sample', c='red', edgecolor='black')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_title('TF-IDF PCA of Titles')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tfidf_pca_titles.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"TF-IDF PCA failed: {e}")
    
    def _create_summary_stats(self, sample_df, output_dir, sample_name):
        """Create summary statistics."""
        orig_authors = self._get_author_counts(self.original_df)
        sample_authors = self._get_author_counts(sample_df)
        orig_categories = self._get_category_counts(self.original_df)
        sample_categories = self._get_category_counts(sample_df)
        
        summary = {
            'Metric': ['Total Papers', 'Unique Authors', 'Unique Categories', 
                      'Avg Authors/Paper', 'Avg Categories/Paper', 'Year Range'],
            'Original': [
                len(self.original_df),
                len(orig_authors),
                len(orig_categories),
                f"{self.original_df['num_authors'].mean():.1f}",
                f"{self.original_df['num_categories'].mean():.1f}",
                f"{self.original_df['year'].min()}-{self.original_df['year'].max()}"
            ],
            'Sample': [
                len(sample_df),
                len(sample_authors),
                len(sample_categories),
                f"{sample_df['num_authors'].mean():.1f}",
                f"{sample_df['num_categories'].mean():.1f}",
                f"{sample_df['year'].min()}-{sample_df['year'].max()}"
            ]
        }
        
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    def analyze_sample(self, sample_folder, sample_name):
        """Analyze a single sample."""
        print(f"Analyzing {sample_name}...")
        
        output_dir = f"data/figures/annotations/samples/{sample_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        sample_df = self._load_sample_data(sample_folder)
        self.create_all_plots(sample_df, output_dir, sample_name)
        
        print(f"{sample_name} complete")

def main():
    """Main function."""
    analyzer = SampleAnalyzer()
    
    samples_dir = "data/annotations/samples"
    if not os.path.exists(samples_dir):
        print(f"Error: {samples_dir} not found")
        return
    
    sample_folders = [f for f in os.listdir(samples_dir) 
                     if f.startswith('n') and os.path.isdir(os.path.join(samples_dir, f))]
    
    if not sample_folders:
        print("No sample folders found")
        return
    
    print(f"Found samples: {sorted(sample_folders)}")
    
    for folder in sorted(sample_folders):
        try:
            analyzer.analyze_sample(os.path.join(samples_dir, folder), folder)
        except Exception as e:
            print(f"{folder} failed: {e}")
    
    print("Analysis complete")

if __name__ == "__main__":
    main()

