# analyze_dataset.py

"""
Statistical analysis of PhilPapers dataset.

Usage: python analyze_dataset.py [csv_file] [--output-dir directory]
"""

import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import glob
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze PhilPapers dataset.")
    parser.add_argument(
        'csv_file', 
        nargs='?', 
        default='/opt/extra/avijit/projects/moralkg/data/metadata/2025-07-09-en-combined-metadata.csv',
        help='Path to CSV file'
    )
    parser.add_argument(
        '--output-dir', 
        default='data/scripts/phil-papers/sampling/analysis-results',
        help='Output directory for analysis results'
    )
    return parser.parse_args()

class PhilPapersAnalyzer:
    """Analyzes PhilPapers dataset statistics and metadata fields."""
    
    def __init__(self, papers_data: List[Dict]):
        self.papers = papers_data
        self.total_papers = len(papers_data)
        
    def analyze_temporal_distribution(self) -> Dict:
        """Analyze year field statistics and temporal patterns."""
        
        years = []
        invalid_years = []
        
        for paper in self.papers:
            year_str = paper.get('year', '').strip()
            if not year_str:
                invalid_years.append('[empty]')
                continue
                
            try:
                year = int(year_str)
                if year <= datetime.now().year:
                    years.append(year)
                else:
                    invalid_years.append(year_str)
            except ValueError:
                invalid_years.append(year_str)
        
        if not years:
            return {'error': 'No valid years found'}
            
        # Basic statistics
        stats = {
            'valid_count': len(years),
            'invalid_count': len(invalid_years),
            'min_year': min(years),
            'max_year': max(years),
            'mean': np.mean(years),
            'median': np.median(years),
            'std': np.std(years),
            'q25': np.percentile(years, 25),
            'q75': np.percentile(years, 75)
        }
        
        # Decade distribution
        decade_counts = Counter((year // 10) * 10 for year in years)
        stats['decade_distribution'] = dict(sorted(decade_counts.items()))
        
        # Temporal distribution (quartiles)
        sorted_years = sorted(years)
        n_buckets = 4
        quartile_buckets = {}
        
        for i in range(n_buckets):
            start_idx = i * len(sorted_years) // n_buckets
            end_idx = (i + 1) * len(sorted_years) // n_buckets if i < n_buckets - 1 else len(sorted_years)
            
            bucket_years = sorted_years[start_idx:end_idx]
            quartile_buckets[f"quartile_{i+1}"] = {
                'year_range': f"{min(bucket_years)}-{max(bucket_years)}",
                'paper_count': len(bucket_years),
                'percentage': len(bucket_years) / len(years) * 100
            }
        
        stats['equal_distribution_quartiles'] = quartile_buckets
        
        # Invalid year examples (first 10)
        stats['invalid_examples'] = list(set(invalid_years))[:10]
        
        return stats

    def analyze_document_types(self) -> Dict:
        """Analyze document type field distribution."""
        
        type_counts = Counter()
        empty_count = 0
        
        for paper in self.papers:
            doc_type = paper.get('type', '').strip()
            if doc_type:
                type_counts[doc_type] += 1
            else:
                empty_count += 1
        
        # Categorize types
        categories = {
            'articles': 0,
            'books': 0,
            'reviews': 0,
            'other': 0
        }
        
        for doc_type, count in type_counts.items():
            type_lower = doc_type.lower()
            if 'article' in type_lower or 'articolo' in type_lower:
                categories['articles'] += count
            elif 'book' in type_lower or 'livro' in type_lower:
                categories['books'] += count
            elif 'review' in type_lower:
                categories['reviews'] += count
            else:
                categories['other'] += count
        
        return {
            'total_types': len(type_counts),
            'empty_count': empty_count,
            'type_counts': dict(type_counts.most_common()),
            'categories': categories,
            'most_common': type_counts.most_common(10)
        }

    def analyze_authors(self) -> Dict:
        """Analyze author field statistics and collaboration patterns."""
        
        author_counts = Counter()
        collaboration_stats = []
        papers_with_no_authors = 0
        
        for paper in self.papers:
            authors_str = paper.get('authors', '').strip()
            if not authors_str:
                papers_with_no_authors += 1
                collaboration_stats.append(0)
                continue
                
            # Split by semicolon and clean
            authors = [a.strip() for a in authors_str.split(';') if a.strip()]
            collaboration_stats.append(len(authors))
            
            for author in authors:
                author_counts[author] += 1
        
        # Calculate statistics
        valid_collab = [x for x in collaboration_stats if x > 0]
        
        stats = {
            'unique_authors': len(author_counts),
            'total_author_instances': sum(author_counts.values()),
            'papers_with_no_authors': papers_with_no_authors,
            'mean_authors_per_paper': np.mean(valid_collab) if valid_collab else 0,
            'median_authors_per_paper': np.median(valid_collab) if valid_collab else 0,
            'max_authors_per_paper': max(valid_collab) if valid_collab else 0,
            'single_author_papers': sum(1 for x in collaboration_stats if x == 1),
            'multi_author_papers': sum(1 for x in collaboration_stats if x > 1)
        }
        
        # Author productivity
        productivity = Counter(author_counts.values())
        stats['productivity_distribution'] = dict(sorted(productivity.items()))
        
        # Most prolific authors
        stats['most_prolific'] = author_counts.most_common(15)
        
        # Multi-paper authors (potential overlap)
        multi_paper_authors = sum(1 for count in author_counts.values() if count > 1)
        stats['authors_with_multiple_papers'] = multi_paper_authors
        stats['author_overlap_risk'] = multi_paper_authors / len(author_counts) if author_counts else 0
        
        return stats

    def analyze_titles(self) -> Dict:
        """Analyze title field statistics for TF-IDF parameters."""
        
        title_lengths = []
        word_counts = []
        empty_titles = 0
        
        for paper in self.papers:
            title = paper.get('title', '').strip()
            if not title:
                empty_titles += 1
                continue
                
            # Character length
            title_lengths.append(len(title))
            
            # Word count (simple splitting)
            words = re.findall(r'\w+', title.lower())
            word_counts.append(len(words))
        
        if not word_counts:
            return {'error': 'No valid titles found'}
        
        # Basic statistics
        stats = {
            'total_papers': self.total_papers,
            'empty_titles': empty_titles,
            'papers_with_titles': len(word_counts),
            'mean_title_length': np.mean(title_lengths),
            'median_title_length': np.median(title_lengths),
            'std_title_length': np.std(title_lengths),
            'mean_words_per_title': np.mean(word_counts),
            'median_words_per_title': np.median(word_counts),
            'std_words_per_title': np.std(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'q25_words': np.percentile(word_counts, 25),
            'q75_words': np.percentile(word_counts, 75)
        }
        
        # Word count distribution (for TF-IDF thresholds)
        word_count_dist = Counter(word_counts)
        stats['word_count_distribution'] = dict(sorted(word_count_dist.items())[:20])
        
        # Exclusion analysis for different minimum word thresholds
        exclusion_rates = {}
        for min_words in [1, 2, 3, 4, 5, 8, 10]:
            excluded = sum(1 for wc in word_counts if wc < min_words)
            exclusion_rates[min_words] = {
                'excluded_papers': excluded,
                'remaining_papers': len(word_counts) - excluded,
                'exclusion_rate': excluded / len(word_counts)
            }
        
        stats['exclusion_analysis'] = exclusion_rates
        
        return stats

    def analyze_missing_data(self) -> Dict:
        """Analyze missing data patterns across all fields."""
        
        if not self.papers:
            return {'error': 'No data to analyze'}
        
        fields = self.papers[0].keys()
        missing_stats = {}
        
        for field in fields:
            missing_count = sum(1 for paper in self.papers if not paper.get(field, '').strip())
            missing_stats[field] = {
                'missing_count': missing_count,
                'present_count': self.total_papers - missing_count,
                'missing_rate': missing_count / self.total_papers
            }
        
        return missing_stats

def format_output(analyzer: PhilPapersAnalyzer, analysis_date: str, input_file: str) -> str:
    """Format all analysis results into a comprehensive text report."""
    
    # Run all analyses
    print("Running analyses...")
    temporal_stats = analyzer.analyze_temporal_distribution()
    doc_type_stats = analyzer.analyze_document_types()
    author_stats = analyzer.analyze_authors()
    title_stats = analyzer.analyze_titles()
    missing_data_stats = analyzer.analyze_missing_data()
    
    # Format output
    output = []
    output.append("=" * 80)
    output.append("PHILPAPERS DATASET ANALYSIS")
    output.append("=" * 80)
    output.append(f"Analysis Date: {analysis_date}")
    output.append(f"Input File: {input_file}")
    output.append(f"Total Papers: {analyzer.total_papers:,}")
    output.append("")
    
    # === TEMPORAL DISTRIBUTION ===
    output.append("TEMPORAL DISTRIBUTION")
    output.append("-" * 40)
    if 'error' not in temporal_stats:
        output.append(f"Valid Years: {temporal_stats['valid_count']:,} ({temporal_stats['valid_count']/analyzer.total_papers:.1%})")
        output.append(f"Invalid/Missing Years: {temporal_stats['invalid_count']:,} ({temporal_stats['invalid_count']/analyzer.total_papers:.1%})")
        output.append(f"Year Range: {temporal_stats['min_year']} - {temporal_stats['max_year']}")
        output.append(f"Mean Year: {temporal_stats['mean']:.1f}")
        output.append(f"Median Year: {temporal_stats['median']:.1f}")
        output.append(f"Standard Deviation: {temporal_stats['std']:.1f}")
        output.append(f"Quartiles: Q1={temporal_stats['q25']:.0f}, Q3={temporal_stats['q75']:.0f}")
        output.append("")
        
        
        output.append("Equal Distribution Quartiles:")
        for quartile, data in temporal_stats['equal_distribution_quartiles'].items():
            output.append(f"  {quartile} ({data['year_range']}): {data['paper_count']:,} papers ({data['percentage']:.1f}%)")
        output.append("")
        
        output.append("Decade Distribution:")
        for decade, count in temporal_stats['decade_distribution'].items():
            pct = count / temporal_stats['valid_count'] * 100
            output.append(f"  {decade}s: {count:,} papers ({pct:.1f}%)")
        output.append("")
        
        if temporal_stats['invalid_examples']:
            output.append("Invalid Year Examples:")
            output.append(f"  {', '.join(temporal_stats['invalid_examples'])}")
            output.append("")
    else:
        output.append(f"ERROR: {temporal_stats['error']}")
        output.append("")
    
    # === DOCUMENT TYPES ===
    output.append("DOCUMENT TYPE ANALYSIS")
    output.append("-" * 40)
    output.append(f"Unique Document Types: {doc_type_stats['total_types']}")
    output.append(f"Papers with Missing Type: {doc_type_stats['empty_count']:,}")
    output.append("")
    
    output.append("Type Categories:")
    for category, count in doc_type_stats['categories'].items():
        pct = count / analyzer.total_papers * 100
        output.append(f"  {category.capitalize()}: {count:,} papers ({pct:.1f}%)")
    output.append("")
    
    output.append("Most Common Document Types:")
    for doc_type, count in doc_type_stats['most_common'][:10]:
        pct = count / analyzer.total_papers * 100
        # Truncate long type names
        display_type = doc_type[:50] + "..." if len(doc_type) > 50 else doc_type
        output.append(f"  {display_type}: {count:,} ({pct:.1f}%)")
    output.append("")
    
    # === AUTHOR ANALYSIS ===
    output.append("AUTHOR AND COLLABORATION ANALYSIS")
    output.append("-" * 40)
    output.append(f"Unique Authors: {author_stats['unique_authors']:,}")
    output.append(f"Total Author Instances: {author_stats['total_author_instances']:,}")
    output.append(f"Papers with No Authors: {author_stats['papers_with_no_authors']:,}")
    output.append(f"Mean Authors per Paper: {author_stats['mean_authors_per_paper']:.2f}")
    output.append(f"Median Authors per Paper: {author_stats['median_authors_per_paper']:.1f}")
    output.append(f"Maximum Authors per Paper: {author_stats['max_authors_per_paper']}")
    output.append("")
    
    output.append("Collaboration Patterns:")
    single_pct = author_stats['single_author_papers'] / analyzer.total_papers * 100
    multi_pct = author_stats['multi_author_papers'] / analyzer.total_papers * 100
    output.append(f"  Single-author papers: {author_stats['single_author_papers']:,} ({single_pct:.1f}%)")
    output.append(f"  Multi-author papers: {author_stats['multi_author_papers']:,} ({multi_pct:.1f}%)")
    output.append("")
    
    output.append("Author Productivity (papers per author):")
    for papers_count, author_count in sorted(author_stats['productivity_distribution'].items())[:15]:
        if papers_count == 1:
            output.append(f"  {papers_count} paper: {author_count:,} authors")
        else:
            output.append(f"  {papers_count} papers: {author_count:,} authors")
    output.append("")
    
    overlap_pct = author_stats['author_overlap_risk'] * 100
    output.append(f"Author Overlap Risk: {overlap_pct:.1f}% ({author_stats['authors_with_multiple_papers']:,} authors with multiple papers)")
    output.append("")
    
    output.append("Most Prolific Authors:")
    for author, count in author_stats['most_prolific']:
        output.append(f"  {author}: {count} papers")
    output.append("")
    
    # === TITLE ANALYSIS ===
    output.append("TITLE ANALYSIS (for TF-IDF parameters)")
    output.append("-" * 40)
    if 'error' not in title_stats:
        output.append(f"Papers with Titles: {title_stats['papers_with_titles']:,}")
        output.append(f"Empty Titles: {title_stats['empty_titles']:,}")
        output.append("")
        
        output.append("Title Length Statistics (characters):")
        output.append(f"  Mean: {title_stats['mean_title_length']:.1f}")
        output.append(f"  Median: {title_stats['median_title_length']:.1f}")
        output.append(f"  Standard Deviation: {title_stats['std_title_length']:.1f}")
        output.append("")
        
        output.append("Words per Title Statistics:")
        output.append(f"  Mean: {title_stats['mean_words_per_title']:.2f}")
        output.append(f"  Median: {title_stats['median_words_per_title']:.1f}")
        output.append(f"  Standard Deviation: {title_stats['std_words_per_title']:.2f}")
        output.append(f"  Range: {title_stats['min_words']} - {title_stats['max_words']} words")
        output.append(f"  Quartiles: Q1={title_stats['q25_words']:.0f}, Q3={title_stats['q75_words']:.0f}")
        output.append("")
        
        output.append("Word Count Distribution (most common):")
        for word_count, paper_count in title_stats['word_count_distribution'].items():
            pct = paper_count / title_stats['papers_with_titles'] * 100
            output.append(f"  {word_count} words: {paper_count:,} papers ({pct:.1f}%)")
        output.append("")
        
        output.append("Exclusion Analysis for Minimum Word Thresholds:")
        for threshold, stats in title_stats['exclusion_analysis'].items():
            excluded_pct = stats['exclusion_rate'] * 100
            remaining = stats['remaining_papers']
            output.append(f"  Min {threshold} words: excludes {stats['excluded_papers']:,} papers ({excluded_pct:.1f}%), keeps {remaining:,}")
        output.append("")
    else:
        output.append(f"ERROR: {title_stats['error']}")
        output.append("")
    
    # === MISSING DATA ANALYSIS ===
    output.append("MISSING DATA ANALYSIS")
    output.append("-" * 40)
    output.append("Missing Data by Field:")
    for field, stats in missing_data_stats.items():
        missing_pct = stats['missing_rate'] * 100
        output.append(f"  {field}: {stats['missing_count']:,} missing ({missing_pct:.2f}%)")
    output.append("")
    
    return "\n".join(output)

def main():
    """Main execution function."""
    args = parse_args()
    
    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        print(f"Error: File {csv_file} not found!")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PhilPapers Dataset Analysis")
    print(f"Input: {csv_file}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print("Loading dataset...")
    papers_data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers_data.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(papers_data):,} papers")
    print()
    
    # Analyze
    analyzer = PhilPapersAnalyzer(papers_data)
    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate report
    report = format_output(analyzer, analysis_date, str(csv_file))
    
    # Save report
    output_file = output_dir / 'analysis.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Analysis complete!")
    print(f"Report saved to: {output_file}")

if __name__ == '__main__':
    main() 
