# sampling.py

"""
Get n=X representative samples from PhilPapers dataset.

Preserves the temporal, authorial, categorical, semantic, etc. distribution of
the dataset while ensuring diversity through TF-IDF title dissimilarity and
author constraints.

Usage: python sampling.py -n N [options]
"""

import csv
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from typing import List, Dict, Tuple, Set
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Sampling for PhilPapers dataset")
    parser.add_argument('--sample-size', '-n', type=int, required=True, help='Number of papers to sample')
    parser.add_argument('--csv-file', '-i', default='/opt/extra/avijit/projects/moralkg/data/metadata/phil-papers/2025-05-07-en.csv', help='Input CSV file')
    parser.add_argument('--output-dir', '-o', default='/opt/extra/avijit/projects/moralkg/data/sample-papers', help='Output directory')
    parser.add_argument('--min-title-words', '-t',  type=int, default=4, help='Min words in title or TF-IDF diversity (default: 4)')
    parser.add_argument('--diversity-weight', '-w', type=float, default=0.6, help='TF-IDF diversity weight vs random (default: 0.6)')
    parser.add_argument('--no-author-overlap', '-a', action='store_true', help='Prevent author overlap in sample')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    return parser.parse_args()

def clean_title(title: str) -> str:
    """Clean title for TF-IDF processing."""
    import re
    if not title:
        return ""
    # lowercase, remove special chars, normalize whitespace
    text = re.sub(r'[^\w\s]', ' ', title.lower())
    return re.sub(r'\s+', ' ', text.strip())

def categorize_document_type(doc_type: str) -> str:
    """Categorize documents by type - preserves natural type distribution."""
    if not doc_type:
        return 'other'
    
    type_lower = doc_type.lower()
    if 'article' in type_lower or 'articolo' in type_lower:
        return 'article'
    elif 'book' in type_lower or 'livro' in type_lower:
        return 'book' 
    elif 'review' in type_lower:
        return 'review'
    else:
        return 'other'

def extract_authors(authors_str: str) -> Set[str]:
    """Extract author set from semicolon-separated string."""
    if not authors_str:
        return set()
    return {a.strip() for a in authors_str.split(';') if a.strip()}

def calculate_tfidf_similarity(papers: List[Dict], min_words: int) -> Tuple[np.ndarray, List[int]]:
    """
    Calculate TF-IDF similarity matrix for titles with sufficient length.
    
    Returns:
        similarity_matrix: Cosine similarity matrix
        valid_indices: Indices of papers with valid titles
    """
    
    # Filter papers with sufficient title length
    valid_papers = []
    valid_indices = []
    
    for i, paper in enumerate(papers):
        title = clean_title(paper.get('title', ''))
        if len(title.split()) >= min_words:
            valid_papers.append(title)
            valid_indices.append(i)
    
    print(f"Using {len(valid_papers)} papers for similarity calculation")
    
    # TF-IDF vectorization - conservative parameters to preserve diversity
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(valid_papers)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix, valid_indices

def diversified_sampling_within_stratum(
    papers: List[Dict],
    indices: List[int], 
    similarity_matrix: np.ndarray,
    valid_sim_indices: List[int],
    n_samples: int,
    diversity_weight: float,
    used_authors: Set[str],
    no_author_overlap: bool
) -> List[int]:
    """
    Sample papers from a stratum using TF-IDF diversity + author constraints.
    
    Strategy:
    1. Filter for author overlap constraints
    2. First selection: random
    3. Subsequent: maximize title diversity while respecting constraints
    """
    if n_samples <= 0 or not papers:
        return []
    
    # Filter available papers based on author overlap
    available = []
    available_indices = []
    
    for i, paper in enumerate(papers):
        global_idx = indices[i]
        
        if no_author_overlap:
            paper_authors = extract_authors(paper.get('authors', ''))
            if paper_authors.intersection(used_authors):
                continue  # Skip if author overlap
        
        available.append(paper)
        available_indices.append(global_idx)
    
    if not available:
        return []
    
    n_samples = min(n_samples, len(available))
    selected = []
    selected_sim_indices = []
    
    # First selection: random
    first_idx = random.randint(0, len(available) - 1)
    selected.append(available_indices[first_idx])
    
    # Track similarity index if available
    if available_indices[first_idx] in valid_sim_indices:
        selected_sim_indices.append(valid_sim_indices.index(available_indices[first_idx]))
    
    # Update used authors
    if no_author_overlap:
        used_authors.update(extract_authors(available[first_idx].get('authors', '')))
    
    # Subsequent selections: maximize diversity
    for _ in range(n_samples - 1):
        best_score = -1
        best_idx = -1
        
        for i, global_idx in enumerate(available_indices):
            if global_idx in selected:
                continue
            
            # Check author constraint
            if no_author_overlap:
                paper_authors = extract_authors(available[i].get('authors', ''))
                if paper_authors.intersection(used_authors):
                    continue
            
            # Calculate diversity score
            diversity_score = 1.0  # Default if no similarity data
            
            if global_idx in valid_sim_indices and selected_sim_indices:
                sim_idx = valid_sim_indices.index(global_idx)
                # Minimum similarity to any selected paper
                similarities = [similarity_matrix[sim_idx, sel_idx] for sel_idx in selected_sim_indices]
                diversity_score = 1.0 - max(similarities)  # Diversity = inverse of max similarity
            
            # Combine diversity with randomness
            random_score = random.random()
            final_score = diversity_weight * diversity_score + (1 - diversity_weight) * random_score
            
            if final_score > best_score:
                best_score = final_score
                best_idx = i
        
        if best_idx == -1:
            break  # No more valid papers
        
        selected.append(available_indices[best_idx])
        
        # Track similarity index
        if available_indices[best_idx] in valid_sim_indices:
            selected_sim_indices.append(valid_sim_indices.index(available_indices[best_idx]))
        
        # Update used authors
        if no_author_overlap:
            used_authors.update(extract_authors(available[best_idx].get('authors', '')))
    
    return selected

def sample(papers: List[Dict], args) -> Tuple[List[Dict], Dict]:
    """
    Perform representative sampling that preserves natural dataset distributions.
    
    Strategy:
    1. Stratify by document type only (preserves natural temporal distribution)
    2. Sample proportionally within each type
    3. Use TF-IDF diversity within strata
    4. Apply author constraints if requested
    """
    print("Creating document type strata...")
    
    # Group papers by document type - preserves natural temporal patterns within types
    strata = defaultdict(list)
    strata_indices = defaultdict(list)
    
    for i, paper in enumerate(papers):
        doc_type = categorize_document_type(paper.get('type', ''))
        strata[doc_type].append(paper)
        strata_indices[doc_type].append(i)
    
    print(f"Created {len(strata)} document type strata:")
    for stratum, papers_list in strata.items():
        pct = len(papers_list) / len(papers) * 100
        print(f"  {stratum}: {len(papers_list):,} papers ({pct:.1f}%)")
    
    # Calculate TF-IDF similarity matrix once for all papers
    similarity_matrix, valid_sim_indices = calculate_tfidf_similarity(papers, args.min_title_words)
    
    # Proportional allocation to preserve natural distribution
    stratum_targets = {}
    for stratum, papers_list in strata.items():
        proportion = len(papers_list) / len(papers)
        target = max(1, round(proportion * args.sample_size))
        stratum_targets[stratum] = target
    
    print(f"\nTarget samples per stratum (proportional to natural distribution):")
    for stratum, target in stratum_targets.items():
        print(f"  {stratum}: {target} papers")
    
    # Sample from each stratum
    sampled_papers = []
    used_authors = set()
    sampling_stats = {}
    
    print("\nSampling papers...")
    for stratum, papers_list in strata.items():
        target = stratum_targets[stratum]
        
        print(f"  {stratum}: sampling {target} from {len(papers_list)} papers")
        
        selected_indices = diversified_sampling_within_stratum(
            papers_list,
            strata_indices[stratum],
            similarity_matrix,
            valid_sim_indices,
            target,
            args.diversity_weight,
            used_authors,
            args.no_author_overlap
        )
        
        selected_papers = [papers[idx] for idx in selected_indices]
        sampled_papers.extend(selected_papers)
        
        sampling_stats[stratum] = {
            'total_papers': len(papers_list),
            'sampled_papers': len(selected_papers),
            'sampling_rate': len(selected_papers) / len(papers_list)
        }
    
    # Analysis of the sample
    sample_years = []
    for paper in sampled_papers:
        try:
            year = int(paper.get('year', ''))
            if year <= datetime.now().year:
                sample_years.append(year)
        except (ValueError, TypeError):
            pass
    
    analysis = {
        'total_input_papers': len(papers),
        'total_sampled_papers': len(sampled_papers),
        'target_sample_size': args.sample_size,
        'sampling_stats': sampling_stats,
        'sample_temporal_stats': {
            'valid_years': len(sample_years),
            'year_range': f"{min(sample_years)}-{max(sample_years)}" if sample_years else "N/A",
            'mean_year': np.mean(sample_years) if sample_years else None
        },
        'unique_authors_in_sample': len({
            author for paper in sampled_papers 
            for author in extract_authors(paper.get('authors', ''))
        }),
        'parameters': {
            'min_title_words': args.min_title_words,
            'diversity_weight': args.diversity_weight,
            'no_author_overlap': args.no_author_overlap,
            'seed': args.seed
        }
    }
    
    return sampled_papers, analysis

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=== REPRESENTATIVE SAMPLING OF PHILPAPERS DATASET ===")
    print(f"Sample size: {args.sample_size}")
    print(f"Author overlap: {'prevented' if args.no_author_overlap else 'allowed'}")
    print(f"TF-IDF diversity weight: {args.diversity_weight}")
    print()
    
    # Load data
    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        sys.exit(1)
    
    print(f"Loading papers from {csv_file}...")
    papers = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            papers = list(reader)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(papers):,} papers")
    
    # Perform sampling
    sampled_papers, analysis = sample(papers, args)
    
    print(f"\nSampled {len(sampled_papers)} papers")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample papers
    sample_file = output_dir / 'sample.csv'
    print(f"Saving sample to {sample_file}")
    
    with open(sample_file, 'w', newline='', encoding='utf-8') as f:
        if sampled_papers:
            writer = csv.DictWriter(f, fieldnames=sampled_papers[0].keys())
            writer.writeheader()
            writer.writerows(sampled_papers)
    
    # Save paper codes (identifiers)
    codes_file = output_dir / 'codes'
    print(f"Saving paper codes to {codes_file}")
    
    with open(codes_file, 'w', encoding='utf-8') as f:
        for paper in sampled_papers:
            identifier = paper.get('identifier', '').strip()
            if identifier:
                f.write(f"{identifier}\n")
    
    # Save analysis
    analysis_file = output_dir / 'analysis.json'
    print(f"Saving analysis to {analysis_file}")
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== SAMPLING SUMMARY ===")
    print(f"Total papers: {analysis['total_input_papers']:,}")
    print(f"Sampled: {analysis['total_sampled_papers']} ({analysis['total_sampled_papers']/analysis['total_input_papers']*100:.2f}%)")
    print(f"Unique authors: {analysis['unique_authors_in_sample']}")
    
    if analysis['sample_temporal_stats']['valid_years'] > 0:
        print(f"Year range: {analysis['sample_temporal_stats']['year_range']}")
        print(f"Mean year: {analysis['sample_temporal_stats']['mean_year']:.1f}")
    
    print("\nSampling by document type:")
    for doc_type, stats in analysis['sampling_stats'].items():
        rate = stats['sampling_rate'] * 100
        print(f"  {doc_type}: {stats['sampled_papers']}/{stats['total_papers']} ({rate:.1f}%)")
    
    print(f"\nFiles saved to {output_dir}/")

if __name__ == '__main__':
    main()
