#!/usr/bin/env python3

"""
Add new papers to an existing sample from the filtered dataset.

This script allows users to expand their sample by:
1. Loading the existing sample and stratified dataset
2. Finding available papers not yet in the sample
3. Presenting random candidates interactively with full metadata
4. Adding approved papers to sample.csv

Usage: python sampling-add.py -n 100 --add 10 [options]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import rootutils


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "sampling.log", mode='a'),  # Append mode
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("sampling-add")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Add new papers to an existing sample")
    
    parser.add_argument(
        '--sample-size', '-n', type=int, help='Current sample size (for directory path)',
        required=True
    )
    parser.add_argument(
        '--add-count', '--add', type=int, help='Number of papers to add',
        default=10
    )
    parser.add_argument(
        '--sample-dir', '-d', type=str, help='Sample directory path',
        default=None  # To be set based on sample size
    )
    parser.add_argument(
        '--seed', '-s', type=int, help='Random seed for reproducibility',
        default=42
    )
    parser.add_argument(
        '--philosophy-only', '-p', help='Only show philosophy-related papers',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Set default sample directory if not provided
    if args.sample_dir is None:
        args.sample_dir = f'data/annotations/samples/n{args.sample_size}'
    
    return args


def load_data(sample_dir: Path, logger: logging.Logger) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load sample.csv and stratified_dataset.csv from the sample directory.
    
    Args:
        sample_dir: Path to sample directory
        logger: Logger instance
        
    Returns:
        Tuple of (sample_df, stratified_df) or (None, None) if files don't exist
    """
    sample_file = sample_dir / 'sample.csv'
    stratified_file = sample_dir / 'stratified_dataset.csv'
    
    if not sample_dir.exists():
        logger.error(f"Sample directory does not exist: {sample_dir}")
        return None, None
    
    if not sample_file.exists():
        logger.error(f"Sample file does not exist: {sample_file}")
        return None, None
    
    if not stratified_file.exists():
        logger.error(f"Stratified dataset does not exist: {stratified_file}")
        logger.error("Run sampling.py first to generate the stratified dataset")
        return None, None
    
    try:
        sample_df = pd.read_csv(sample_file)
        stratified_df = pd.read_csv(stratified_file)
        
        logger.info(f"Loaded sample with {len(sample_df)} papers")
        logger.info(f"Loaded stratified dataset with {len(stratified_df)} papers")
        
        return sample_df, stratified_df
        
    except Exception as e:
        logger.error(f"Error reading files: {e}")
        return None, None


def get_docling_file_path(identifier: str) -> Optional[Path]:
    """Get the docling file path for a given identifier."""
    docling_dir = Path("data/docling")
    
    md_file = docling_dir / f"{identifier}.md"
    txt_file = docling_dir / f"{identifier}.txt"
    
    if md_file.exists():
        return md_file
    elif txt_file.exists():
        return txt_file
    else:
        return None


def load_metadata(identifier: str) -> Optional[pd.Series]:
    """Load full metadata for a paper from the original dataset."""
    metadata_file = Path("data/metadata/2025-07-09-en-combined-metadata.csv")
    
    if not metadata_file.exists():
        return None
    
    try:
        df = pd.read_csv(metadata_file)
        paper_metadata = df[df['identifier'] == identifier]
        
        if len(paper_metadata) > 0:
            return paper_metadata.iloc[0]
        else:
            return None
    except Exception:
        return None


def is_philosophy_related(category_names: str) -> bool:
    """Check if paper contains philosophy-related keywords."""
    if pd.isna(category_names):
        return False
    
    keywords = ["philosophy", "ethic", "moral", "value", "virtue"]
    category_lower = str(category_names).lower()
    
    return any(keyword in category_lower for keyword in keywords)


def display_paper_details(identifier: str, stratified_df: pd.DataFrame, candidate_num: int = 0) -> None:
    """Display detailed information about a paper."""
    # Find paper in stratified dataset
    paper_data = stratified_df[stratified_df['identifier'] == identifier]
    
    if len(paper_data) == 0:
        print(f"Paper {identifier} not found in stratified dataset")
        return
    
    paper = paper_data.iloc[0]
    
    # Load full metadata
    metadata = load_metadata(identifier)
    data_source = metadata if metadata is not None else paper
    
    prefix = f"CANDIDATE {candidate_num}" if candidate_num > 0 else "PAPER"
    print("\n" + "="*80)
    print(f"{prefix}: {identifier}")
    print("="*80)
    
    # Display key fields
    print(f"Title: {data_source.get('title', 'N/A')}")
    print(f"Authors: {data_source.get('authors', 'N/A')}")
    print(f"Year: {data_source.get('year', 'N/A')}")
    print(f"Categories: {data_source.get('category_names', 'N/A')}")
    print(f"Type: {data_source.get('type', 'N/A')}")
    print(f"Journal: {data_source.get('journal', 'N/A')}")
    print(f"Abstract: {str(data_source.get('abstract', 'N/A'))[:200]}{'...' if len(str(data_source.get('abstract', ''))) > 200 else ''}")
    
    # Show docling file info
    docling_file = get_docling_file_path(identifier)
    if docling_file:
        print(f"Docling file: {docling_file}")
        try:
            with open(docling_file, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())
            print(f"Word count: {word_count}")
        except Exception:
            print("Word count: Unable to read file")
    else:
        print("Docling file: NOT FOUND")
    
    # Show stratum info
    if 'stratum' in paper:
        print(f"Stratum: {paper['stratum']}")
    elif 'cluster|year|category' in paper:
        print(f"Stratum: {paper['cluster|year|category']}")
    
    # Show if philosophy-related
    category_names = data_source.get('category_names', '') or ''
    is_phil = is_philosophy_related(str(category_names))
    print(f"Philosophy-related: {'Yes' if is_phil else 'No'}")


def find_candidates(stratified_df: pd.DataFrame, sample_df: pd.DataFrame, 
                   count: int, philosophy_only: bool, seed: int, logger: logging.Logger) -> List[str]:
    """
    Find candidate papers to add to the sample.
    
    Args:
        stratified_df: Full stratified dataset
        sample_df: Current sample
        count: Number of candidates to find
        philosophy_only: Whether to only show philosophy-related papers
        seed: Random seed
        logger: Logger instance
        
    Returns:
        List of candidate paper identifiers
    """
    # Exclude papers already in the sample
    sampled_ids = list(sample_df['identifier'])
    available_papers = stratified_df[~stratified_df['identifier'].isin(sampled_ids)]  # type: ignore
    
    logger.info(f"Found {len(available_papers)} papers not in current sample")
    
    # Filter for philosophy papers if requested
    if philosophy_only:
        philosophy_mask = available_papers['category_names'].apply(is_philosophy_related)  # type: ignore
        available_papers = available_papers[philosophy_mask]
        logger.info(f"Filtered to {len(available_papers)} philosophy-related papers")
    
    if len(available_papers) == 0:
        logger.warning("No candidate papers available")
        return []
    
    # Randomly sample candidates
    num_candidates = min(count, len(available_papers))
    candidates = available_papers.sample(n=num_candidates, random_state=seed)  # type: ignore
    
    logger.info(f"Selected {num_candidates} random candidates")
    return candidates['identifier'].tolist()


def add_papers(sample_df: pd.DataFrame, new_paper_ids: List[str], 
               stratified_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add new papers to the sample.
    
    Args:
        sample_df: Current sample DataFrame
        new_paper_ids: List of paper IDs to add
        stratified_df: Stratified dataset for stratum info
        logger: Logger instance
        
    Returns:
        Updated sample DataFrame
    """
    if not new_paper_ids:
        return sample_df
    
    new_rows = []
    
    for paper_id in new_paper_ids:
        # Find the paper in stratified dataset
        paper_data = stratified_df[stratified_df['identifier'] == paper_id]
        
        if len(paper_data) == 0:
            logger.warning(f"Paper {paper_id} not found in stratified dataset")
            continue
        
        paper = paper_data.iloc[0]
        
        # Create new row for sample
        cluster_year_category = (
            str(paper.get('cluster', '')) + "|" +
            str(paper.get('year_quantile', '')) + "|" +
            str(paper.get('category_clean', ''))
        )
        
        new_row = {
            'identifier': paper_id,
            'cluster|year|category': cluster_year_category
        }
        
        new_rows.append(new_row)
    
    if new_rows:
        # Add new rows to sample
        new_df = pd.DataFrame(new_rows)
        updated_sample = pd.concat([sample_df, new_df], ignore_index=True)
        
        logger.info(f"Successfully added {len(new_rows)} papers to sample")
        return updated_sample
    
    return sample_df


def main():
    """Main execution function."""
    # Setup
    root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
    args = parse_args()
    
    # Create output directory and setup logging
    sample_dir = Path(args.sample_dir)
    logger = setup_logging(sample_dir)
    
    logger.info("=== ADDING PAPERS TO SAMPLE ===")
    logger.info(f"Sample directory: {sample_dir}")
    logger.info(f"Papers to add: {args.add_count}")
    logger.info(f"Philosophy only: {args.philosophy_only}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load data
    sample_df, stratified_df = load_data(sample_dir, logger)
    
    if sample_df is None or stratified_df is None:
        logger.error("Cannot proceed without sample and stratified dataset")
        sys.exit(1)
    
    # Show current sample info
    print(f"\n=== CURRENT SAMPLE ===")
    print(f"Current sample size: {len(sample_df)}")
    print(f"Target additions: {args.add_count}")
    
    # Find candidates
    candidates = find_candidates(stratified_df, sample_df, args.add_count * 3, 
                               args.philosophy_only, args.seed, logger)
    
    if not candidates:
        logger.error("No candidate papers found")
        sys.exit(1)
    
    # Present candidates for approval
    approved_papers = []
    
    print(f"\nReviewing {len(candidates)} candidates...")
    print("For each candidate, enter 'y' or 'yes' to add, anything else to skip (default: N)")
    
    for i, candidate_id in enumerate(candidates):
        if len(approved_papers) >= args.add_count:
            print(f"\nTarget reached! Approved {len(approved_papers)} papers.")
            break
            
        print(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        display_paper_details(candidate_id, stratified_df, candidate_num=i+1)
        
        # Get user input
        try:
            response = input(f"\nAdd this paper to sample? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                approved_papers.append(candidate_id)
                print(f"✓ Added ({len(approved_papers)}/{args.add_count})")
            else:
                print("✗ Skipped")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
    
    if not approved_papers:
        logger.info("No papers were approved for addition")
        return
    
    # Add approved papers to sample
    updated_sample = add_papers(sample_df, approved_papers, stratified_df, logger)
    
    # Save updated sample
    sample_file = sample_dir / 'sample.csv'
    updated_sample.to_csv(sample_file, index=False)
    logger.info(f"Updated sample saved to: {sample_file}")
    
    # Create backup if this is the first addition
    backup_file = sample_dir / 'sample_before_additions.csv'
    if not backup_file.exists():
        sample_df.to_csv(backup_file, index=False)
        logger.info(f"Original sample backed up to: {backup_file}")
    
    # Print summary
    print(f"\n=== ADDITION SUMMARY ===")
    print(f"Original sample size: {len(sample_df)}")
    print(f"Papers added: {len(approved_papers)}")
    print(f"New sample size: {len(updated_sample)}")
    
    if len(approved_papers) > 0:
        print(f"\nAdded papers:")
        for paper_id in approved_papers:
            print(f"  • {paper_id}")
    
    # Check philosophy percentage in updated sample
    if len(updated_sample) > 0:
        def check_philosophy(row):
            metadata = load_metadata(row['identifier'])
            if metadata is not None:
                category_names = metadata.get('category_names', '') or ''
                return is_philosophy_related(str(category_names))
            return False
        
        philosophy_count = updated_sample.apply(check_philosophy, axis=1).sum()  # type: ignore
        philosophy_percentage = philosophy_count / len(updated_sample) * 100
        print(f"\nPhilosophy papers in sample: {philosophy_count}/{len(updated_sample)} ({philosophy_percentage:.1f}%)")


if __name__ == '__main__':
    main() 