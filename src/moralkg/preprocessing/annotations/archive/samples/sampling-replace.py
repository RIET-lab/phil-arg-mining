#!/usr/bin/env python3

"""
Replace a paper in the sample with a similar paper from the same stratum.

This script allows users to audit and refine their sample by:
1. Taking a paper identifier as input
2. Finding 10 similar papers from the same cluster & stratification
3. Presenting choices interactively with full metadata
4. Updating sample.csv with the replacement

Usage: python sampling-replace.py -n 100 -p PAPER_ID [options]
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
    
    return logging.getLogger("sampling-replace")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Replace a paper in the sample with a similar paper")
    
    parser.add_argument(
        '--sample-size', '-n', type=int, help='Sample size (for directory path)',
        required=True
    )
    parser.add_argument(
        '--paper-id', '-p', type=str, help='Identifier of paper to replace',
        required=True
    )
    parser.add_argument(
        '--sample-dir', '-d', type=str, help='Sample directory path',
        default=None  # To be set based on sample size
    )
    parser.add_argument(
        '--seed', '-s', type=int, help='Random seed for reproducibility',
        default=42
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


def display_paper_details(identifier: str, stratified_df: pd.DataFrame, is_current: bool = False) -> None:
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
    
    prefix = "CURRENT PAPER" if is_current else "REPLACEMENT OPTION"
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


def find_replacement_candidates(paper_id: str, stratified_df: pd.DataFrame, sample_df: pd.DataFrame, 
                              seed: int, logger: logging.Logger) -> List[str]:
    """
    Find replacement candidates from the same stratum.
    
    Args:
        paper_id: Identifier of paper to replace
        stratified_df: Full stratified dataset
        sample_df: Current sample
        seed: Random seed
        logger: Logger instance
        
    Returns:
        List of candidate paper identifiers
    """
    # Find the paper in stratified dataset
    paper_data = stratified_df[stratified_df['identifier'] == paper_id]
    
    if len(paper_data) == 0:
        logger.error(f"Paper {paper_id} not found in stratified dataset")
        return []
    
    paper = paper_data.iloc[0]
    
    # Get the stratum
    stratum = paper.get('stratum', '')
    if not stratum:
        logger.error(f"No stratum information found for paper {paper_id}")
        return []
    
    logger.info(f"Finding candidates from stratum: {stratum}")
    
    # Find all papers in the same stratum
    stratum_papers = stratified_df[stratified_df['stratum'] == stratum]
    
    # Exclude papers already in the sample
    sampled_ids = list(sample_df['identifier'])
    available_papers = stratum_papers[~stratum_papers['identifier'].isin(sampled_ids)]  # type: ignore
    
    # Also exclude the current paper itself
    available_papers = available_papers[available_papers['identifier'] != paper_id]
    
    logger.info(f"Found {len(available_papers)} available papers in stratum")
    
    if len(available_papers) == 0:
        logger.warning("No replacement candidates available in the same stratum")
        return []
    
    # Randomly sample up to 10 candidates
    num_candidates = min(10, len(available_papers))
    candidates = available_papers.sample(n=num_candidates, random_state=seed)  # type: ignore
    
    return candidates['identifier'].tolist()


def replace_paper(sample_df: pd.DataFrame, old_paper_id: str, new_paper_id: str, 
                  stratified_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Replace a paper in the sample with a new paper.
    
    Args:
        sample_df: Current sample DataFrame
        old_paper_id: Paper to replace
        new_paper_id: Replacement paper
        stratified_df: Stratified dataset for stratum info
        logger: Logger instance
        
    Returns:
        Updated sample DataFrame
    """
    # Find the new paper in stratified dataset to get stratum info
    new_paper_data = stratified_df[stratified_df['identifier'] == new_paper_id]
    
    if len(new_paper_data) == 0:
        logger.error(f"New paper {new_paper_id} not found in stratified dataset")
        return sample_df
    
    new_paper = new_paper_data.iloc[0]
    
    # Create new row for sample
    new_stratum = new_paper.get('stratum', '')
    cluster_year_category = (
        str(new_paper.get('cluster', '')) + "|" +
        str(new_paper.get('year_quantile', '')) + "|" +
        str(new_paper.get('category_clean', ''))
    )
    
    new_row = {
        'identifier': new_paper_id,
        'cluster|year|category': cluster_year_category
    }
    
    # Replace the old paper with the new one
    updated_sample = sample_df.copy()
    mask = updated_sample['identifier'] == old_paper_id
    
    if not mask.any():
        logger.error(f"Paper {old_paper_id} not found in current sample")
        return sample_df
    
    # Update the row
    for col, val in new_row.items():
        updated_sample.loc[mask, col] = val
    
    logger.info(f"Successfully replaced {old_paper_id} with {new_paper_id}")
    return updated_sample


def main():
    """Main execution function."""
    # Setup
    root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
    args = parse_args()
    
    # Create output directory and setup logging
    sample_dir = Path(args.sample_dir)
    logger = setup_logging(sample_dir)
    
    logger.info("=== PAPER REPLACEMENT ===")
    logger.info(f"Sample directory: {sample_dir}")
    logger.info(f"Paper to replace: {args.paper_id}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load data
    sample_df, stratified_df = load_data(sample_dir, logger)
    
    if sample_df is None or stratified_df is None:
        logger.error("Cannot proceed without sample and stratified dataset")
        sys.exit(1)
    
    # Check if paper is in current sample
    if args.paper_id not in sample_df['identifier'].values:
        logger.error(f"Paper {args.paper_id} is not in the current sample")
        logger.info("Available papers in sample:")
        for paper_id in sample_df['identifier']:
            logger.info(f"  {paper_id}")
        sys.exit(1)
    
    # Display current paper
    print("\n" + "="*80)
    print("PAPER TO REPLACE")
    print("="*80)
    display_paper_details(args.paper_id, stratified_df, is_current=True)
    
    # Find replacement candidates
    candidates = find_replacement_candidates(args.paper_id, stratified_df, sample_df, args.seed, logger)
    
    if not candidates:
        logger.error("No replacement candidates found")
        sys.exit(1)
    
    # Present options to user
    print(f"\nFound {len(candidates)} replacement candidates from the same stratum:")
    print("=" * 80)
    
    for i, candidate_id in enumerate(candidates, 1):
        print(f"\n--- Option {i}: {candidate_id} ---")
        display_paper_details(candidate_id, stratified_df)
    
    # Get user choice
    print("\n" + "="*80)
    print("REPLACEMENT SELECTION")
    print("="*80)
    print(f"Choose a replacement for {args.paper_id}:")
    print("0: Keep current paper (no replacement)")
    
    for i, candidate_id in enumerate(candidates, 1):
        print(f"{i}: Replace with {candidate_id}")
    
    try:
        choice = input(f"\nEnter your choice (0-{len(candidates)}): ").strip()
        choice_num = int(choice)
        
        if choice_num == 0:
            print("No replacement made.")
            logger.info("User chose to keep current paper")
            return
        
        if 1 <= choice_num <= len(candidates):
            new_paper_id = candidates[choice_num - 1]
            print(f"Replacing {args.paper_id} with {new_paper_id}")
            
            # Perform replacement
            updated_sample = replace_paper(sample_df, args.paper_id, new_paper_id, stratified_df, logger)
            
            # Save updated sample
            sample_file = sample_dir / 'sample.csv'
            updated_sample.to_csv(sample_file, index=False)
            logger.info(f"Updated sample saved to: {sample_file}")
            
            # Create backup if this is the first replacement
            backup_file = sample_dir / 'sample_before_replacements.csv'
            if not backup_file.exists():
                sample_df.to_csv(backup_file, index=False)
                logger.info(f"Original sample backed up to: {backup_file}")
            
            print(f"\n✓ Successfully replaced {args.paper_id} with {new_paper_id}")
        else:
            print("Invalid choice.")
            sys.exit(1)
            
    except (ValueError, KeyboardInterrupt):
        print("\nOperation cancelled.")
        sys.exit(1)


if __name__ == '__main__':
    main()
