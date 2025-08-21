#!/usr/bin/env python3

"""
Flexible replacement script for papers in the sample.

This script provides multiple fallback strategies for replacing papers:
1. Same stratum (cluster|year|category) - most conservative
2. Same cluster, different year/category - moderate
3. Random replacement - most flexible

Usage: python sampling-replace-flexible.py -n 100 --papers PAPER1,PAPER2,... [options]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import rootutils


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "sampling.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("sampling-replace-flexible")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flexibly replace papers in the sample")
    
    parser.add_argument(
        '--sample-size', '-n', type=int, help='Sample size (for directory path)',
        required=True
    )
    parser.add_argument(
        '--papers', '-p', type=str, help='Comma-separated list of paper IDs to replace',
        required=True
    )
    parser.add_argument(
        '--sample-dir', '-d', type=str, help='Sample directory path',
        default=None
    )
    parser.add_argument(
        '--seed', '-s', type=int, help='Random seed for reproducibility',
        default=42
    )
    parser.add_argument(
        '--strategy', type=str, choices=['conservative', 'moderate', 'flexible'],
        default='flexible', help='Replacement strategy (default: flexible)'
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='Interactive mode - ask for confirmation for each replacement'
    )
    
    args = parser.parse_args()
    
    if args.sample_dir is None:
        args.sample_dir = f'data/annotations/samples/n{args.sample_size}'
    
    # Parse paper list
    args.paper_list = [p.strip() for p in args.papers.split(',') if p.strip()]
    
    return args


def load_data(sample_dir: Path, logger: logging.Logger) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load sample.csv and stratified_dataset.csv from the sample directory."""
    sample_file = '/opt/extra/avijit/projects/moralkg/' / sample_dir / 'sample.csv'
    stratified_file = '/opt/extra/avijit/projects/moralkg/' / sample_dir / 'stratified_dataset.csv'
    
    if not sample_dir.exists():
        logger.error(f"Sample directory does not exist: {sample_dir}")
        return None, None
    
    if not sample_file.exists():
        logger.error(f"Sample file does not exist: {sample_file}")
        return None, None
    
    if not stratified_file.exists():
        logger.error(f"Stratified dataset does not exist: {stratified_file}")
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


def display_paper_details(identifier: str, stratified_df: pd.DataFrame, strategy_used: str = None) -> None:
    """Display detailed information about a paper."""
    paper_data = stratified_df[stratified_df['identifier'] == identifier]
    
    if len(paper_data) == 0:
        print(f"Paper {identifier} not found in stratified dataset")
        return
    
    paper = paper_data.iloc[0]
    metadata = load_metadata(identifier)
    data_source = metadata if metadata is not None else paper
    
    prefix = f"REPLACEMENT ({strategy_used})" if strategy_used else "PAPER"
    print("\n" + "="*80)
    print(f"{prefix}: {identifier}")
    print("="*80)
    
    print(f"Title: {data_source.get('title', 'N/A')}")
    print(f"Authors: {data_source.get('authors', 'N/A')}")
    print(f"Year: {data_source.get('year', 'N/A')}")
    print(f"Categories: {data_source.get('category_names', 'N/A')}")
    print(f"Type: {data_source.get('type', 'N/A')}")
    print(f"Journal: {data_source.get('journal', 'N/A')}")
    print(f"Abstract: {str(data_source.get('abstract', 'N/A'))[:200]}{'...' if len(str(data_source.get('abstract', ''))) > 200 else ''}")
    
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
    
    if 'stratum' in paper:
        print(f"Stratum: {paper['stratum']}")
    elif 'cluster|year|category' in paper:
        print(f"Stratum: {paper['cluster|year|category']}")


def find_replacement_strategies(paper_id: str, stratified_df: pd.DataFrame, sample_df: pd.DataFrame, 
                              strategy: str, seed: int, logger: logging.Logger) -> List[tuple]:
    """
    Find replacement candidates using multiple strategies.
    
    Returns:
        List of tuples (candidate_id, strategy_used)
    """
    paper_data = stratified_df[stratified_df['identifier'] == paper_id]
    
    if len(paper_data) == 0:
        logger.error(f"Paper {paper_id} not found in stratified dataset")
        return []
    
    paper = paper_data.iloc[0]
    sampled_ids = set(sample_df['identifier'])
    available_papers = stratified_df[~stratified_df['identifier'].isin(sampled_ids)]
    available_papers = available_papers[available_papers['identifier'] != paper_id]
    
    candidates = []
    
    # Strategy 1: Same stratum (conservative)
    if strategy in ['conservative', 'moderate', 'flexible']:
        stratum = paper.get('stratum', '')
        if stratum:
            same_stratum = available_papers[available_papers['stratum'] == stratum]
            if len(same_stratum) > 0:
                sample_size = min(3, len(same_stratum))
                selected = same_stratum.sample(n=sample_size, random_state=seed)
                for _, candidate in selected.iterrows():
                    candidates.append((candidate['identifier'], 'same_stratum'))
    
    # Strategy 2: Same cluster, different year/category (moderate)
    if strategy in ['moderate', 'flexible'] and len(candidates) < 5:
        cluster = paper.get('cluster', '')
        if cluster is not None:
            same_cluster = available_papers[available_papers['cluster'] == cluster]
            # Exclude same stratum papers already found
            if 'stratum' in paper:
                same_cluster = same_cluster[same_cluster['stratum'] != paper.get('stratum')]
            
            if len(same_cluster) > 0:
                sample_size = min(5, len(same_cluster))
                selected = same_cluster.sample(n=sample_size, random_state=seed)
                for _, candidate in selected.iterrows():
                    candidates.append((candidate['identifier'], 'same_cluster'))
    
    # Strategy 3: Random replacement (flexible)
    if strategy == 'flexible' and len(candidates) < 8:
        # Exclude already selected candidates
        already_selected = {cand[0] for cand in candidates}
        random_pool = available_papers[~available_papers['identifier'].isin(already_selected)]
        
        if len(random_pool) > 0:
            sample_size = min(5, len(random_pool))
            selected = random_pool.sample(n=sample_size, random_state=seed)
            for _, candidate in selected.iterrows():
                candidates.append((candidate['identifier'], 'random'))
    
    logger.info(f"Found {len(candidates)} replacement candidates for {paper_id}")
    return candidates


def replace_paper_in_sample(sample_df: pd.DataFrame, old_paper_id: str, new_paper_id: str, 
                           stratified_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Replace a paper in the sample with a new paper."""
    new_paper_data = stratified_df[stratified_df['identifier'] == new_paper_id]
    
    if len(new_paper_data) == 0:
        logger.error(f"New paper {new_paper_id} not found in stratified dataset")
        return sample_df
    
    new_paper = new_paper_data.iloc[0]
    
    # Create stratum info for new paper
    cluster_year_category = (
        str(new_paper.get('cluster', '')) + "|" +
        str(new_paper.get('year_quantile', '')) + "|" +
        str(new_paper.get('category_clean', ''))
    )
    
    # Replace in sample
    updated_sample = sample_df.copy()
    mask = updated_sample['identifier'] == old_paper_id
    
    if not mask.any():
        logger.error(f"Paper {old_paper_id} not found in current sample")
        return sample_df
    
    updated_sample.loc[mask, 'identifier'] = new_paper_id
    updated_sample.loc[mask, 'cluster|year|category'] = cluster_year_category
    
    logger.info(f"Successfully replaced {old_paper_id} with {new_paper_id}")
    return updated_sample


def process_replacements(paper_list: List[str], sample_df: pd.DataFrame, stratified_df: pd.DataFrame,
                        args, logger: logging.Logger) -> pd.DataFrame:
    """Process all requested replacements."""
    updated_sample = sample_df.copy()
    replacement_log = []
    
    for i, paper_id in enumerate(paper_list, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING REPLACEMENT {i}/{len(paper_list)}: {paper_id}")
        print(f"{'='*80}")
        
        # Check if paper is in sample
        if paper_id not in updated_sample['identifier'].values:
            logger.warning(f"Paper {paper_id} not found in sample, skipping")
            continue
        
        # Show current paper
        print(f"\n--- CURRENT PAPER ---")
        display_paper_details(paper_id, stratified_df)
        
        # Find replacement candidates
        candidates = find_replacement_strategies(paper_id, stratified_df, updated_sample, 
                                               args.strategy, args.seed + i, logger)
        
        if not candidates:
            logger.warning(f"No replacement candidates found for {paper_id}")
            continue
        
        # In interactive mode, show options and get user choice
        if args.interactive:
            print(f"\n--- REPLACEMENT OPTIONS ---")
            print("0: Keep current paper (no replacement)")
            
            for j, (candidate_id, strategy_used) in enumerate(candidates, 1):
                print(f"{j}: Replace with {candidate_id} ({strategy_used})")
                display_paper_details(candidate_id, stratified_df, strategy_used)
            
            try:
                choice = input(f"\nEnter your choice (0-{len(candidates)}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    print("Keeping current paper.")
                    continue
                
                if 1 <= choice_num <= len(candidates):
                    new_paper_id, strategy_used = candidates[choice_num - 1]
                else:
                    print("Invalid choice, skipping.")
                    continue
                    
            except (ValueError, KeyboardInterrupt):
                print("\nSkipping this replacement.")
                continue
        else:
            # Non-interactive: use first candidate
            new_paper_id, strategy_used = candidates[0]
            print(f"\n--- AUTOMATIC REPLACEMENT ---")
            display_paper_details(new_paper_id, stratified_df, strategy_used)
        
        # Perform replacement
        updated_sample = replace_paper_in_sample(updated_sample, paper_id, new_paper_id, 
                                               stratified_df, logger)
        
        replacement_log.append({
            'original': paper_id,
            'replacement': new_paper_id,
            'strategy': strategy_used
        })
        
        print(f"\n✓ Replaced {paper_id} with {new_paper_id} (strategy: {strategy_used})")
    
    # Print summary
    if replacement_log:
        print(f"\n{'='*80}")
        print("REPLACEMENT SUMMARY")
        print(f"{'='*80}")
        for entry in replacement_log:
            print(f"{entry['original']} → {entry['replacement']} ({entry['strategy']})")
    
    return updated_sample


def main():
    """Main execution function."""
    root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
    args = parse_args()
    
    sample_dir = Path(args.sample_dir)
    logger = setup_logging(sample_dir)
    
    logger.info("=== FLEXIBLE PAPER REPLACEMENT ===")
    logger.info(f"Sample directory: {sample_dir}")
    logger.info(f"Papers to replace: {len(args.paper_list)}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Interactive mode: {args.interactive}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load data
    sample_df, stratified_df = load_data(sample_dir, logger)
    
    if sample_df is None or stratified_df is None:
        logger.error("Cannot proceed without sample and stratified dataset")
        sys.exit(1)
    
    # Validate papers exist in sample
    missing_papers = [p for p in args.paper_list if p not in sample_df['identifier'].values]
    if missing_papers:
        logger.warning(f"Papers not found in sample: {missing_papers}")
        valid_papers = [p for p in args.paper_list if p in sample_df['identifier'].values]
        if not valid_papers:
            logger.error("No valid papers to replace")
            sys.exit(1)
        args.paper_list = valid_papers
    
    print(f"\nProcessing {len(args.paper_list)} replacement requests...")
    
    # Process replacements
    updated_sample = process_replacements(args.paper_list, sample_df, stratified_df, args, logger)
    
    # Save updated sample
    sample_file = sample_dir / 'sample.csv'
    updated_sample.to_csv(sample_file, index=False)
    logger.info(f"Updated sample saved to: {sample_file}")
    
    # Create backup if this is the first flexible replacement
    backup_file = sample_dir / 'sample_before_flexible_replacements.csv'
    if not backup_file.exists():
        sample_df.to_csv(backup_file, index=False)
        logger.info(f"Original sample backed up to: {backup_file}")
    
    print(f"\n✓ Replacement process completed!")


if __name__ == '__main__':
    main() 