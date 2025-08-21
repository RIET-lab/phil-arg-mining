#!/usr/bin/env python3

"""
Supplement existing samples with suggestions from suggestions.csv.

This script implements a sampling strategy that:
1. Checks for existing sample directory (e.g., n100)
2. Reads existing sample.csv and suggestions.csv
3. Calculates shortfall between desired size and current size
4. Interactively presents suggestions with metadata for user approval
5. Updates sample.csv with the approved suggestions

Usage: python sampling-suggestions.py -n 100 [options]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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
    
    return logging.getLogger("sampling-suggestions")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Supplement existing samples with suggestions")
    
    parser.add_argument(
        '--sample-size', '-n', type=int, help='Target number of papers in sample',
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


def check_existing_sample(sample_dir: Path, logger: logging.Logger) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Check for existing sample.csv and suggestions.csv in the directory.
    
    Args:
        sample_dir: Path to sample directory
        logger: Logger instance
        
    Returns:
        Tuple of (existing_sample_df, suggestions_df) or (None, None) if files don't exist
    """
    sample_file = sample_dir / 'sample.csv'
    suggestions_file = sample_dir / 'suggestions.csv'
    
    if not sample_dir.exists():
        logger.error(f"Sample directory does not exist: {sample_dir}")
        return None, None
    
    if not sample_file.exists():
        logger.error(f"Sample file does not exist: {sample_file}")
        return None, None
    
    if not suggestions_file.exists():
        logger.error(f"Suggestions file does not exist: {suggestions_file}")
        return None, None
    
    try:
        sample_df = pd.read_csv(sample_file)
        suggestions_df = pd.read_csv(suggestions_file)
        
        logger.info(f"Found existing sample with {len(sample_df)} papers")
        logger.info(f"Found suggestions file with {len(suggestions_df)} suggestions")
        
        return sample_df, suggestions_df
        
    except Exception as e:
        logger.error(f"Error reading existing files: {e}")
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


def display_suggestion(suggestion: pd.Series, metadata: Optional[pd.Series] = None) -> None:
    """Display a suggestion with all available metadata."""
    print("\n" + "="*80)
    print(f"SUGGESTION: {suggestion['identifier']}")
    print("="*80)
    
    # Use full metadata if available, otherwise use suggestion data
    data_source = metadata if metadata is not None else suggestion
    
    # Display key fields
    print(f"Title: {data_source.get('title', 'N/A')}")
    print(f"Authors: {data_source.get('authors', 'N/A')}")
    print(f"Year: {data_source.get('year', 'N/A')}")
    print(f"Categories: {data_source.get('category_names', 'N/A')}")
    print(f"Type: {data_source.get('type', 'N/A')}")
    print(f"Journal: {data_source.get('journal', 'N/A')}")
    print(f"Abstract: {str(data_source.get('abstract', 'N/A'))[:200]}{'...' if len(str(data_source.get('abstract', ''))) > 200 else ''}")
    
    # Show docling file location
    docling_file = get_docling_file_path(str(suggestion['identifier']))
    if docling_file:
        print(f"Docling file: {docling_file}")
        
        # Show word count
        try:
            with open(docling_file, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())
            print(f"Word count: {word_count}")
        except Exception:
            print("Word count: Unable to read file")
    else:
        print("Docling file: NOT FOUND")
    
    print(f"Stratum: {suggestion.get('cluster|year|category', 'N/A')}")
    print(f"Suggestion reason: {suggestion.get('suggestion_reason', 'N/A')}")


def supplement_sample(sample_df: pd.DataFrame, suggestions_df: pd.DataFrame, 
                     target_size: int, seed: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Supplement existing sample with suggestions to reach target size.
    Interactively ask user to approve each suggestion.
    
    Args:
        sample_df: Existing sample DataFrame
        suggestions_df: Suggestions DataFrame
        target_size: Target sample size
        seed: Random seed
        logger: Logger instance
        
    Returns:
        Updated sample DataFrame
    """
    current_size = len(sample_df)
    shortfall = target_size - current_size
    
    logger.info(f"Current sample size: {current_size}")
    logger.info(f"Target sample size: {target_size}")
    logger.info(f"Shortfall: {shortfall}")
    
    if shortfall <= 0:
        logger.info("No supplementation needed - sample already meets or exceeds target size")
        return sample_df
    
    if len(suggestions_df) == 0:
        logger.warning("No suggestions available for supplementation")
        return sample_df
    
    # Randomize suggestions order
    logger.info(f"Randomizing suggestions order (seed: {seed})")
    randomized_suggestions = suggestions_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    approved_suggestions = []
    
    print(f"\nNeed to fill {shortfall} positions. Reviewing suggestions...")
    print("For each suggestion, enter 'y' or 'yes' to accept, anything else to reject (default: N)")
    
    for i, (_, suggestion) in enumerate(randomized_suggestions.iterrows()):
        if len(approved_suggestions) >= shortfall:
            print(f"\nTarget reached! Accepted {len(approved_suggestions)} suggestions.")
            break
            
        print(f"\n--- Suggestion {i+1}/{len(randomized_suggestions)} ---")
        
        # Load full metadata
        metadata = load_metadata(str(suggestion['identifier']))
        
        # Display suggestion
        display_suggestion(suggestion, metadata)
        
        # Get user input
        try:
            response = input(f"\nAccept this suggestion? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                approved_suggestions.append(suggestion)
                print(f"✓ Accepted ({len(approved_suggestions)}/{shortfall})")
            else:
                print("✗ Rejected")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
    
    if len(approved_suggestions) == 0:
        logger.info("No suggestions were approved")
        return sample_df
    
    # Convert approved suggestions to DataFrame and remove suggestion_reason column
    approved_df = pd.DataFrame(approved_suggestions)
    if 'suggestion_reason' in approved_df.columns:
        approved_df = approved_df.drop('suggestion_reason', axis=1)
    
    # Combine existing sample with approved suggestions
    updated_sample = pd.concat([sample_df, approved_df], ignore_index=True)
    
    logger.info(f"Updated sample size: {len(updated_sample)}")
    logger.info(f"Added {len(approved_suggestions)} approved suggestions to sample")
    
    return updated_sample


def main():
    """Main execution function."""
    # Setup
    root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
    args = parse_args()
    
    # Create output directory and setup logging
    sample_dir = Path(args.sample_dir)
    logger = setup_logging(sample_dir)
    
    logger.info("=== SAMPLE SUPPLEMENTATION ===")
    logger.info(f"Sample directory: {sample_dir}")
    logger.info(f"Target sample size: {args.sample_size}")
    logger.info(f"Random seed: {args.seed}")
    
    # Check for existing sample and suggestions
    sample_df, suggestions_df = check_existing_sample(sample_dir, logger)
    
    if sample_df is None or suggestions_df is None:
        logger.error("Cannot proceed without existing sample and suggestions files")
        sys.exit(1)
    
    # Supplement sample
    updated_sample = supplement_sample(sample_df, suggestions_df, args.sample_size, args.seed, logger)
    
    # Save updated sample
    sample_file = sample_dir / 'sample.csv'
    updated_sample.to_csv(sample_file, index=False)
    logger.info(f"Updated sample saved to: {sample_file}")
    
    # Create backup of original if this is the first supplementation
    backup_file = sample_dir / 'sample_original.csv'
    if not backup_file.exists():
        sample_df.to_csv(backup_file, index=False)
        logger.info(f"Original sample backed up to: {backup_file}")

    # Print summary
    logger.info("\n=== SUPPLEMENTATION SUMMARY ===")
    logger.info(f"Original sample size: {len(sample_df)}")
    logger.info(f"Final sample size: {len(updated_sample)}")
    logger.info(f"Suggestions added: {len(updated_sample) - len(sample_df)}")
    logger.info(f"Target achieved: {len(updated_sample) >= args.sample_size}")
    
    if len(updated_sample) < args.sample_size:
        logger.warning(f"Could not reach target size of {args.sample_size}")
        logger.warning("Consider relaxing constraints or generating more suggestions")


if __name__ == '__main__':
    main()
