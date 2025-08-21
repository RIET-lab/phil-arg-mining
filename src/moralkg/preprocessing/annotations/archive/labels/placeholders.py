#!/usr/bin/env python

"""
placeholders.py

Create placeholder claims/premises for the sample size specified 
in the command-line argument. Placeholders are stored in 
`data/arguments/IDENTIFIER.json`.

Placeholder claims/premises are stored as a dict mapping claims/premises to the
extraction method. E.g.:
    { "This paper argues..." : "LLM 3.5" }

Usage: placholders.py -n [size]
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict

import pandas as pd
import rootutils 

ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add placeholder extracted claims/premises for a given sample size"
    )
    parser.add_argument(
        "-n", "--size", 
        type=int, 
        required=True,
        help="Sample size (e.g., 100, 150). Must already be sampled."
    )
    parser.add_argument(
        "-f", "--sample", 
        type=str, 
        help="Sample file CSV. Must be properly formatted with 'identifier' column. Defaults to `data/annotations/samples/n{sample_size}`."
    )
    return parser.parse_args()


def load_sample_data(sample_path: Path) -> pd.DataFrame | None:
    """Load sample data from CSV file."""
    try:
        df = pd.read_csv(sample_path)
        print(f"Loading sample data from: {sample_path}")
        return df
    except FileNotFoundError:
        print(f"Sample file not found. Path: {sample_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {textwrap.indent(str(e), '  ')}\n")
        return None


def create_placeholder_map(identifier: str) -> Dict[str, str]:
    """Create a placeholder map of claims to extraction methods for a given identifier."""
    placeholder_claims = {
        f"Placeholder claim 1 for {identifier}": "SAM",
        f"Placeholder claim 2 for {identifier}": "SciBART", 
        f"Placeholder claim 3 for {identifier}": "OAI 4o",
        f"Placeholder claim 4 for {identifier}": "Gemini 2.5",
        f"Placeholder claim 5 for {identifier}": "Claude 3.7"
    }
    return placeholder_claims


def save_placeholder_file(identifier: str, claims_map: Dict[str, str]) -> bool:
    """Save placeholder claims map to JSON file."""
    arguments_dir = ROOT / "data" / "arguments"    
    output_path = arguments_dir / f"{identifier}.json"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(claims_map, f, indent=2, ensure_ascii=False)
        print(f"  Saved placeholder map: {output_path}")
        return True
    except Exception as e:
        print(f"  Error saving placeholder map for '{identifier}': {textwrap.indent(str(e), '  ')}\n")
        return False


def create_placeholders_for_sample(df: pd.DataFrame) -> int:
    """Create placeholder files for all identifiers in the sample."""
    print("Creating placeholder claims/premises...")
    
    if 'identifier' not in df.columns:
        print("Error: Sample CSV must contain 'identifier' column")
        return False
        
    for _, row in df.iterrows():
        try:
            identifier = str(row['identifier'])
            claims_map = create_placeholder_map(identifier)
            save_placeholder_file(identifier, claims_map)
        except Exception as e:
            print(f"An error occurred: {textwrap.indent(str(e), '  ')}\n")
            return False
    
    return True


def main():
    """Main function to create placeholder claims/premises."""
    args = parse_args()
    
    # Construct path to sample file
    if args.sample:
        sample_path = Path(args.sample)
    else:
        sample_dir = ROOT / "data" / "annotations" / "samples" / f"n{args.size}"
        sample_path = sample_dir / "sample.csv"
    
    try:
        # Load sample data
        df = load_sample_data(sample_path)
        if df is None:
            return
        
        print(f"Found {len(df)} identifiers in sample")
        
        # Create placeholder files
        if create_placeholders_for_sample(df):
            print(f"Placeholder creation successfully completed.")
        else:
            print("Some placeholder files failed to create. Check for errors above.")
            return
            
    except Exception as e:
        print(f"An error occurred: {textwrap.indent(str(e), '  ')}\n")
        return


if __name__ == "__main__":
    main()