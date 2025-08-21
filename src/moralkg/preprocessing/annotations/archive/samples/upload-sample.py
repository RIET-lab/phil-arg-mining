#!/usr/bin/env python

"""
upload-sample.py

Uploads the papers data associated with the identifiers located in 
`data/annotations/samples/nX/sample.csv` to the `moral-kg-sample` HF dataset.

The dataset is in the format:
- identifier    | str       | The Phil-Papers ID associated with each paper
- title         | str       | The title of the paper
- authors       | list:str  | The authors attributed to the paper
- year          | str       | The publication year of the paper
- categories    | list:str  | The category_names attributed to the paper
- map           | dict      | The claim:method map that contains each claim  
                              extracted from the text and its associated 
                              extraction method.
- text          | str       | The paper content (in plain text or markdown)


Usage: python upload-sample.py -n [size] [--overwrite]
"""

import argparse
import json
import textwrap
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
import rootutils

ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload papers sample to the moral-kg-sample HF dataset"
    )
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        required=True,
        help="Sample size (e.g., 100, 150). Must already be sampled.",
    )
    parser.add_argument(
        "-f",
        "--sample",
        type=str,
        help=(
            "Sample file CSV. Must be properly formatted with 'identifier' column."
            + " Defaults to `data/annotations/samples/n{sample_size}/sample.csv`."
        ),
    )
    parser.add_argument(
        "-x",
        "--overwrite",
        action="store_true",
        help="Overwrite the existing dataset (default: append to dataset)",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default="data/metadata/2025-07-09-en-combined-metadata.csv",
        help="File that contains the paper metadata.",
    )
    parser.add_argument(
        "-p",
        "--papers",
        default="data/docling",
        help="Directory that contains the paper files (.txt or .md)",
    )
    parser.add_argument(
        "-a",
        "--arguments",
        default="data/annotations/labels",
        help="Directory that contains the argument maps (claim:method)",
    )
    parser.add_argument(
        "-md",
        "--markdown",
        action="store_true",
        help="Use .md instead of .txt files (default: .txt, fallback to .md)",
    )
    parser.add_argument(
        "-s",
        "--skip",
        action="store_true",
        help="If a resource file attached to a paper is not found (papers or arguments) skip uploading that paper and move on.",
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
        print(f"An error occurred: {textwrap.indent(str(e), '  ') + '\n'}")
        return None


def load_metadata(metadata_path: Path) -> pd.DataFrame | None:
    """Load metadata from CSV file."""
    try:
        df = pd.read_csv(metadata_path)
        print(f"Loading metadata from: {metadata_path}")
        return df
    except FileNotFoundError:
        print(f"Metadata file not found. Path: {metadata_path}")
        return None
    except Exception as e:
        print(f"An error occurred loading metadata: {textwrap.indent(str(e), '  ')}\n")
        return None


def prepare_dataset_records(
    sample_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    papers_dir: Path,
    arguments_dir: Path,
    use_markdown: bool = False,
    skip_file_not_found: bool = False,
) -> List[Dict]:
    """Prepare records for HuggingFace dataset format by collating data from multiple sources."""
    records = []

    # Create a metadata lookup dictionary for faster access
    metadata_lookup = {}
    for _, row in metadata_df.iterrows():
        identifier = str(row.get("identifier", ""))
        if identifier:
            metadata_lookup[identifier] = {
                "title": str(row.get("title", "")),
                "authors": str(row.get("authors", "")),
                "year": str(row.get("year", "")),
                "categories": [
                    cat.strip()
                    for cat in str(row.get("category_names", "")).split(";")
                    if cat.strip()
                ],
            }

    for _, row in sample_df.iterrows():
        identifier = str(row.get("identifier", ""))
        if not identifier:
            print(f"Warning: Skipping row with empty identifier")
            continue

        # Get metadata
        metadata = metadata_lookup.get(identifier, {})
        title = metadata.get("title", "")
        authors = metadata.get("authors", "")
        year = metadata.get("year", "")
        categories = metadata.get("categories", "")

        # Get text content
        text = ""
        text_file_found = False

        # Try preferred format first (txt by default, md if --markdown flag)
        if use_markdown:
            primary_ext = ".md"
            fallback_ext = ".txt"
        else:
            primary_ext = ".txt"
            fallback_ext = ".md"

        primary_file = papers_dir / f"{identifier}{primary_ext}"
        fallback_file = papers_dir / f"{identifier}{fallback_ext}"

        try:
            if primary_file.exists():
                with open(primary_file, "r", encoding="utf-8") as f:
                    text = f.read()
                text_file_found = True
            elif fallback_file.exists():
                print(f"Warning: {primary_file} not found, using {fallback_file}")
                with open(fallback_file, "r", encoding="utf-8") as f:
                    text = f.read()
                text_file_found = True
            else:
                print(f"Error: Neither {primary_file} nor {fallback_file} found")
                if skip_file_not_found:
                    print(f"WARNING: Skipping {identifier} due to missing text file")
                    continue
        except Exception as e:
            print(f"Error reading text file for {identifier}: {e}")
            if skip_file_not_found:
                print(f"WARNING: Skipping {identifier} due to text file read error")
                continue

        # Get claims/premises
        map_data = {}
        arguments_file = arguments_dir / f"{identifier}.json"

        try:
            if arguments_file.exists():
                with open(arguments_file, "r", encoding="utf-8") as f:
                    map_data = json.load(f)
                print(f"Loaded map data for {identifier} from {arguments_file.name}")
            else:
                print(f"Warning: Arguments file not found: {arguments_file}")
                if skip_file_not_found:
                    print(
                        f"WARNING: Skipping {identifier} due to missing arguments file"
                    )
                    continue
        except Exception as e:
            print(f"Error reading arguments file for {identifier}: {e}")
            if skip_file_not_found:
                print(
                    f"WARNING: Skipping {identifier} due to arguments file read error"
                )
                continue

        # Create record in the specified order: identifier, title, authors, year, categories, map, text
        record = {
            "identifier": identifier,
            "title": title,
            "authors": authors,
            "year": year,
            "categories": categories,
            "map": map_data,
            "text": text,
        }
        records.append(record)

    return records


def upload_to_huggingface(
    records: List[Dict],
    sample_size: int,
    overwrite: bool = False,
) -> bool:
    """Upload records to HuggingFace dataset."""
    dataset_name = "RIET-lab/moral-kg-sample"

    # Convert map dictionaries to str
    for record in records:
        record["map"] = json.dumps(record["map"])

    # Create dataset from records
    dataset = Dataset.from_list(records)

    if overwrite:
        try:
            dataset.push_to_hub(
                dataset_name,
                private=True,
                commit_message=f"Overwrote sample (n={sample_size})",
            )
            print(f"Overwrote moral-kg-sample with new sample (n={sample_size})")
            return True
        except Exception as e:
            print(
                f"Could not overwrite dataset: {textwrap.indent(str(e), '  ') + '\n'}"
            )
            return False
    else:
        try:
            # Load existing dataset and concatenate
            existing_dataset = load_dataset(dataset_name, split="train")
            existing_dataset = existing_dataset  # type: ignore
            combined_dataset = concatenate_datasets([existing_dataset, dataset])  # type: ignore

            combined_dataset.push_to_hub(
                dataset_name,
                private=True,
                commit_message=f"Appended sample (n={sample_size})",
            )
            print(f"Appended moral-kg-sample with new sample (n={sample_size})")
            return True
        except Exception as e:
            print(
                f"Could not append to dataset: {textwrap.indent(str(e), '  ') + '\n'}"
            )
            return False


def main():
    """Main function to execute the upload process."""
    args = parse_args()

    # Construct paths
    if args.sample:
        sample_file = Path(args.sample)
    else:
        sample_dir = ROOT / "data" / "annotations" / "samples" / f"n{args.size}"
        sample_file = sample_dir / "sample.csv"

    metadata_file = ROOT / args.metadata
    papers_dir = ROOT / args.papers
    arguments_dir = ROOT / args.arguments

    try:
        # Load sample data
        sample_df = load_sample_data(sample_file)
        if sample_df is None:
            print("Failed to load sample data. Exiting.")
            sys.exit(1)

        # Load metadata
        metadata_df = load_metadata(metadata_file)
        if metadata_df is None:
            print("Failed to load metadata. Exiting.")
            sys.exit(1)

        print(f"Sample data shape: {sample_df.shape}")
        print(f"Metadata shape: {metadata_df.shape}")
        print(sample_df.head().to_string())

        # Prepare records for HuggingFace format
        records = prepare_dataset_records(
            sample_df=sample_df,
            metadata_df=metadata_df,
            papers_dir=papers_dir,
            arguments_dir=arguments_dir,
            use_markdown=args.markdown,
            skip_file_not_found=args.skip,
        )

        # Upload to HuggingFace
        result = upload_to_huggingface(
            records=records, sample_size=args.size, overwrite=args.overwrite
        )

        if result:
            print("Sample upload completed successfully.")
            print(f"Mode: {'Overwrite' if args.overwrite else 'Append'}")
            print(f"Records uploaded: {len(records)}")
        else:
            print("Sample upload not completed. Check for errors.")

    except Exception as e:
        print(f"An error occurred: {textwrap.indent(str(e), '  ') + '\n'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
