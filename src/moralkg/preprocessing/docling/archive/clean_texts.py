#!/usr/bin/env python3

"""
Clean docling'd files (tranlate /uniXXXX ligature patterns that commonly appear).
"""

import argparse
import json
import logging
import os.path
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import langdetect


def preprocess_uni_patterns(text):
    """Convert /uniXXXX patterns to proper unicode characters."""

    # Common ligature mappings
    uni_mappings = {
        "FB01": "fi",
        "FB02": "fl",
        "FB03": "ffi",
        "FB04": "ffl",
        "FB05": "ft",
        "FB06": "st",
        # We can add more as needed
    }

    # Replace known ligature patterns
    for uni_code, replacement in uni_mappings.items():
        text = text.replace(f"/uni{uni_code}", replacement)

    # Handle any remaining /uniXXXX patterns by converting hex to Unicode
    def replace_uni_pattern(match):
        hex_code = match.group(1)
        try:
            unicode_char = chr(int(hex_code, 16))
            return unicode_char
        except ValueError:
            # If conversion fails, return original
            return match.group(0)

    text = re.sub(r"/uni([0-9A-Fa-f]{4})", replace_uni_pattern, text)

    # Fix apostrophe spacing issues
    # Handle patterns like "she ll ' make" -> "she'll make"
    text = re.sub(r"\b(\w+)\s+'\s*(\w+)", r"\1'\2", text)
    # Handle patterns like "it ' s" -> "it's"
    text = re.sub(r"\b(\w+)\s+'\s+s\b", r"\1's", text)
    # Handle patterns like "don ' t" -> "don't"
    text = re.sub(r"\b(\w+)\s+'\s+(\w+)", r"\1'\2", text)

    # Ensure periods and other punctuation are preceded by non-whitespace
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Normalize whitespace within paragraphs, but preserve paragraph breaks.
    # Split on double newlines (paragraphs), normalize each, then rejoin.
    paragraphs = re.split(r"(\n{2,})", text)
    normalized_paragraphs = []
    for part in paragraphs:
        if re.fullmatch(r"\n{2,}", part):
            # This is a paragraph break, keep as is
            normalized_paragraphs.append(part)
        else:
            # Normalize whitespace within the paragraph
            norm = re.sub(r"[ \t\r\f\v]+", " ", part)
            norm = re.sub(
                r" *\n *", " ", norm
            )  # Remove single newlines within paragraphs
            norm = norm.strip()
            normalized_paragraphs.append(norm)
    text = "".join(normalized_paragraphs)

    return text


def detect_language(text: str, min_length: int = 100) -> Union[str, None]:
    """
    Detect the language of a text.

    Args:
        text: The text to analyze
        min_length: Minimum text length for analysis

    Returns:
        Language code or None if detection failed
    """
    # Skip short texts
    if len(text) < min_length:
        return None

    try:
        return langdetect.detect(text)
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return None


class ProcessTracker:
    """Track processed files to enable resuming interrupted operations."""

    def __init__(self, tracker_file: str = "processed_files.json"):
        """Initialize the tracker with a tracker file path."""
        self.tracker_file = tracker_file
        self.processed_files: Set[str] = set()
        self.load_processed_files()

    def load_processed_files(self) -> None:
        """Load previously processed files from the tracker file."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, "r") as f:
                    data = json.load(f)
                    self.processed_files = set(data.get("processed_files", []))
                    logging.info(
                        f"Loaded {len(self.processed_files)} previously processed files"
                    )
            except Exception as e:
                logging.error(f"Error loading tracker file: {e}")
                self.processed_files = set()

    def is_processed(self, file_path: Union[str, Path]) -> bool:
        """Check if a file has already been processed."""
        return str(file_path) in self.processed_files

    def mark_processed(self, file_path: Union[str, Path]) -> None:
        """Mark a file as processed and update the tracker file."""
        self.processed_files.add(str(file_path))
        self._save_tracker()

    def _save_tracker(self) -> None:
        """Save the current state to the tracker file."""
        try:
            with open(self.tracker_file, "w") as f:
                json.dump(
                    {
                        "processed_files": list(self.processed_files),
                        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logging.error(f"Error saving tracker file: {e}")


class TextCleaner:
    """Text cleaner for processing files with Unicode ligature patterns."""

    def __init__(self, english_only: bool = False, skip_existing: bool = False):
        """
        Initialize the text cleaner.

        Args:
            english_only: If True, will skip files detected as non-English
            skip_existing: If True, will skip files that already exist in output location
        """
        self._setup_logging()
        self.english_only = english_only
        self.skip_existing = skip_existing
        self.tracker = ProcessTracker()

        # Statistics for language distribution
        self.stats = {
            "english": 0,
            "non_english": 0,
            "skipped_existing": 0,
            "errors": 0,
        }

    def _setup_logging(self):
        """Set up basic logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("text_cleaning.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """Clean the text using preprocess_uni_patterns."""
        self.logger.debug(f"Starting text cleaning for {len(text)} characters")
        cleaned_text = preprocess_uni_patterns(text)
        self.logger.debug(f"Cleaning completed")
        return cleaned_text

    def process_file(
        self, input_path: Path, output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Process a single file.

        Returns:
            Dict with processing result information including status and language
        """
        result = {
            "processed": False,
            "skipped": False,
            "reason": None,
            "language": None,
        }

        input_path = Path(input_path)

        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            result["reason"] = "file_not_found"
            self.stats["errors"] += 1
            return result

        # Check if file was already processed in a previous run
        if self.tracker.is_processed(input_path):
            self.logger.info(f"Skipping previously processed file: {input_path}")
            result["skipped"] = True
            result["reason"] = "already_processed"
            return result

        # Determine output path
        if output_path is None:
            output_path = (
                input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
            )

        # Check if output file already exists and we're skipping existing files
        if self.skip_existing and output_path.exists():
            self.logger.info(f"Skipping existing output file: {output_path}")
            result["skipped"] = True
            result["reason"] = "output_exists"
            self.stats["skipped_existing"] += 1
            return result

        self.logger.info(f"Processing file: {input_path} -> {output_path}")

        # Read file with encoding detection fallback
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            self.logger.warning("UTF-8 decoding failed, trying latin-1")
            try:
                with open(input_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                self.logger.error(f"Failed to read file: {e}")
                result["reason"] = "read_error"
                self.stats["errors"] += 1
                return result

        # Check language
        lang = detect_language(content)
        result["language"] = lang

        # Skip non-English files if requested
        if self.english_only and lang and lang != "en":
            self.logger.info(
                f"Skipping non-English file: {input_path} (detected: {lang})"
            )
            result["skipped"] = True
            result["reason"] = "non_english"
            self.stats["non_english"] += 1
            return result

        # Update language stats
        if lang == "en":
            self.stats["english"] += 1
        elif lang is not None:
            # Count as non-English even if we're processing it (english_only=False)
            self.stats["non_english"] += 1

        # Clean the text
        cleaned_content = self.clean_text(content)

        # Write cleaned content
        try:
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            self.logger.info(f"Cleaned file written to: {output_path}")

            # Mark as processed for future runs
            self.tracker.mark_processed(input_path)

            result["processed"] = True
            return result

        except Exception as e:
            self.logger.error(f"Error writing output file: {e}")
            result["reason"] = "write_error"
            self.stats["errors"] += 1
            return result

    def process_batch(
        self,
        input_paths: List[Path],
        output_dir: Optional[Path] = None,
        parallel: bool = False,
        workers: int = 4,
    ) -> Dict[str, int]:
        """
        Process multiple files with optional parallelization.

        Returns:
            Dict with counts of processed and skipped files
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if parallel and workers > 1:
            # Parallel processing is handled differently since we can't directly
            # update the shared stats dictionary from worker processes
            self.logger.info(f"Starting parallel processing with {workers} workers")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []

                for input_path in input_paths:
                    output_path = None
                    if output_dir:
                        output_path = (
                            output_dir / f"{input_path.stem}_cleaned{input_path.suffix}"
                        )

                    # Check if we should skip this file based on tracker or existing file
                    if self.tracker.is_processed(input_path):
                        self.logger.debug(
                            f"Skipping previously processed file: {input_path}"
                        )
                        continue

                    if self.skip_existing and output_path.exists():
                        self.logger.debug(
                            f"Skipping existing output file: {output_path}"
                        )
                        self.stats["skipped_existing"] += 1
                        continue

                    futures.append(
                        executor.submit(
                            process_file_standalone,
                            str(input_path),
                            str(output_path) if output_path else None,
                            self.english_only,
                            self.skip_existing,
                        )
                    )

                # Collect results from all futures
                language_counts = {}
                results = {"processed": 0, "skipped": 0, "languages": {}}

                for future in as_completed(futures):
                    try:
                        result = future.result()

                        # Update result counts
                        if result.get("processed"):
                            results["processed"] += 1
                        elif result.get("skipped"):
                            results["skipped"] += 1

                        # Track language statistics
                        lang = result.get("language")
                        if lang:
                            if lang in language_counts:
                                language_counts[lang] += 1
                            else:
                                language_counts[lang] = 1

                    except Exception as e:
                        self.logger.error(f"Error in parallel processing: {e}")
                        results["skipped"] += 1

                # Update language statistics from the parallel runs
                for lang, count in language_counts.items():
                    if lang == "en":
                        self.stats["english"] += count
                    else:
                        self.stats["non_english"] += count

                return {**self.stats, **results, "languages": language_counts}
        else:
            # Sequential processing - directly updates self.stats
            for input_path in input_paths:
                try:
                    output_path = None
                    if output_dir:
                        output_path = (
                            output_dir / f"{input_path.stem}_cleaned{input_path.suffix}"
                        )

                    # Process the file and update stats
                    self.process_file(input_path, output_path)

                except Exception as e:
                    self.logger.error(f"Error processing {input_path}: {e}")
                    self.stats["errors"] += 1

            # Calculate language percentages
            total_with_lang = self.stats["english"] + self.stats["non_english"]
            language_percentages = {}
            if total_with_lang > 0:
                language_percentages["english"] = (
                    self.stats["english"] / total_with_lang
                ) * 100
                language_percentages["non_english"] = (
                    self.stats["non_english"] / total_with_lang
                ) * 100

            return {
                **self.stats,
                "total": len(input_paths),
                "percentages": language_percentages,
            }


def process_file_standalone(
    input_path: str,
    output_path: Optional[str],
    english_only: bool = False,
    skip_existing: bool = False,
) -> Dict[str, any]:
    """
    Standalone file processing function for multiprocessing.
    Returns result dictionary with processing status and language info.
    """
    cleaner = TextCleaner(english_only=english_only, skip_existing=skip_existing)
    return cleaner.process_file(
        Path(input_path), Path(output_path) if output_path else None
    )


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Docling'd paper cleaner")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "-b", "--batch", action="store_true", help="Process directory in batch mode"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-e", "--english-only", action="store_true", help="Skip non-English files"
    )
    parser.add_argument(
        "-s",
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the output location",
    )
    parser.add_argument(
        "-t",
        "--tracker-file",
        default="processed_files.json",
        help="Path to the tracker file for resuming interrupted operations",
    )

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize cleaner with options
        cleaner = TextCleaner(
            english_only=args.english_only, skip_existing=args.skip_existing
        )

        # Set tracker file if provided
        if args.tracker_file:
            cleaner.tracker.tracker_file = args.tracker_file
            cleaner.tracker.load_processed_files()

        input_path = Path(args.input)

        if args.batch or input_path.is_dir():
            # Batch processing
            if not input_path.is_dir():
                print("Error: Batch mode requires a directory input")
                sys.exit(1)

            # Find all text files
            input_files = []
            for ext in ["*.txt", "*.md"]:
                input_files.extend(input_path.glob(ext))

            if not input_files:
                print(f"No text files found in {input_path}")
                sys.exit(1)

            print(f"Found {len(input_files)} files to process")

            output_dir = Path(args.output) if args.output else input_path / "cleaned"

            start_time = time.time()
            results = cleaner.process_batch(
                input_files, output_dir, parallel=True, workers=args.jobs
            )
            elapsed_time = time.time() - start_time

            # Print summary statistics
            print(f"\nBatch processing complete in {elapsed_time:.2f}s")
            print(f"- Total files found: {len(input_files)}")
            print(f"- Files processed: {results.get('processed', 0)}")

            # Print language statistics if available
            if "percentages" in results:
                print("\nLanguage Statistics:")
                total_with_lang = results.get("english", 0) + results.get(
                    "non_english", 0
                )

                if total_with_lang > 0:
                    en_percent = results["percentages"].get("english", 0)
                    non_en_percent = results["percentages"].get("non_english", 0)
                    print(
                        f"- English files: {results.get('english', 0)} ({en_percent:.1f}%)"
                    )
                    print(
                        f"- Non-English files: {results.get('non_english', 0)} ({non_en_percent:.1f}%)"
                    )

            # Print language breakdown from parallel processing
            if "languages" in results:
                print("\nDetected Languages:")
                for lang, count in sorted(
                    results["languages"].items(), key=lambda x: x[1], reverse=True
                ):
                    percent = (count / sum(results["languages"].values())) * 100
                    print(f"- {lang}: {count} ({percent:.1f}%)")

            # Additional statistics
            if results.get("skipped_existing", 0) > 0:
                print(f"\nSkipped {results['skipped_existing']} existing files")

            if results.get("errors", 0) > 0:
                print(f"\nEncountered {results['errors']} errors during processing")

            print(f"\nOutput location: {output_dir}")

        else:
            # Single file processing
            output_path = Path(args.output) if args.output else None
            result = cleaner.process_file(input_path, output_path)

            if result["processed"]:
                print(f"File successfully processed")
                if result["language"]:
                    print(f"Detected language: {result['language']}")
            elif result["skipped"]:
                reason = result["reason"]
                if reason == "non_english":
                    print(
                        f"File skipped: non-English content detected ({result['language']})"
                    )
                elif reason == "output_exists":
                    print(f"File skipped: output file already exists")
                elif reason == "already_processed":
                    print(f"File skipped: already processed in a previous run")
                else:
                    print(f"File skipped: {reason}")

    except KeyboardInterrupt:
        print("\nProcessing cancelled by user")
        print("You can resume processing later with the same tracker file.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
