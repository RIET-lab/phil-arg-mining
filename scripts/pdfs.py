from __future__ import annotations

import argparse
from datetime import datetime
import rootutils
from pathlib import Path
from typing import List, Optional

# Ensure repo root and add src to PYTHONPATH
rootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from moralkg.config import Config
from moralkg.preprocessing.pdfs import DownloadFilters, download_pdfs
from moralkg.logging import get_logger


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download PhilPapers/PhilArchive PDFs using config-driven paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # IO options
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input CSV file. If omitted, auto-select most recent in configured directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PDFs. Defaults to config philpapers.papers.pdfs.dir.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML",
    )

    # Identifier options
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Explicit identifiers to download (skips reading metadata CSV)",
    )

    # Filtering options
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=None,
        help="Language codes to include (matches substrings in language column)",
    )
    parser.add_argument(
        "-s",
        "--start-date",
        type=str,
        default=None,
        help="ISO start date (inclusive). Compared to parsed year column.",
    )
    parser.add_argument(
        "-e",
        "--end-date",
        type=str,
        default=None,
        help="ISO end date (inclusive). Compared to parsed year column.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Maximum number of papers to download",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already-downloaded PDFs",
    )

    # Processing options
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Sleep time between requests (seconds)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before limiting/downloading",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = Config.load(Path(args.config))
    logger = get_logger("scripts.pdfs")

    shuffle = not args.no_shuffle
    filters = DownloadFilters(
        languages=args.languages,
        start_date=_parse_datetime(args.start_date),
        end_date=_parse_datetime(args.end_date),
        limit=args.limit,
        shuffle=shuffle,
        skip_existing=args.skip_existing,
        sleep_time=args.sleep_time,
    )

    input_csv = Path(args.input) if args.input else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    success, total = download_pdfs(
        cfg,
        input_csv=input_csv,
        output_dir=output_dir,
        filters=filters,
        identifiers=args.ids,
    )
    logger.info("Completed: %s/%s PDFs downloaded successfully", success, total)


if __name__ == "__main__":
    main()
