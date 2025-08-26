#!/usr/bin/env python3

"""
Unified CLI for Docling PDF processing.

This script composes reusable APIs in `src/moralkg/preprocessing/docling/` to:
- list/select input PDFs
- process in parallel with optional GPU
- export Markdown (default) and optionally other formats
- track failures to a JSON record

Example:
  python scripts/docling.py -i data/pdfs -o data/docling --skip-existing --use-gpu --num-workers 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moralkg.preprocessing.docling import (
    ExportOptions,
    list_input_files,
    process_parallel,
)
from moralkg.preprocessing.docling.filter import DoclingTextFilter


def parse_args():
    p = argparse.ArgumentParser(
        description="Docling PDF processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input-dir", type=Path, required=True, help="PDF input dir")
    p.add_argument("-o", "--output-dir", type=Path, required=True, help="Output dir")
    p.add_argument("--files", nargs="+", default=None, help="Specific filenames inside input dir")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output already exists")
    p.add_argument("--num-workers", type=int, default=None, help="Worker processes (defaults to CPU cores)")
    p.add_argument("--chunk-size", type=int, default=1, help="Files per task chunk")
    p.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    p.add_argument("--gpu-id", type=int, default=None, help="Specific GPU id to use")
    p.add_argument("--failure-record", type=Path, default=None, help="Path to failed_files.json")
    p.add_argument("--write-text", action="store_true", help="Also export plain text .txt")
    p.add_argument("--write-json", action="store_true", help="Also export .json")
    p.add_argument("--write-html", action="store_true", help="Also export .html")
    p.add_argument("--write-yaml", action="store_true", help="Also export .yaml")
    p.add_argument("--write-doctags", action="store_true", help="Also export doctags")
    p.add_argument("--filter", action="store_true", help="Post-process outputs with docling filter")
    p.add_argument("--filter-ext", nargs="+", default=["md"], help="Extensions to filter when --filter is set")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    inputs = list_input_files(
        input_dir=args.input_dir,
        specific_files=args.files,
        skip_existing=args.skip_existing,
        output_dir=args.output_dir,
        expected_ext="md",
    )

    if not inputs:
        logging.info("No input PDFs to process")
        return 0

    export_opts = ExportOptions(
        write_markdown=True,
        write_text=args.write_text,
        write_json=args.write_json,
        write_html=args.write_html,
        write_yaml=args.write_yaml,
        write_doctags=args.write_doctags,
    )

    success, partial, failed = process_parallel(
        inputs=inputs,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
        failure_record_path=args.failure_record,
        export_options=export_opts,
    )

    if args.filter:
        cleaner = DoclingTextFilter(overwrite=True)
        total = cleaner.process_directory(args.output_dir, output_dir=None, exts=args.filter_ext)
        logging.info(f"Filtered {total} files in {args.output_dir}")

    logging.info(f"Done. Success: {success}, Partial: {partial}, Failed: {failed}, Total: {len(inputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


