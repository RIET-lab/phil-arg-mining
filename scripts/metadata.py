#!/usr/bin/env python3
"""
End-to-end PhilPapers/PhilArchive metadata pipeline.

Steps (toggle with flags):
  1) Harvest OAI-PMH XML
  2) Parse XML to CSV
  3) Parse category JSONs to CSVs
  4) Combine parsed CSV with categories
  5) Reformat combined CSV (column order, normalization)

Defaults read from config.yaml:
  - philpapers.metadata.dir: base directory for metadata files
  - philpapers.metadata.file: explicit CSV to use when combining (optional)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from moralkg import Config
from moralkg.preprocessing import metadata as pp
import yaml


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _meta_dir(cfg: Config) -> Path:
    md = cfg.get("philpapers.metadata.dir")
    if not md:
        raise ValueError("Missing 'philpapers.metadata.dir' in config.yaml")
    p = Path(md)
    if not p.is_absolute():
        # Resolve relative to repo root using Config's ROOT logic
        from pathlib import Path as _P
        import rootutils as _r
        _ROOT = _P(_r.setup_root(__file__, indicator=".git"))
        p = _ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PhilPapers metadata pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Step toggles
    p.add_argument("--harvest", action="store_true", help="Run OAI-PMH harvesting")
    p.add_argument("--parse", action="store_true", help="Parse harvested XML to CSV")
    p.add_argument("--parse-categories", action="store_true", help="Parse categories JSONs to CSVs")
    p.add_argument("--combine", action="store_true", help="Combine parsed CSV with categories")
    p.add_argument("--reformat", action="store_true", help="Reformat combined CSV")
    p.add_argument("--all", action="store_true", help="Run all steps (harvest, parse, parse-categories, combine, reformat)")

    # Harvest options
    p.add_argument("--server-url", default="https://api.philpapers.org/oai.pl", help="OAI-PMH server URL")
    p.add_argument("--from-date", default=None, help="Harvest from date (YYYY-MM-DD)")
    p.add_argument("--until-date", default=None, help="Harvest until date (YYYY-MM-DD)")
    p.add_argument("--metadata-prefix", default="oai_dc", help="OAI metadataPrefix")
    p.add_argument("--set", dest="oai_set", default=None, help="OAI set name")
    p.add_argument("--xml-out", default=None, help="Output XML path (default: <dir>/phil-papers/<YYYY-MM-DD>.xml)")

    # Parse options
    p.add_argument("--input-glob", default=None, help="Glob of XML files to parse (defaults to <dir>/phil-papers/*.xml or *.xml)")
    p.add_argument("--schema", default=None, help="Path to metadata schema YAML/JSON")
    p.add_argument("--languages", nargs="+", default=None, help="Filter dc:language codes (e.g. en fr)")
    p.add_argument("--start-date", default=None, help="Parse start datestamp (YYYY-MM-DD)")
    p.add_argument("--end-date", default=None, help="Parse end datestamp (YYYY-MM-DD)")
    p.add_argument("--record-range", nargs=2, type=int, metavar=("START", "END"), default=None, help="Inclusive record indices (1-based)")
    p.add_argument("-n", "--limit", type=int, default=None, help="Max records to parse (alias for --record-range 1 N)")
    p.add_argument("--exclude-deleted", action="store_true", help="Exclude records with header status='deleted'")
    p.add_argument("--parsed-out", default=None, help="Output parsed CSV path (default: <dir>/phil-papers/<YYYY-MM-DD>-en.csv)")

    # Categories options
    p.add_argument("--categories-json", default=None, help="Path to categories.json")
    p.add_argument("--archive-categories-json", default=None, help="Path to archive_categories.json")
    p.add_argument("--categories-csv-out", default=None, help="Output categories CSV path")
    p.add_argument("--paper-categories-csv-out", default=None, help="Output paper_categories CSV path")

    # Combine options
    p.add_argument("--metadata-csv", default=None, help="Parsed metadata CSV to combine (default: from --parsed-out or config 'philpapers.metadata.file')")
    p.add_argument("--combined-out", default=None, help="Output combined CSV path (default: <dir>/<YYYY-MM-DD>-en-combined-metadata.csv)")
    p.add_argument("--include-hierarchy", action="store_true", help="Include category hierarchy columns")
    p.add_argument("--include-parent-info", action="store_true", help="Include parent category info columns")

    # Reformat options
    p.add_argument("--reformatted-out", default=None, help="Output reformatted CSV path (default: <dir>/<YYYY-MM-DD>-en-combined-metadata-reformatted.csv)")

    return p.parse_args()


def run():
    args = parse_args()
    cfg = Config.load()
    base_dir = _meta_dir(cfg)
    date_tag = _today()

    # Defaults
    philpapers_dir = base_dir / "phil-papers"
    philpapers_dir.mkdir(parents=True, exist_ok=True)

    xml_out = Path(args.xml_out) if args.xml_out else (philpapers_dir / f"{date_tag}.xml")
    parsed_out = Path(args.parsed_out) if args.parsed_out else (philpapers_dir / f"{date_tag}-en.csv")
    categories_csv_out = Path(args.categories_csv_out) if args.categories_csv_out else (base_dir / "categories_parsed.csv")
    paper_categories_csv_out = Path(args.paper_categories_csv_out) if args.paper_categories_csv_out else (base_dir / "paper_categories_parsed.csv")
    combined_out = Path(args.combined_out) if args.combined_out else (base_dir / f"{date_tag}-en-combined-metadata.csv")
    reformatted_out = Path(args.reformatted_out) if args.reformatted_out else (base_dir / f"{date_tag}-en-combined-metadata-reformatted.csv")

    # Select steps
    do_all = args.all
    do_harvest = args.harvest or do_all
    do_parse = args.parse or do_all
    do_parse_categories = args.parse_categories or do_all
    do_combine = args.combine or do_all
    do_reformat = args.reformat or do_all

    # 1) Harvest
    if do_harvest:
        print("Harvesting OAI-PMH XML...")
        pp.Harvester().harvest(
            server_url=args.server_url,
            output_path=str(xml_out),
            from_date=args.from_date,
            until_date=args.until_date,
            metadata_prefix=args.metadata_prefix,
            set_name=args.oai_set,
            verbose=True,
        )
        print(f"  -> Wrote: {xml_out}")

    # 2) Parse XML
    if do_parse:
        print("Parsing XML to CSV...")
        record_range = tuple(args.record_range) if args.record_range else ((1, args.limit) if args.limit else None)

        filters = pp.ParseFilters(
            start_date=datetime.fromisoformat(args.start_date) if args.start_date else None,
            end_date=datetime.fromisoformat(args.end_date) if args.end_date else None,
            languages=args.languages,
            include_deleted=not args.exclude_deleted,
            record_range=record_range,
        )
        count = pp.parse_metadata(
            input_glob=args.input_glob or str(philpapers_dir / "*.xml"),
            output_csv=str(parsed_out),
            schema=None if not args.schema else None,  # let file-based schema be passed later if needed
            filters=filters,
        )
        print(f"  Wrote: {parsed_out} ({count} records)")

    # 3) Parse categories JSONs
    if do_parse_categories:
        print("Parsing category JSONs...")
        parser = pp.PhilPapersParser()
        # Default JSONs if not provided
        categories_json = Path(args.categories_json) if args.categories_json else (base_dir / "categories.json")
        archive_json = Path(args.archive_categories_json) if args.archive_categories_json else (base_dir / "archive_categories.json")
        if categories_json.exists():
            parser.parse_categories(str(categories_json))
            parser.export_categories_csv(str(categories_csv_out))
            print(f"  Wrote: {categories_csv_out}")
        else:
            print(f"  categories.json not found: {categories_json}")
        if archive_json.exists():
            parser.parse_archive_categories(str(archive_json))
            parser.export_paper_categories_csv(str(paper_categories_csv_out))
            print(f"  Wrote: {paper_categories_csv_out}")
        else:
            print(f"  archive_categories.json not found: {archive_json}")

    # 4) Combine
    if do_combine:
        print("  Combining parsed metadata with categories...")
        # Prefer explicit path, then parsed_out, then Config override
        metadata_csv = Path(args.metadata_csv) if args.metadata_csv else (
            parsed_out if parsed_out.exists() else None
        )
        if metadata_csv is None or not metadata_csv.exists():
            # Try config override
            cfg_file = cfg.get("philpapers.metadata.file")
            if cfg_file:
                metadata_csv = Path(cfg_file)
                if not metadata_csv.is_absolute():
                    metadata_csv = base_dir / metadata_csv
        if metadata_csv is None or not metadata_csv.exists():
            raise FileNotFoundError("Could not determine metadata CSV to combine. Use --metadata-csv, run --parse, or set config philpapers.metadata.file")

        ok = pp.combine_metadata(
            categories_csv=str(categories_csv_out),
            paper_categories_csv=str(paper_categories_csv_out),
            metadata_csv=str(metadata_csv),
            output_csv=str(combined_out),
            include_hierarchy=args.include_hierarchy,
            include_parent_info=args.include_parent_info,
        )
        if not ok:
            raise RuntimeError("Combine step failed")
        print(f"  Wrote: {combined_out}")

    # 5) Reformat
    if do_reformat:
        print("Reformatting combined CSV...")
        src = combined_out if combined_out.exists() else parsed_out
        if not src.exists():
            raise FileNotFoundError(f"No input found to reformat: {src}")
        pp.reformat_combined_metadata(str(src), str(reformatted_out))
        print(f"  Wrote: {reformatted_out}")

    print("Done.")


if __name__ == "__main__":
    run()

