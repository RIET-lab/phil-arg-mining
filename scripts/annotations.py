#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

from moralkg import Config, get_logger
from moralkg.preprocessing.annotations import (
    create_sample,
    generate_annotations_for_sample,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotations pipeline orchestration")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_sample = sub.add_parser("sample", help="Create sampling CSV from metadata")
    s_sample.add_argument("--input-file", type=str, default=None)
    s_sample.add_argument("--output-dir", type=str, default=None)
    s_sample.add_argument("--sample-size", type=int, default=None)
    s_sample.add_argument("--seed", type=int, default=None)
    s_sample.add_argument("--allow-author-repeats", action="store_true")
    s_sample.add_argument("--allow-year-repeats", action="store_true")
    s_sample.add_argument("--allow-category-repeats", action="store_true")

    s_gen = sub.add_parser("prepare", help="Prepare output locations for labels/arguments")
    s_gen.add_argument("--sample-csv", type=str, default=None)
    s_gen.add_argument("--papers-dir", type=str, default=None)
    s_gen.add_argument("--labels-dir", type=str, default=None)
    s_gen.add_argument("--arguments-dir", type=str, default=None)
    s_gen.add_argument("--prefer-markdown", action="store_true")

    return p.parse_args()


def main() -> None:
    rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
    logger = get_logger("scripts.annotations")
    args = parse_args()

    if args.cmd == "sample":
        from moralkg.preprocessing.annotations.samples import SamplingConfig

        cfg = SamplingConfig(
            input_file=args.input_file,
            output_dir=args.output_dir,
            sample_size=args.sample_size or 0,
            seed=args.seed or 0,
            allow_author_repeats=args.allow_author_repeats,
            allow_year_repeats=args.allow_year_repeats,
            allow_category_repeats=args.allow_category_repeats,
        )
        out = create_sample(cfg)
        logger.info(f"Sample created: {out}")
    elif args.cmd == "prepare":
        from moralkg.preprocessing.annotations.labels import GenerationConfig

        cfg = GenerationConfig(
            sample_csv=args.sample_csv,
            papers_dir=args.papers_dir,
            labels_dir=args.labels_dir,
            arguments_dir=args.arguments_dir,
            prefer_markdown=args.prefer_markdown,
        )
        paths = generate_annotations_for_sample(cfg)
        logger.info(f"Prepared {len(paths)} paths for arguments/labels")


if __name__ == "__main__":
    main()


