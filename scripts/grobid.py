from __future__ import annotations

import argparse
from pathlib import Path

from moralkg.config import Config
from moralkg.logging import get_logger
from moralkg.preprocessing.grobid import GrobidSettings, process_references


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GROBID processing steps")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_refs = sub.add_parser("process-references", help="Extract references with Grobid")
    p_refs.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Directory containing PDFs (default: philpapers.papers.pdfs.dir)",
    )
    p_refs.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output directory for TEI/XML (default: philpapers.papers.grobid.dir)",
    )
    p_refs.add_argument("--n", type=int, default=None, help="Number of worker threads")
    p_refs.add_argument("--force", action="store_true", help="Force overwrite outputs")
    p_refs.add_argument("--quiet", action="store_true", help="Reduce verbosity")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config.load()
    default_src = cfg.get("philpapers.papers.pdfs.dir")
    default_dst = cfg.get("philpapers.papers.grobid.dir")

    if args.cmd == "process-references":
        src_dir = args.src or Path(default_src)
        dst_dir = args.dst or Path(default_dst)
        settings = GrobidSettings.from_config(cfg)
        process_references(
            input_dir=src_dir,
            output_dir=dst_dir,
            settings=settings,
            n_workers=args.n,
            force=args.force,
            verbose=not args.quiet,
        )
        logger.info("Grobid processing complete: %s -> %s", src_dir, dst_dir)


if __name__ == "__main__":
    main()


