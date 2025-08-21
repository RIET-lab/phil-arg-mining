# parse-papers.py

# Use this script to parse the papers in moralkg/data/pdfs into moralkg/data/docling.

# Note: This uses the docling batch suggested implementation almost verbatim.
# The only changes are to the imports and the output directory.

# Further note: File modified s.t. the following arguments can be specified:
# - the I/O directories
# - the specific input files to process
# - T/F if existing processed papers should be skipped

import argparse
from dspipe import Pipe
from docling.document_converter import DocumentConverter
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
import yaml
from docling_core.types.doc.base import ImageRefMode
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


_log = logging.getLogger(__name__)

USE_V2 = True
USE_LEGACY = False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse PDF papers using Docling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # IO options
    parser.add_argument('-i', '--input-dir', default="/opt/extra/avijit/projects/moralkg/data/pdfs",
                       help="Input directory containing PDF files")
    parser.add_argument('-o', '--output-dir', default="/opt/extra/avijit/projects/moralkg/data/docling",
                       help="Output directory for parsed documents (.txt, .md, etc.)")
    
    # File selection options
    parser.add_argument('--files', nargs='+', default=None,
                       help="Specific PDF filenames within input directory to process (include .pdf extension). Defaults to all.")
    
    # Processing options
    parser.add_argument('--skip-existing', action='store_true',
                       help="Skip processing files that already exist in the output directory")
    
    return parser.parse_args()

def should_skip_file(pdf_path, output_dir, skip_existing):
    """Check if file should be skipped based on existing outputs"""
    if not skip_existing:
        return False
    
    doc_filename = pdf_path.stem
    
    # check if the files exists as a .txt file.
    output_files = [
        output_dir / f"{doc_filename}.txt"
        # can add other formats here if desired.
    ]
    
    return any(f.exists() for f in output_files)

def get_input_files(args):
    """Get list of PDF files to process based on arguments"""
    input_dir = Path(args.input_dir)
    
    if args.files:
        # Process specific files within input directory
        input_doc_paths = []
        for filename in args.files:
            file_path = input_dir / filename
            if file_path.exists() and file_path.suffix.lower() == '.pdf':
                input_doc_paths.append(file_path)
            else:
                _log.warning(f"File not found or not a PDF: {filename}")
    else:
        # Process all PDFs in input directory
        input_doc_paths = list(input_dir.glob("*.pdf"))
    
    if not input_doc_paths:
        raise RuntimeError(f"No PDF files found to process")
    
    # Apply skip existing filter
    if args.skip_existing:
        output_dir = Path(args.output_dir)
        original_count = len(input_doc_paths)
        input_doc_paths = [p for p in input_doc_paths if not should_skip_file(p, output_dir, True)]
        skipped_count = original_count - len(input_doc_paths)
        if skipped_count > 0:
            _log.info(f"Skipped {skipped_count} files with existing outputs")
    
    return input_doc_paths

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                conv_res.document.save_as_json(
                    output_dir / f"{doc_filename}.json",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_html(
                    output_dir / f"{doc_filename}.html",
                    image_mode=ImageRefMode.EMBEDDED,
                )
                conv_res.document.save_as_doctags(
                    output_dir / f"{doc_filename}.doctags.txt"
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.md",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.txt",
                    image_mode=ImageRefMode.PLACEHOLDER,
                    strict_text=True,
                )

                # Export Docling document format to YAML:
                with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

                # Export Docling document format to doctags:
                with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_doctags())

                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

                # Export Docling document format to text:
                with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

            if USE_LEGACY:
                # Export Deep Search document JSON format:
                with (output_dir / f"{doc_filename}.legacy.json").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))

                # Export Text format:
                with (output_dir / f"{doc_filename}.legacy.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(
                        conv_res.legacy_document.export_to_markdown(strict_text=True)
                    )

                # Export Markdown format:
                with (output_dir / f"{doc_filename}.legacy.md").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown())

                # Export Document Tags format:
                with (output_dir / f"{doc_filename}.legacy.doctags.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_document_tokens())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    input_doc_paths = get_input_files(args)
    output_dir = Path(args.output_dir)
    
    _log.info(f"Processing {len(input_doc_paths)} PDF files")
    _log.info(f"Input directory: {args.input_dir}")
    _log.info(f"Output directory: {output_dir}")

    # buf = BytesIO(Path("./test/data/2206.01062.pdf").open("rb").read())
    # docs = [DocumentStream(name="my_doc.pdf", stream=buf)]
    # input = DocumentConversionInput.from_streams(docs)

    # # Turn on inline debug visualizations:
    # settings.debug.visualize_layout = True
    # settings.debug.visualize_ocr = True
    # settings.debug.visualize_tables = True
    # settings.debug.visualize_cells = True

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )

    start_time = time.time()

    conv_results = doc_converter.convert_all(
        input_doc_paths,
        raises_on_error=False,  # to let conversion run through all and examine results at the end
    )
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=output_dir
    )

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The example failed converting {failure_count} on {len(input_doc_paths)}."
        )



if __name__ == "__main__":
    main()
