from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.base import ImageRefMode


@dataclass
class ExportOptions:
    write_markdown: bool = True
    write_text: bool = False
    write_json: bool = False
    write_html: bool = False
    write_yaml: bool = False
    write_doctags: bool = False


def build_converter(use_gpu: bool = False, gpu_id: Optional[int] = None) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    if use_gpu:
        pipeline_options.accelerator_options = (
            AcceleratorOptions(device=f"cuda:{gpu_id}")
            if gpu_id is not None
            else AcceleratorOptions(device=AcceleratorDevice.AUTO)
        )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
    options: Optional[ExportOptions] = None,
) -> Tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = options or ExportOptions()

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        doc_filename = conv_res.input.file.stem

        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            _export_one(conv_res, output_dir, doc_filename, opts)
        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            partial_success_count += 1
            _export_one(conv_res, output_dir, doc_filename, opts)
        else:
            failure_count += 1

    return success_count, partial_success_count, failure_count


def _export_one(
    conv_res: ConversionResult, output_dir: Path, base_name: str, opts: ExportOptions
) -> None:
    # V2 exports
    if opts.write_json:
        conv_res.document.save_as_json(
            output_dir / f"{base_name}.json", image_mode=ImageRefMode.PLACEHOLDER
        )
    if opts.write_html:
        conv_res.document.save_as_html(
            output_dir / f"{base_name}.html", image_mode=ImageRefMode.EMBEDDED
        )
    if opts.write_doctags:
        conv_res.document.save_as_doctags(output_dir / f"{base_name}.doctags.txt")
    if opts.write_markdown:
        conv_res.document.save_as_markdown(
            output_dir / f"{base_name}.md", image_mode=ImageRefMode.PLACEHOLDER
        )
        # Ensure plain markdown text is also written from export API if requested
        with (output_dir / f"{base_name}.md").open("w") as fp:
            fp.write(conv_res.document.export_to_markdown())
    if opts.write_text:
        with (output_dir / f"{base_name}.txt").open("w") as fp:
            fp.write(conv_res.document.export_to_markdown(strict_text=True))
    if opts.write_yaml:
        import yaml  # Local import to avoid hard dependency if unused

        with (output_dir / f"{base_name}.yaml").open("w") as fp:
            fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))


__all__ = [
    "ExportOptions",
    "build_converter",
    "export_documents",
]


