from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from docling.datamodel.base_models import ConversionStatus

from .api import ExportOptions, build_converter, export_documents
from .failures import record_failure


_log = logging.getLogger(__name__)


def chunk_list(paths: Sequence[Path], chunk_size: int) -> Iterable[List[Path]]:
    for i in range(0, len(paths), chunk_size):
        yield list(paths[i : i + chunk_size])


def _process_single(
    file_path: Path,
    output_dir: Path,
    use_gpu: bool,
    gpu_id: Optional[int],
    failure_record_path: Optional[Path],
    export_options: ExportOptions,
) -> Tuple[str, str]:
    try:
        converter = build_converter(use_gpu=use_gpu, gpu_id=gpu_id)
        results = converter.convert_all([file_path], raises_on_error=False)

        # Count and export
        for conv_res in results:
            doc_filename = conv_res.input.file.stem
            if conv_res.status == ConversionStatus.SUCCESS:
                export_documents(results, output_dir, export_options)
                return ("success", doc_filename)
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                export_documents(results, output_dir, export_options)
                record_failure(
                    failure_record_path,
                    file_path.name,
                    "Partial conversion",
                    "partial_conversion",
                )
                return ("partial", doc_filename)
            else:
                record_failure(
                    failure_record_path,
                    file_path.name,
                    "Conversion failed",
                    "conversion_failed",
                )
                return ("failed", doc_filename)
        # No results; treat as failure
        record_failure(
            failure_record_path, file_path.name, "No result", "conversion_failed"
        )
        return ("failed", file_path.stem)
    except Exception as e:
        record_failure(
            failure_record_path, file_path.name, str(e), type(e).__name__
        )
        return ("exception", file_path.stem)


def process_parallel(
    inputs: List[Path],
    output_dir: Path,
    num_workers: Optional[int] = None,
    chunk_size: int = 1,
    use_gpu: bool = False,
    gpu_id: Optional[int] = None,
    failure_record_path: Optional[Path] = None,
    export_options: Optional[ExportOptions] = None,
) -> Tuple[int, int, int]:
    if use_gpu:
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    pool_workers = num_workers or cpu_count()
    export_opts = export_options or ExportOptions(write_markdown=True)

    _log.info(f"Using {pool_workers} workers; GPU: {'ON' if use_gpu else 'OFF'}")

    chunks = list(chunk_list(inputs, chunk_size))
    start = time.time()
    results: List[Tuple[str, str]] = []

    with Pool(processes=pool_workers) as pool:
        for chunk in chunks:
            process_func = partial(
                _process_single,
                output_dir=output_dir,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
                failure_record_path=failure_record_path,
                export_options=export_opts,
            )
            results.extend(pool.map(process_func, chunk))

    elapsed = time.time() - start
    success = sum(1 for status, _ in results if status == "success")
    partial = sum(1 for status, _ in results if status == "partial")
    failed = sum(1 for status, _ in results if status in {"failed", "exception"})

    _log.info(
        f"Completed {len(results)} files in {elapsed:.1f}s (success={success}, partial={partial}, failed={failed})"
    )
    return success, partial, failed


