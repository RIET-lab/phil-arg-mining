# parse-papers-parallel.py

# Simple parallel version using multiprocessing
# Now also skipping failed files and using GPU acceleration if available.
# parse-papers-parallel.py



import argparse
import multiprocessing as mp
from pathlib import Path
import logging
import time
import os
from functools import partial
import json
from datetime import datetime
import threading

from docling.document_converter import DocumentConverter
from docling_core.types.doc.base import ImageRefMode
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import PdfFormatOption
import yaml
import json

_log = logging.getLogger(__name__)

USE_V2 = True
USE_LEGACY = False

# Thread-safe lock for updating failed files record
failed_files_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse PDF papers using Docling (Parallel)",
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
    parser.add_argument('--skip-failed', action='store_true',
                       help="Skip processing files that have previously failed (based on failed_files.json)")
    parser.add_argument('--num-workers', type=int, default=None,
                       help="Number of parallel workers (defaults to number of CPU cores)")
    parser.add_argument('--chunk-size', type=int, default=1,
                       help="Number of files to process per worker at a time")
    parser.add_argument('--use-gpu', action='store_true',
                       help="Enable GPU acceleration for faster processing")
    
    parser.add_argument('--gpu-id', type=int, default=None,
                       help="Specific GPU ID to use (if using GPU acceleration). Defaults to None (auto-selects GPUs).")
    
    parser.add_argument('--failure-record-path', type=str, default="/opt/extra/avijit/projects/moralkg/data/failed_files.json",
                       help="Path to the JSON file for recording failed files")
    
    return parser.parse_args()

def load_failed_files(failure_record_path):
    """Load the failed files record from disk"""
    if failure_record_path is None:
        return {}
    
    failed_files_path = Path(failure_record_path)
    if failed_files_path.exists():
        try:
            with open(failed_files_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            _log.warning(f"Could not load failed files record: {e}")
            return {}
    return {}

def save_failed_files(failure_record_path, failed_files_dict):
    """Save the failed files record to disk"""
    if failure_record_path is None:
        return
    
    failed_files_path = Path(failure_record_path)
    try:
        # Ensure parent directory exists
        failed_files_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_files_path, 'w') as f:
            json.dump(failed_files_dict, f, indent=2, sort_keys=True)
    except IOError as e:
        _log.error(f"Could not save failed files record: {e}")

def update_failed_files_record(failure_record_path, failures):
    """Thread-safe update of failed files record"""
    if not failures or failure_record_path is None:
        return
    
    with failed_files_lock:
        failed_files = load_failed_files(failure_record_path)
        
        for failure in failures:
            filename = failure['filename']
            if filename in failed_files:
                failed_files[filename]['attempts'] += 1
                failed_files[filename]['last_failure'] = failure['timestamp']
                failed_files[filename]['last_error'] = failure['error']
            else:
                failed_files[filename] = {
                    'first_failure': failure['timestamp'],
                    'last_failure': failure['timestamp'],
                    'last_error': failure['error'],
                    'attempts': 1
                }
        
        save_failed_files(failure_record_path, failed_files)
        _log.info(f"Updated failed files record with {len(failures)} new failures")

def should_skip_file(pdf_path, output_dir, skip_existing):
    """Check if file should be skipped based on existing outputs"""
    if not skip_existing:
        return False
    
    doc_filename = pdf_path.stem
    output_files = [output_dir / f"{doc_filename}.md"]
    return any(f.exists() for f in output_files)

def should_skip_failed_file(pdf_path, failed_files_dict):
    """Check if file should be skipped based on failed files record"""
    return pdf_path.name in failed_files_dict

def get_input_files(args):
    """Get list of PDF files to process based on arguments"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if args.files:
        input_doc_paths = []
        for filename in args.files:
            file_path = input_dir / filename
            if file_path.exists() and file_path.suffix.lower() == '.pdf':
                input_doc_paths.append(file_path)
            else:
                _log.warning(f"File not found or not a PDF: {filename}")
    else:
        input_doc_paths = list(input_dir.glob("*.pdf"))
    
    if not input_doc_paths:
        raise RuntimeError(f"No PDF files found to process")
    
    original_count = len(input_doc_paths)
    
    # Filter out existing files if requested
    if args.skip_existing:
        input_doc_paths = [p for p in input_doc_paths if not should_skip_file(p, output_dir, True)]
        skipped_existing = original_count - len(input_doc_paths)
        if skipped_existing > 0:
            _log.info(f"Skipped {skipped_existing} files with existing outputs")
    
    # Filter out previously failed files if requested
    if args.skip_failed and args.failure_record_path is not None:
        failed_files = load_failed_files(args.failure_record_path)
        if failed_files:
            before_failed_filter = len(input_doc_paths)
            input_doc_paths = [p for p in input_doc_paths if not should_skip_failed_file(p, failed_files)]
            skipped_failed = before_failed_filter - len(input_doc_paths)
            if skipped_failed > 0:
                _log.info(f"Skipped {skipped_failed} files that previously failed")
                _log.info(f"Total files in failed record: {len(failed_files)}")
        else:
            _log.info("No failed files record found, not skipping any files")
    elif args.skip_failed and args.failure_record_path is None:
        _log.warning("--skip-failed specified but no --failure-record-path provided, ignoring --skip-failed")
    
    return input_doc_paths

def process_file_batch(file_paths, output_dir=None, use_gpu=False, gpu_id=None):
    """Process a batch of files in a single worker process"""
    worker_id = os.getpid()
    batch_start_time = time.time()
    
    _log.info(f"Worker {worker_id}: Starting batch of {len(file_paths)} files (GPU: {'ON' if use_gpu else 'OFF'}, GPU ID: {gpu_id})")
    
    # Initialize converter in each worker process
    _log.info(f"Worker {worker_id}: Initializing DocumentConverter...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    
    # Enable GPU acceleration if requested
    if use_gpu:
        if gpu_id is not None:
            _log.info(f"Worker {worker_id}: Using GPU ID {gpu_id} for processing")
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=f"cuda:{gpu_id}",
            )
        else:
            _log.info(f"Worker {worker_id}: Using available GPUs in round-robin fashion")
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.AUTO
            )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, 
                backend=DoclingParseV4DocumentBackend
            )
        }
    )
    _log.info(f"Worker {worker_id}: DocumentConverter initialized successfully")
    
    results = {
        'success': 0,
        'partial_success': 0,
        'failure': 0,
        'processed_files': [],
        'failed_files': []  # New: detailed failure information
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(file_paths, 1):
        file_start_time = time.time()
        try:
            _log.info(f"Worker {worker_id}: Processing file {i}/{len(file_paths)}: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            conv_results = doc_converter.convert_all([file_path], raises_on_error=False)
            
            for conv_res in conv_results:
                doc_filename = conv_res.input.file.stem
                
                if conv_res.status == ConversionStatus.SUCCESS:
                    export_start_time = time.time()
                    _log.info(f"Worker {worker_id}: Successfully converted {file_path.name}, exporting formats...")
                    
                    # Export documents (same as original script, but with some formats commented out)
                    if USE_V2:
                        conv_res.document.save_as_markdown(
                            output_dir / f"{doc_filename}.md",
                            image_mode=ImageRefMode.PLACEHOLDER,
                        )

                        with (output_dir / f"{doc_filename}.md").open("w") as fp:
                            fp.write(conv_res.document.export_to_markdown())

                    if USE_LEGACY:
                        with (output_dir / f"{doc_filename}.legacy.json").open("w", encoding="utf-8") as fp:
                            fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))
                        with (output_dir / f"{doc_filename}.legacy.txt").open("w", encoding="utf-8") as fp:
                            fp.write(conv_res.legacy_document.export_to_markdown(strict_text=True))
                        with (output_dir / f"{doc_filename}.legacy.md").open("w", encoding="utf-8") as fp:
                            fp.write(conv_res.legacy_document.export_to_markdown())
                        with (output_dir / f"{doc_filename}.legacy.doctags.txt").open("w", encoding="utf-8") as fp:
                            fp.write(conv_res.legacy_document.export_to_document_tokens())
                    
                    export_time = time.time() - export_start_time
                    file_time = time.time() - file_start_time
                    _log.info(f"Worker {worker_id}: {file_path.name} completed successfully in {file_time:.1f}s (export: {export_time:.1f}s)")
                    results['success'] += 1
                    results['processed_files'].append(str(file_path))
                    
                elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                    _log.warning(f"Document {conv_res.input.file} was partially converted")
                    error_messages = []
                    for item in conv_res.errors:
                        error_msg = str(item.error_message)
                        _log.warning(f"Worker {worker_id}: \t{error_msg}")
                        error_messages.append(error_msg)
                    
                    # Still export partial results
                    if USE_V2:
                        conv_res.document.save_as_markdown(
                            output_dir / f"{doc_filename}.md",
                            image_mode=ImageRefMode.PLACEHOLDER,
                        )
                        with (output_dir / f"{doc_filename}.md").open("w") as fp:
                            fp.write(conv_res.document.export_to_markdown())
                    
                    results['partial_success'] += 1
                    results['processed_files'].append(str(file_path))
                    
                    # Record as partial failure for tracking
                    results['failed_files'].append({
                        'filename': file_path.name,
                        'timestamp': datetime.now().isoformat(),
                        'error': f"Partial conversion: {'; '.join(error_messages[:3])}"  # Limit error message length
                    })
                    
                else:
                    file_time = time.time() - file_start_time
                    error_messages = []
                    if hasattr(conv_res, 'errors') and conv_res.errors:
                        for item in conv_res.errors:
                            error_messages.append(str(item.error_message))
                    
                    error_summary = '; '.join(error_messages[:2]) if error_messages else "Unknown conversion error"
                    _log.error(f"Worker {worker_id}: {file_path.name} failed to convert in {file_time:.1f}s: {error_summary}")
                    results['failure'] += 1
                    
                    # Record detailed failure information
                    results['failed_files'].append({
                        'filename': file_path.name,
                        'timestamp': datetime.now().isoformat(),
                        'error': error_summary
                    })
                    
        except Exception as e:
            file_time = time.time() - file_start_time
            error_msg = f"Exception during processing: {str(e)}"
            _log.error(f"Worker {worker_id}: Error processing {file_path.name} in {file_time:.1f}s: {error_msg}")
            results['failure'] += 1
            
            # Record detailed failure information
            results['failed_files'].append({
                'filename': file_path.name,
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            })
    
    batch_time = time.time() - batch_start_time
    _log.info(f"Worker {worker_id}: Completed batch in {batch_time:.1f}s - Success: {results['success']}, Partial: {results['partial_success']}, Failed: {results['failure']}")
    
    return results

def chunk_list(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    input_doc_paths = get_input_files(args)
    output_dir = Path(args.output_dir)
    
    # Determine number of workers
    num_workers = args.num_workers or mp.cpu_count()
    _log.info(f"Using {num_workers} worker processes")
    _log.info(f"GPU acceleration: {'ENABLED' if args.use_gpu else 'DISABLED'}")
    
    _log.info(f"Processing {len(input_doc_paths)} PDF files")
    _log.info(f"Input directory: {args.input_dir}")
    _log.info(f"Output directory: {output_dir}")
    
    if args.failure_record_path:
        _log.info(f"Failure record: {args.failure_record_path}")
        if args.skip_failed:
            failed_files = load_failed_files(args.failure_record_path)
            _log.info(f"Skip failed files: ENABLED ({len(failed_files)} files in failed record)")
        else:
            _log.info("Skip failed files: DISABLED")
    else:
        _log.info("Failure record: DISABLED (no --failure-record-path specified)")
        if args.skip_failed:
            _log.warning("--skip-failed specified but no failure record path provided")
    
    # Split files into chunks
    file_chunks = list(chunk_list(input_doc_paths, args.chunk_size))
    _log.info(f"Split into {len(file_chunks)} chunks of size {args.chunk_size}")
    
    start_time = time.time()
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        # Assign GPU IDs to chunks if using GPU
        if args.use_gpu:
            if args.gpu_id is not None:
                _log.info(f"Using specified GPU ID: {args.gpu_id}")
                # Create partial function with fixed output_dir and GPU setting
                process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=True, gpu_id=args.gpu_id)
                results = pool.map(process_func, file_chunks)
            else:
                _log.info("Using available GPUs in round-robin fashion")
                try:
                    import torch
                    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    _log.info(f"Detected {gpu_count} available GPUs")
                    if gpu_count > 0:
                        # Assign chunks to GPUs in round-robin fashion
                        chunk_gpu_pairs = []
                        for i, chunk in enumerate(file_chunks):
                            gpu_id = i % gpu_count
                            chunk_gpu_pairs.append((chunk, output_dir, args.use_gpu, gpu_id))
                        _log.info(f"Assigned {len(chunk_gpu_pairs)} chunks to {gpu_count} GPUs")
                        
                        # Create partial function with fixed output_dir and GPU setting
                        process_func = partial(process_file_batch)
                        results = pool.starmap(process_func, chunk_gpu_pairs)

                    else:
                        _log.warning("No GPUs available, falling back to CPU")
                        # Create partial function with fixed output_dir and GPU setting
                        process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=False)
                        results = pool.map(process_func, file_chunks)
                except ImportError:
                    _log.warning("PyTorch not available, falling back to CPU")
                    process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=False)
                    results = pool.map(process_func, file_chunks)
        else:
            # Create partial function with fixed output_dir and no GPU
            process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=False)
            results = pool.map(process_func, file_chunks)
    
    # Aggregate results and update failed files record
    total_success = sum(r['success'] for r in results)
    total_partial_success = sum(r['partial_success'] for r in results)
    total_failure = sum(r['failure'] for r in results)
    
    # Collect all failed files from all workers
    all_failed_files = []
    for result in results:
        all_failed_files.extend(result['failed_files'])
    
    # Update the failed files record
    if all_failed_files:
        update_failed_files_record(args.failure_record_path, all_failed_files)
        if args.failure_record_path:
            _log.info(f"Recorded {len(all_failed_files)} failed/partial files in {args.failure_record_path}")
        else:
            _log.info(f"Would have recorded {len(all_failed_files)} failed/partial files (no failure record path specified)")
    
    end_time = time.time() - start_time
    
    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")
    _log.info(f"Processed {total_success + total_partial_success + total_failure} docs, "
              f"of which {total_failure} failed and {total_partial_success} were partially converted.")
    _log.info(f"Successfully converted: {total_success}")
    
    if total_failure > 0:
        _log.warning(f"Failed to convert {total_failure} documents")
    
    if total_partial_success > 0:
        _log.warning(f"Partially converted {total_partial_success} documents")

if __name__ == "__main__":
    main()