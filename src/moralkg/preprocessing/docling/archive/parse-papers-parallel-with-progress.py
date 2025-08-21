#!/usr/bin/env python3
"""
PDF processor with real-time failure tracking that updates the failure record
immediately as failures occur, rather than waiting until the end.
"""

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
import traceback

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

# Thread-safe locks
failed_files_lock = threading.Lock()
overall_progress_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse PDF papers using Docling (Real-time Failure Tracking)",
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

    parser.add_argument('--failure-record-path', type=str, default="/opt/extra/avijit/projects/moralkg/data/scripts/docling/failed_files.json",
                       help="Path to the JSON file for recording failed files")
    
    # Progress tracking options
    parser.add_argument('--progress-dir', type=str, default="/opt/extra/avijit/projects/moralkg/data/scripts/docling/docling-progress",
                       help="Directory to store per-worker progress files (enables progress tracking)")
    parser.add_argument('--progress-interval', type=int, default=30,
                       help="Seconds between progress updates (default: 30)")
    
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
    else:
        _log.debug(f"No failed files record found at {failure_record_path}")
    return {}

def save_failed_files(failure_record_path, failed_files_dict):
    """Save the failed files record to disk with atomic write"""
    if failure_record_path is None:
        return
    
    failed_files_path = Path(failure_record_path)
    try:
        # Ensure parent directory exists
        failed_files_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = failed_files_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(failed_files_dict, f, indent=2, sort_keys=True)
        
        # Atomic rename
        temp_path.rename(failed_files_path)
        _log.debug(f"Saved {len(failed_files_dict)} failed files to {failure_record_path}")
        
    except IOError as e:
        _log.error(f"Could not save failed files record: {e}")

def ensure_failure_record_exists(failure_record_path):
    """Ensure failure record file exists, creating empty one if needed"""
    if failure_record_path is None:
        return
    
    failed_files_path = Path(failure_record_path)
    if not failed_files_path.exists():
        _log.info(f"Creating initial failure record at {failure_record_path}")
        try:
            # Ensure parent directory exists
            failed_files_path.parent.mkdir(parents=True, exist_ok=True)
            with open(failed_files_path, 'w') as f:
                json.dump({}, f, indent=2)
        except IOError as e:
            _log.error(f"Could not create failure record: {e}")

def record_single_failure(failure_record_path, filename, error_msg, error_type):
    """
    Immediately record a single failure to the failure record file.
    This is called by workers as soon as a failure occurs.
    """
    if failure_record_path is None:
        return
    
    failure_info = {
        'filename': filename,
        'timestamp': datetime.now().isoformat(),
        'error': error_msg,
        'error_type': error_type
    }
    
    with failed_files_lock:
        try:
            failed_files = load_failed_files(failure_record_path)
            
            if filename in failed_files:
                failed_files[filename]['attempts'] += 1
                failed_files[filename]['last_failure'] = failure_info['timestamp']
                failed_files[filename]['last_error'] = failure_info['error']
                failed_files[filename]['error_type'] = failure_info['error_type']
                _log.info(f"Updated failure record for {filename} (attempt #{failed_files[filename]['attempts']})")
            else:
                failed_files[filename] = {
                    'first_failure': failure_info['timestamp'],
                    'last_failure': failure_info['timestamp'],
                    'last_error': failure_info['error'],
                    'error_type': failure_info['error_type'],
                    'attempts': 1
                }
                _log.info(f"Added {filename} to failure record (first failure)")
            
            save_failed_files(failure_record_path, failed_files)
            
        except Exception as e:
            _log.error(f"Could not record failure for {filename}: {e}")

def should_skip_file(pdf_path, output_dir, skip_existing):
    """Check if file should be skipped based on existing outputs"""
    if not skip_existing:
        return False
    
    doc_filename = pdf_path.stem
    output_files = [output_dir / f"{doc_filename}.md"]
    return any(os.path.exists(f) for f in output_files)

def should_skip_failed_file(pdf_path, failed_files_dict):
    """Check if file should be skipped based on failed files record"""
    return pdf_path.name in failed_files_dict

def create_overall_progress_file(progress_dir, total_files):
    """Create the overall progress tracking file"""
    if progress_dir is None:
        return None
    
    progress_dir = Path(progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    
    overall_file = progress_dir / "overall_progress.json"
    overall_data = {
        'total_files': total_files,
        'files_processed': 0,
        'files_succeeded': 0,
        'files_failed': 0,
        'files_partial': 0,
        'completed_workers': 0,
        'start_time': datetime.now().isoformat(),
        'last_update': datetime.now().isoformat(),
        'status': 'initializing'
    }
    
    try:
        with open(overall_file, 'w') as f:
            json.dump(overall_data, f, indent=2)
        return str(overall_file)
    except IOError as e:
        _log.error(f"Could not create overall progress file: {e}")
        return None

def update_overall_progress_file(progress_dir, **updates):
    """Thread-safe update of overall progress file"""
    if progress_dir is None:
        return
    
    with overall_progress_lock:
        overall_file = Path(progress_dir) / "overall_progress.json"
        if not overall_file.exists():
            return
        
        try:
            # Read current data
            with open(overall_file, 'r') as f:
                overall_data = json.load(f)
            
            # Update fields
            for key, value in updates.items():
                if key.startswith('add_'):
                    # Handle additive updates
                    field = key[4:]  # Remove 'add_' prefix
                    overall_data[field] = overall_data.get(field, 0) + value
                else:
                    overall_data[key] = value
            
            overall_data['last_update'] = datetime.now().isoformat()
            
            # Write back
            with open(overall_file, 'w') as f:
                json.dump(overall_data, f, indent=2)
                
        except (IOError, json.JSONDecodeError) as e:
            _log.error(f"Could not update overall progress file: {e}")

def create_progress_file(progress_dir, worker_id, worker_total_files, overall_total_files):
    """Create initial progress file for a worker with both batch and overall totals"""
    if progress_dir is None:
        return None
    
    progress_dir = Path(progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    
    progress_file = progress_dir / f"progress_worker_{worker_id}.json"
    progress_data = {
        'worker_id': worker_id,
        'worker_total_files': worker_total_files,  # This worker's batch size
        'overall_total_files': overall_total_files,  # Total across all workers
        'total_files': overall_total_files,  # Keep this for backward compatibility
        'files_processed': 0,
        'files_succeeded': 0,
        'files_failed': 0,
        'files_partial': 0,
        'current_file': None,
        'start_time': datetime.now().isoformat(),
        'last_update': datetime.now().isoformat(),
        'status': 'starting'
    }
    
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        return str(progress_file)
    except IOError as e:
        _log.error(f"Could not create progress file: {e}")
        return None

def update_progress_file(progress_file_path, **updates):
    """Update progress file with new information"""
    if progress_file_path is None:
        return
    
    try:
        # Read current data
        with open(progress_file_path, 'r') as f:
            progress_data = json.load(f)
        
        # Update fields
        for key, value in updates.items():
            progress_data[key] = value
        
        progress_data['last_update'] = datetime.now().isoformat()
        
        # Write back
        with open(progress_file_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    except (IOError, json.JSONDecodeError) as e:
        _log.error(f"Could not update progress file: {e}")

def get_aggregated_progress(progress_dir):
    """Aggregate progress from overall progress file and active worker files"""
    if progress_dir is None:
        return None
    
    progress_dir = Path(progress_dir)
    if not progress_dir.exists():
        return None
    
    # First, try to read the overall progress file
    overall_file = progress_dir / "overall_progress.json"
    if overall_file.exists():
        try:
            with open(overall_file, 'r') as f:
                overall_data = json.load(f)
            
            # Use overall data as base
            total_stats = {
                'total_files': overall_data.get('total_files', 0),
                'files_processed': overall_data.get('files_processed', 0),
                'files_succeeded': overall_data.get('files_succeeded', 0),
                'files_failed': overall_data.get('files_failed', 0),
                'files_partial': overall_data.get('files_partial', 0),
                'completed_workers': overall_data.get('completed_workers', 0),
                'active_workers': 0,
                'workers': [],
                'start_time': overall_data.get('start_time'),
                'using_overall_tracking': True
            }
            
        except (IOError, json.JSONDecodeError) as e:
            _log.warning(f"Could not read overall progress file: {e}")
            # Fall back to old method
            return get_legacy_aggregated_progress(progress_dir)
    else:
        # Fall back to old method
        return get_legacy_aggregated_progress(progress_dir)
    
    # Add data from active workers
    for progress_file in progress_dir.glob("progress_worker_*.json"):
        try:
            with open(progress_file, 'r') as f:
                worker_data = json.load(f)
            
            if worker_data.get('status') in ['processing', 'initializing']:
                total_stats['active_workers'] += 1
            
            worker_info = {
                'worker_id': worker_data.get('worker_id'),
                'worker_batch_size': worker_data.get('worker_total_files', worker_data.get('total_files', 0)),
                'processed': worker_data.get('files_processed', 0),
                'succeeded': worker_data.get('files_succeeded', 0),
                'failed': worker_data.get('files_failed', 0),
                'partial': worker_data.get('files_partial', 0),
                'current_file': worker_data.get('current_file'),
                'status': worker_data.get('status', 'unknown'),
                'start_time': worker_data.get('start_time'),
                'last_update': worker_data.get('last_update')
            }
            
            total_stats['workers'].append(worker_info)
            
        except (IOError, json.JSONDecodeError) as e:
            _log.warning(f"Could not read progress file {progress_file}: {e}")
    
    return total_stats

def get_legacy_aggregated_progress(progress_dir):
    """Fallback method for aggregating progress (original implementation)"""
    total_stats = {
        'total_files': 0,
        'files_processed': 0,
        'files_succeeded': 0,
        'files_failed': 0,
        'files_partial': 0,
        'active_workers': 0,
        'workers': [],
        'using_overall_tracking': False
    }
    
    overall_total_from_worker = None
    
    for progress_file in progress_dir.glob("progress_worker_*.json"):
        try:
            with open(progress_file, 'r') as f:
                worker_data = json.load(f)
            
            if overall_total_from_worker is None:
                overall_total_from_worker = worker_data.get('overall_total_files')
            
            if overall_total_from_worker:
                total_stats['total_files'] = overall_total_from_worker
            else:
                total_stats['total_files'] += worker_data.get('total_files', 0)
            
            total_stats['files_processed'] += worker_data.get('files_processed', 0)
            total_stats['files_succeeded'] += worker_data.get('files_succeeded', 0)
            total_stats['files_failed'] += worker_data.get('files_failed', 0)
            total_stats['files_partial'] += worker_data.get('files_partial', 0)
            
            if worker_data.get('status') in ['processing', 'initializing']:
                total_stats['active_workers'] += 1
            
            worker_info = {
                'worker_id': worker_data.get('worker_id'),
                'worker_batch_size': worker_data.get('worker_total_files', worker_data.get('total_files', 0)),
                'processed': worker_data.get('files_processed', 0),
                'succeeded': worker_data.get('files_succeeded', 0),
                'failed': worker_data.get('files_failed', 0),
                'partial': worker_data.get('files_partial', 0),
                'current_file': worker_data.get('current_file'),
                'status': worker_data.get('status', 'unknown'),
                'start_time': worker_data.get('start_time'),
                'last_update': worker_data.get('last_update')
            }
            
            total_stats['workers'].append(worker_info)
            
        except (IOError, json.JSONDecodeError) as e:
            _log.warning(f"Could not read progress file {progress_file}: {e}")
    
    return total_stats

def cleanup_worker_progress_file(progress_dir, worker_id):
    """Clean up a specific worker's progress file after completion"""
    if progress_dir is None:
        return
    
    progress_dir = Path(progress_dir)
    progress_file = progress_dir / f"progress_worker_{worker_id}.json"
    
    try:
        if progress_file.exists():
            progress_file.unlink()
            _log.debug(f"Cleaned up progress file for worker {worker_id}")
    except OSError as e:
        _log.warning(f"Could not remove progress file {progress_file}: {e}")

def cleanup_all_progress_files(progress_dir):
    """Clean up all progress files after completion"""
    if progress_dir is None:
        return
    
    progress_dir = Path(progress_dir)
    if not progress_dir.exists():
        return
    
    # Clean up worker progress files
    for progress_file in progress_dir.glob("progress_worker_*.json"):
        try:
            progress_file.unlink()
        except OSError as e:
            _log.warning(f"Could not remove progress file {progress_file}: {e}")
    
    # Clean up overall progress file
    overall_file = progress_dir / "overall_progress.json"
    try:
        if overall_file.exists():
            overall_file.unlink()
    except OSError as e:
        _log.warning(f"Could not remove overall progress file: {e}")

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

def process_single_file(file_path, doc_converter, output_dir, failure_record_path):
    """
    Process a single PDF file with comprehensive error handling and immediate failure recording.
    Returns a tuple of (success_type, error_info) where:
    - success_type: 'success', 'partial', 'failed', or 'exception'
    - error_info: dict with error details if failed/exception, None if success
    """
    try:
        # This is the critical section where the PdfiumError occurs
        conv_results = doc_converter.convert_all([file_path], raises_on_error=False)
        
        for conv_res in conv_results:
            doc_filename = conv_res.input.file.stem
            
            if conv_res.status == ConversionStatus.SUCCESS:
                _log.info(f"Successfully converted {file_path.name}, exporting formats...")
                
                # Export documents
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
                
                return ('success', None)
                
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                _log.warning(f"Document {conv_res.input.file} was partially converted")
                error_messages = []
                for item in conv_res.errors:
                    error_msg = str(item.error_message)
                    _log.warning(f"\t{error_msg}")
                    error_messages.append(error_msg)
                
                # Still export partial results
                if USE_V2:
                    conv_res.document.save_as_markdown(
                        output_dir / f"{doc_filename}.md",
                        image_mode=ImageRefMode.PLACEHOLDER,
                    )
                    with (output_dir / f"{doc_filename}.md").open("w") as fp:
                        fp.write(conv_res.document.export_to_markdown())
                
                error_summary = '; '.join(error_messages[:3])  # Limit error message length
                
                # IMMEDIATELY record this partial failure
                record_single_failure(failure_record_path, file_path.name, 
                                    f"Partial conversion: {error_summary}", 'partial_conversion')
                
                return ('partial', {
                    'error': f"Partial conversion: {error_summary}",
                    'error_type': 'partial_conversion'
                })
                
            else:
                # Conversion failed
                error_messages = []
                if hasattr(conv_res, 'errors') and conv_res.errors:
                    for item in conv_res.errors:
                        error_messages.append(str(item.error_message))
                
                error_summary = '; '.join(error_messages[:2]) if error_messages else "Unknown conversion error"
                
                # IMMEDIATELY record this failure
                record_single_failure(failure_record_path, file_path.name, 
                                    error_summary, 'conversion_failed')
                
                return ('failed', {
                    'error': error_summary,
                    'error_type': 'conversion_failed'
                })
                
    except Exception as e:
        # This catches the PdfiumError and any other exceptions
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Get more detailed error info
        if hasattr(e, '__module__'):
            error_type = f"{e.__module__}.{error_type}"
        
        _log.error(f"Exception during processing of {file_path.name}: {error_type}: {error_msg}")
        
        # Log the full traceback for debugging
        _log.debug(f"Full traceback for {file_path.name}:\n{traceback.format_exc()}")
        
        # IMMEDIATELY record this exception
        record_single_failure(failure_record_path, file_path.name, 
                            f"{error_type}: {error_msg}", error_type)
        
        return ('exception', {
            'error': f"{error_type}: {error_msg}",
            'error_type': error_type
        })
    
    # Should not reach here, but just in case
    record_single_failure(failure_record_path, file_path.name, 
                        "Unknown processing error", 'unknown')
    return ('failed', {
        'error': "Unknown processing error",
        'error_type': 'unknown'
    })

def process_file_batch(file_paths, output_dir=None, use_gpu=False, gpu_id=None, progress_dir=None, progress_interval=30, overall_total_files=None, failure_record_path=None):
    """Process a batch of files in a single worker process with immediate failure recording"""
    worker_id = os.getpid()
    batch_start_time = time.time()
    
    # Initialize progress tracking with both batch and overall totals
    progress_file_path = create_progress_file(progress_dir, worker_id, len(file_paths), overall_total_files)
    last_progress_update = time.time()
    
    _log.info(f"Worker {worker_id}: Starting batch of {len(file_paths)} files (GPU: {'ON' if use_gpu else 'OFF'}, GPU ID: {gpu_id})")
    if overall_total_files:
        _log.info(f"Worker {worker_id}: Overall total across all workers: {overall_total_files} files")
    
    # Update progress status
    update_progress_file(progress_file_path, status='initializing')
    
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
    
    # Update progress status
    update_progress_file(progress_file_path, status='processing')
    
    results = {
        'success': 0,
        'partial_success': 0,
        'failure': 0,
        'processed_files': [],
        'failed_files': []  # Keep for backward compatibility, but failures are now recorded immediately
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(file_paths, 1):
        file_start_time = time.time()
        
        # Update current file in progress
        update_progress_file(progress_file_path, 
                           current_file=file_path.name,
                           files_processed=i-1)
        
        # Periodic progress update
        if time.time() - last_progress_update > progress_interval:
            update_progress_file(progress_file_path,
                               files_succeeded=results['success'],
                               files_failed=results['failure'],
                               files_partial=results['partial_success'])
            last_progress_update = time.time()
        
        _log.info(f"Worker {worker_id}: Processing file {i}/{len(file_paths)}: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Process the file with comprehensive error handling and immediate failure recording
        success_type, error_info = process_single_file(file_path, doc_converter, output_dir, failure_record_path)
        
        file_time = time.time() - file_start_time
        
        if success_type == 'success':
            _log.info(f"Worker {worker_id}: {file_path.name} completed successfully in {file_time:.1f}s")
            results['success'] += 1
            results['processed_files'].append(str(file_path))
            
        elif success_type == 'partial':
            _log.warning(f"Worker {worker_id}: {file_path.name} partially converted in {file_time:.1f}s: {error_info['error']}")
            results['partial_success'] += 1
            results['processed_files'].append(str(file_path))
            
            # Keep for backward compatibility (but failure was already recorded immediately)
            results['failed_files'].append({
                'filename': file_path.name,
                'timestamp': datetime.now().isoformat(),
                'error': error_info['error'],
                'error_type': error_info['error_type']
            })
            
        else:  # 'failed' or 'exception'
            _log.error(f"Worker {worker_id}: {file_path.name} failed to convert in {file_time:.1f}s: {error_info['error']}")
            results['failure'] += 1
            
            # Keep for backward compatibility (but failure was already recorded immediately)
            results['failed_files'].append({
                'filename': file_path.name,
                'timestamp': datetime.now().isoformat(),
                'error': error_info['error'],
                'error_type': error_info['error_type']
            })
    
    batch_time = time.time() - batch_start_time
    _log.info(f"Worker {worker_id}: Completed batch in {batch_time:.1f}s - Success: {results['success']}, Partial: {results['partial_success']}, Failed: {results['failure']}")
    
    # Final progress update
    update_progress_file(progress_file_path,
                        status='completed',
                        files_processed=len(file_paths),
                        files_succeeded=results['success'],
                        files_failed=results['failure'],
                        files_partial=results['partial_success'],
                        current_file=None)
    
    # Update overall progress with this worker's results
    update_overall_progress_file(progress_dir,
                               add_files_processed=len(file_paths),
                               add_files_succeeded=results['success'],
                               add_files_failed=results['failure'],
                               add_files_partial=results['partial_success'],
                               add_completed_workers=1)
    
    # Clean up this worker's progress file now that results are in overall file
    cleanup_worker_progress_file(progress_dir, worker_id)
    
    return results

def chunk_list(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Fix CUDA multiprocessing issue FIRST
    if args.use_gpu:
        _log.info("GPU enabled - setting multiprocessing start method to 'spawn' to fix CUDA issues")
        try:
            mp.set_start_method('spawn', force=True)
            _log.info("Successfully set multiprocessing start method to 'spawn'")
        except RuntimeError as e:
            _log.warning(f"Could not set start method to 'spawn': {e}")
            _log.warning("This may cause CUDA errors in worker processes")

    input_doc_paths = get_input_files(args)
    output_dir = Path(args.output_dir)
    
    # Ensure failure record exists at startup
    ensure_failure_record_exists(args.failure_record_path)
    
    # Determine number of workers
    num_workers = args.num_workers or mp.cpu_count()
    _log.info(f"Using {num_workers} worker processes")
    _log.info(f"GPU acceleration: {'ENABLED' if args.use_gpu else 'DISABLED'}")
    
    _log.info(f"Processing {len(input_doc_paths)} PDF files")
    _log.info(f"Input directory: {args.input_dir}")
    _log.info(f"Output directory: {output_dir}")
    
    if args.failure_record_path:
        _log.info(f"Failure record: {args.failure_record_path} (REAL-TIME UPDATES)")
        if args.skip_failed:
            failed_files = load_failed_files(args.failure_record_path)
            _log.info(f"Skip failed files: ENABLED ({len(failed_files)} files in failed record)")
        else:
            _log.info("Skip failed files: DISABLED")
    else:
        _log.info("Failure record: DISABLED (no --failure-record-path specified)")
        if args.skip_failed:
            _log.warning("--skip-failed specified but no failure record path provided")
    
    if args.progress_dir:
        _log.info(f"Progress tracking: ENABLED (directory: {args.progress_dir})")
        _log.info(f"Progress update interval: {args.progress_interval} seconds")
        # Clean up any existing progress files and create overall progress file
        cleanup_all_progress_files(args.progress_dir)
        create_overall_progress_file(args.progress_dir, len(input_doc_paths))
    else:
        _log.info("Progress tracking: DISABLED (no --progress-dir specified)")
    
    # Split files into chunks
    file_chunks = list(chunk_list(input_doc_paths, args.chunk_size))
    _log.info(f"Split into {len(file_chunks)} chunks of size {args.chunk_size}")
    
    # Store overall total for passing to workers
    overall_total_files = len(input_doc_paths)
    
    start_time = time.time()
    
    # Start progress monitoring if enabled
    progress_monitor_stop = threading.Event()
    progress_thread = None
    
    if args.progress_dir:
        def monitor_progress():
            while not progress_monitor_stop.wait(60):  # Check every minute
                progress = get_aggregated_progress(args.progress_dir)
                if progress and progress['total_files'] > 0:
                    percent = (progress['files_processed'] / progress['total_files']) * 100
                    elapsed = time.time() - start_time
                    if progress['files_processed'] > 0:
                        eta_seconds = (elapsed / progress['files_processed']) * (progress['total_files'] - progress['files_processed'])
                        eta_hours = eta_seconds / 3600
                        _log.info(f"PROGRESS: {progress['files_processed']}/{progress['total_files']} ({percent:.1f}%) - "
                                f"Success: {progress['files_succeeded']}, Failed: {progress['files_failed']}, "
                                f"Active workers: {progress['active_workers']}, ETA: {eta_hours:.1f}h")
        
        progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        progress_thread.start()
    
    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        # Assign GPU IDs to chunks if using GPU
        if args.use_gpu:
            if args.gpu_id is not None:
                _log.info(f"Using specified GPU ID: {args.gpu_id}")
                process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=True, gpu_id=args.gpu_id, 
                                     progress_dir=args.progress_dir, progress_interval=args.progress_interval,
                                     overall_total_files=overall_total_files, failure_record_path=args.failure_record_path)
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
                            chunk_gpu_pairs.append((chunk, output_dir, args.use_gpu, gpu_id, args.progress_dir, args.progress_interval, overall_total_files, args.failure_record_path))
                        _log.info(f"Assigned {len(chunk_gpu_pairs)} chunks to {gpu_count} GPUs")
                        
                        # Create partial function with fixed output_dir and GPU setting
                        process_func = partial(process_file_batch)
                        results = pool.starmap(process_func, chunk_gpu_pairs)

                    else:
                        _log.warning("No GPUs available, falling back to CPU")
                        process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=False,
                                             progress_dir=args.progress_dir, progress_interval=args.progress_interval,
                                             overall_total_files=overall_total_files, failure_record_path=args.failure_record_path)
                        results = pool.map(process_func, file_chunks)
                except ImportError:
                    _log.warning("PyTorch not available, falling back to CPU")
                    process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=False,
                                         progress_dir=args.progress_dir, progress_interval=args.progress_interval,
                                         overall_total_files=overall_total_files, failure_record_path=args.failure_record_path)
                    results = pool.map(process_func, file_chunks)
        else:
            process_func = partial(process_file_batch, output_dir=output_dir, use_gpu=False,
                                 progress_dir=args.progress_dir, progress_interval=args.progress_interval,
                                 overall_total_files=overall_total_files, failure_record_path=args.failure_record_path)
            results = pool.map(process_func, file_chunks)
    
    # Stop progress monitoring
    if progress_thread:
        progress_monitor_stop.set()
        progress_thread.join(timeout=5)
    
    # Aggregate results 
    total_success = sum(r['success'] for r in results)
    total_partial_success = sum(r['partial_success'] for r in results)
    total_failure = sum(r['failure'] for r in results)
    
    # Final progress report
    if args.progress_dir:
        final_progress = get_aggregated_progress(args.progress_dir)
        if final_progress:
            _log.info(f"FINAL PROGRESS: {final_progress['files_processed']}/{final_progress['total_files']} files processed")
            _log.info(f"Completed workers: {final_progress.get('completed_workers', 'N/A')}")
            if final_progress.get('workers'):
                _log.info(f"Remaining active workers: {len(final_progress['workers'])}")
        
        # Mark overall progress as completed
        update_overall_progress_file(args.progress_dir, status='completed')
    
    # Show final failure statistics from the real-time updated failure record
    if args.failure_record_path:
        final_failed_files = load_failed_files(args.failure_record_path)
        _log.info(f"FINAL FAILURE RECORD: {len(final_failed_files)} total failed files recorded")
        
        if final_failed_files:
            # Analyze error types
            error_types = {}
            for filename, failure_info in final_failed_files.items():
                error_type = failure_info.get('error_type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            _log.info(f"Error type breakdown:")
            for error_type, count in sorted(error_types.items()):
                _log.info(f"  {error_type}: {count} files")
    
    end_time = time.time() - start_time
    
    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")
    _log.info(f"Processed {total_success + total_partial_success + total_failure} docs, "
              f"of which {total_failure} failed and {total_partial_success} were partially converted.")
    _log.info(f"Successfully converted: {total_success}")
    
    if total_failure > 0:
        _log.warning(f"Failed to convert {total_failure} documents")
    
    if total_partial_success > 0:
        _log.warning(f"Partially converted {total_partial_success} documents")
    
    # Clean up all progress files at the end
    if args.progress_dir:
        cleanup_all_progress_files(args.progress_dir)

if __name__ == "__main__":
    main()