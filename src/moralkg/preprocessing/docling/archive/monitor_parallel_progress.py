#!/usr/bin/env python3
"""
Enhanced Progress Monitor for PDF Processing

This script monitors the progress of the parallel PDF processing script
with improved error tracking and failure analysis.

Usage:
    python improved_monitor.py /path/to/progress/dir

Features:
- Real-time progress updates
- Per-worker status breakdown
- ETA calculation
- Rate estimation
- Detailed statistics
- Failure analysis with error type breakdown
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor progress of parallel PDF processing (Enhanced)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('progress_dir', type=str, default='/opt/extra/avijit/projects/moralkg/data/scripts/docling/docling-progress',
                       help="Directory containing worker progress files")
    parser.add_argument('--interval', type=int, default=10,
                       help="Update interval in seconds")
    parser.add_argument('--detailed', action='store_true',
                       help="Show detailed per-worker information")
    parser.add_argument('--once', action='store_true',
                       help="Show progress once and exit (don't loop)")
    parser.add_argument('--show-failures', action='store_true',
                       help="Show failure analysis from failed_files.json")
    parser.add_argument('--failure-record', type=str, 
                       default='/opt/extra/avijit/projects/moralkg/data/scripts/docling/failed_files.json',
                       help="Path to failed files record")
    
    return parser.parse_args()

def load_failure_record(failure_record_path):
    """Load and analyze the failure record"""
    if not failure_record_path or not Path(failure_record_path).exists():
        return None
    
    try:
        with open(failure_record_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read failure record: {e}")
        return None

def analyze_failures(failed_files_dict):
    """Analyze failure patterns"""
    if not failed_files_dict:
        return None
    
    error_types = Counter()
    total_attempts = 0
    recent_failures = []
    
    for filename, failure_info in failed_files_dict.items():
        error_type = failure_info.get('error_type', 'unknown')
        error_types[error_type] += 1
        total_attempts += failure_info.get('attempts', 1)
        
        # Consider failures in the last 24 hours as recent
        try:
            last_failure = datetime.fromisoformat(failure_info.get('last_failure', ''))
            if (datetime.now() - last_failure).total_seconds() < 86400:
                recent_failures.append({
                    'filename': filename,
                    'error_type': error_type,
                    'error': failure_info.get('last_error', 'Unknown error'),
                    'attempts': failure_info.get('attempts', 1),
                    'last_failure': last_failure
                })
        except (ValueError, TypeError):
            pass
    
    return {
        'total_failed_files': len(failed_files_dict),
        'total_attempts': total_attempts,
        'error_types': dict(error_types),
        'recent_failures': sorted(recent_failures, key=lambda x: x['last_failure'], reverse=True)
    }

def get_aggregated_progress(progress_dir):
    """Aggregate progress from overall progress file and active worker files"""
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
                'start_times': [],
                'last_updates': [],
                'has_overall_total': True,
                'using_overall_total': True,
                'using_overall_tracking': True,
                'overall_start_time': overall_data.get('start_time'),
                'overall_last_update': overall_data.get('last_update'),
                'overall_status': overall_data.get('status', 'unknown')
            }
            
            # Parse overall timestamps
            try:
                if overall_data.get('start_time'):
                    total_stats['earliest_start'] = datetime.fromisoformat(overall_data['start_time'])
                if overall_data.get('last_update'):
                    total_stats['most_recent_update'] = datetime.fromisoformat(overall_data['last_update'])
            except (ValueError, TypeError):
                pass
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read overall progress file: {e}")
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
            
            # Parse timestamps
            start_time = None
            last_update = None
            try:
                start_time = datetime.fromisoformat(worker_data.get('start_time', ''))
                last_update = datetime.fromisoformat(worker_data.get('last_update', ''))
                total_stats['start_times'].append(start_time)
                total_stats['last_updates'].append(last_update)
            except (ValueError, TypeError):
                pass
            
            worker_info = {
                'worker_id': worker_data.get('worker_id'),
                'worker_batch_size': worker_data.get('worker_total_files', worker_data.get('total_files', 0)),
                'total_files': worker_data.get('total_files', 0),
                'processed': worker_data.get('files_processed', 0),
                'succeeded': worker_data.get('files_succeeded', 0),
                'failed': worker_data.get('files_failed', 0),
                'partial': worker_data.get('files_partial', 0),
                'current_file': worker_data.get('current_file'),
                'status': worker_data.get('status', 'unknown'),
                'start_time': start_time,
                'last_update': last_update,
                'has_batch_info': 'worker_total_files' in worker_data
            }
            
            total_stats['workers'].append(worker_info)
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read progress file {progress_file}: {e}")
    
    # Update most recent update time if we have worker updates
    if total_stats['last_updates']:
        worker_most_recent = max(total_stats['last_updates'])
        if 'most_recent_update' not in total_stats or worker_most_recent > total_stats['most_recent_update']:
            total_stats['most_recent_update'] = worker_most_recent
    
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
        'start_times': [],
        'last_updates': [],
        'has_overall_total': False,
        'using_overall_total': False,
        'using_overall_tracking': False
    }
    
    overall_total_from_worker = None
    
    for progress_file in progress_dir.glob("progress_worker_*.json"):
        try:
            with open(progress_file, 'r') as f:
                worker_data = json.load(f)
            
            # Check if this is the improved format with overall_total_files
            if 'overall_total_files' in worker_data and overall_total_from_worker is None:
                overall_total_from_worker = worker_data['overall_total_files']
                total_stats['has_overall_total'] = True
            
            total_stats['files_processed'] += worker_data.get('files_processed', 0)
            total_stats['files_succeeded'] += worker_data.get('files_succeeded', 0)
            total_stats['files_failed'] += worker_data.get('files_failed', 0)
            total_stats['files_partial'] += worker_data.get('files_partial', 0)
            
            if worker_data.get('status') in ['processing', 'initializing']:
                total_stats['active_workers'] += 1
            
            # Parse timestamps
            start_time = None
            last_update = None
            try:
                start_time = datetime.fromisoformat(worker_data.get('start_time', ''))
                last_update = datetime.fromisoformat(worker_data.get('last_update', ''))
                total_stats['start_times'].append(start_time)
                total_stats['last_updates'].append(last_update)
            except (ValueError, TypeError):
                pass
            
            worker_info = {
                'worker_id': worker_data.get('worker_id'),
                'worker_batch_size': worker_data.get('worker_total_files', worker_data.get('total_files', 0)),
                'total_files': worker_data.get('total_files', 0),
                'processed': worker_data.get('files_processed', 0),
                'succeeded': worker_data.get('files_succeeded', 0),
                'failed': worker_data.get('files_failed', 0),
                'partial': worker_data.get('files_partial', 0),
                'current_file': worker_data.get('current_file'),
                'status': worker_data.get('status', 'unknown'),
                'start_time': start_time,
                'last_update': last_update,
                'has_batch_info': 'worker_total_files' in worker_data
            }
            
            total_stats['workers'].append(worker_info)
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read progress file {progress_file}: {e}")
    
    # Determine total files using best available method
    if overall_total_from_worker is not None:
        total_stats['total_files'] = overall_total_from_worker
        total_stats['using_overall_total'] = True
    else:
        # Fallback: sum worker totals (old behavior)
        total_stats['total_files'] = sum(w['total_files'] for w in total_stats['workers'])
        total_stats['using_overall_total'] = False
    
    # Calculate timing information
    if total_stats['start_times']:
        total_stats['earliest_start'] = min(total_stats['start_times'])
        total_stats['latest_start'] = max(total_stats['start_times'])
    
    if total_stats['last_updates']:
        total_stats['most_recent_update'] = max(total_stats['last_updates'])
    
    return total_stats

def format_duration(seconds):
    """Format duration in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"

def calculate_eta(processed, total, elapsed_seconds):
    """Calculate estimated time to completion"""
    if processed == 0 or total == 0:
        return None
    
    rate = processed / elapsed_seconds  # files per second
    remaining = total - processed
    eta_seconds = remaining / rate if rate > 0 else float('inf')
    
    return eta_seconds

def display_progress(progress, detailed=False, failure_analysis=None):
    """Display formatted progress information with improvements"""
    if not progress or progress['total_files'] == 0:
        print("No progress data available")
        return
    
    now = datetime.now()
    
    # Basic statistics
    total = progress['total_files']
    processed = progress['files_processed']
    succeeded = progress['files_succeeded']
    failed = progress['files_failed']
    partial = progress['files_partial']
    active = progress['active_workers']
    
    percent = (processed / total) * 100 if total > 0 else 0
    
    print("=" * 80)
    print(f"PDF Processing Progress - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show data source info
    if progress.get('using_overall_tracking'):
        print("Using persistent overall progress tracking")
        if progress.get('completed_workers', 0) > 0:
            print(f"   {progress['completed_workers']} workers completed, {progress['active_workers']} active")
        if progress.get('overall_status'):
            print(f"   Overall status: {progress['overall_status']}")
    elif progress['using_overall_total']:
        print("Using improved progress tracking (overall totals available)")
    else:
        print("Using legacy progress tracking (summing worker batches)")
    
    print("=" * 80)
    
    # Overall progress
    print(f"Overall Progress: {processed:,}/{total:,} files ({percent:.1f}%)")
    
    # Progress bar
    bar_width = 50
    filled = int(bar_width * percent / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"[{bar}] {percent:.1f}%")
    
    # Results breakdown
    print(f"\nResults:")
    print(f"  Success: Succeeded: {succeeded:,} ({succeeded/total*100:.1f}% of total)")
    print(f"  Warning: Partial:   {partial:,} ({partial/total*100:.1f}% of total)")
    print(f"  Error: Failed:    {failed:,} ({failed/total*100:.1f}% of total)")
    
    # Timing information
    if progress.get('earliest_start'):
        elapsed = (now - progress['earliest_start']).total_seconds()
        print(f"\nTiming:")
        print(f"  Elapsed: {format_duration(elapsed)}")
        
        if processed > 0:
            rate = processed / elapsed * 3600  # files per hour
            print(f"  Rate: {rate:.1f} files/hour")
            
            # ETA calculation
            eta_seconds = calculate_eta(processed, total, elapsed)
            if eta_seconds and eta_seconds != float('inf'):
                eta_time = now + timedelta(seconds=eta_seconds)
                print(f"  ETA: {format_duration(eta_seconds)} (around {eta_time.strftime('%H:%M on %m/%d')})")
    
    # Worker status
    print(f"\nWorkers: {active} active, {len(progress['workers'])} total")
    
    if detailed:
        print("\nPer-Worker Details:")
        for worker in sorted(progress['workers'], key=lambda w: w['worker_id']):
            status_icon = {
                'processing': 'PROC',
                'completed': 'DONE',
                'initializing': 'INIT',
                'starting': 'START'
            }.get(worker['status'], 'UNK')
            
            current_file = worker['current_file']
            if current_file and len(current_file) > 40:
                current_file = current_file[:37] + "..."
            
            # Show batch vs overall total info
            if worker['has_batch_info']:
                batch_size = worker['worker_batch_size']
                worker_percent = (worker['processed'] / batch_size * 100) if batch_size > 0 else 0
                batch_info = f"batch:{batch_size}"
            else:
                # Legacy format
                batch_size = worker['total_files']
                worker_percent = (worker['processed'] / batch_size * 100) if batch_size > 0 else 0
                batch_info = f"legacy:{batch_size}"
            
            print(f"  Worker {worker['worker_id']:>6}: {status_icon} "
                  f"{worker['processed']:>3}/{batch_size} ({worker_percent:>5.1f}%) [{batch_info}] "
                  f"S:{worker['succeeded']:>3} F:{worker['failed']:>2} P:{worker['partial']:>2}")
            
            if current_file and worker['status'] == 'processing':
                print(f"                     Currently: {current_file}")
    else:
        # Summary of worker batch sizes
        if progress['workers']:
            batch_sizes = [w['worker_batch_size'] if w['has_batch_info'] else w['total_files'] 
                          for w in progress['workers']]
            total_batch_sum = sum(batch_sizes)
            
            if progress.get('using_overall_tracking'):
                print(f"  Active workers: {len(progress['workers'])}, Completed: {progress.get('completed_workers', 0)}")
                if batch_sizes:
                    print(f"  Active worker batch sizes: {batch_sizes}")
            else:
                print(f"  Worker batch sizes: {batch_sizes}")
                print(f"  Sum of batches: {total_batch_sum} (should equal total: {total})")
                
                if total_batch_sum != total and not progress['using_overall_total']:
                    print(f"  Warning: Mismatch detected! Consider upgrading to improved progress tracking.")
    
    # Failure analysis
    if failure_analysis:
        print(f"\n" + "=" * 80)
        print("FAILURE ANALYSIS")
        print("=" * 80)
        
        print(f"Total failed files: {failure_analysis['total_failed_files']}")
        print(f"Total retry attempts: {failure_analysis['total_attempts']}")
        
        if failure_analysis['error_types']:
            print(f"\nError type breakdown:")
            for error_type, count in sorted(failure_analysis['error_types'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} files")
        
        if failure_analysis['recent_failures']:
            print(f"\nRecent failures (last 24h, top 3): {len(failure_analysis['recent_failures'])}")
            for failure in failure_analysis['recent_failures'][:3]:  # Show top 3
                elapsed = now - failure['last_failure']
                print(f"  {failure['filename']} ({format_duration(elapsed.total_seconds())} ago)")
                print(f"    Type: {failure['error_type']}, Attempts: {failure['attempts']}")
                error_preview = failure['error'][:60] + "..." if len(failure['error']) > 60 else failure['error']
                print(f"    Error: {error_preview}")
    
    # Staleness check
    if progress.get('most_recent_update'):
        staleness = (now - progress['most_recent_update']).total_seconds()
        if staleness > 300:  # 5 minutes
            print(f"\nWarning: Most recent update was {format_duration(staleness)} ago")
        elif staleness > 120:  # 2 minutes
            print(f"\nNote: Most recent update was {format_duration(staleness)} ago")

def main():
    args = parse_args()
    
    progress_dir = Path(args.progress_dir)
    if not progress_dir.exists():
        print(f"Error: Progress directory {progress_dir} does not exist")
        return 1
    
    # Load failure analysis if requested
    failure_analysis = None
    if args.show_failures:
        failed_files = load_failure_record(args.failure_record)
        if failed_files:
            failure_analysis = analyze_failures(failed_files)
        else:
            print(f"Warning: Could not load failure record from {args.failure_record}")
    
    if args.once:
        # Show progress once and exit
        progress = get_aggregated_progress(progress_dir)
        display_progress(progress, detailed=args.detailed, failure_analysis=failure_analysis)
        return 0
    
    print(f"Monitoring progress in {progress_dir}")
    print(f"Update interval: {args.interval} seconds")
    if args.show_failures:
        print(f"Failure analysis: ENABLED")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            progress = get_aggregated_progress(progress_dir)
            
            # Refresh failure analysis periodically
            if args.show_failures:
                failed_files = load_failure_record(args.failure_record)
                if failed_files:
                    failure_analysis = analyze_failures(failed_files)
            
            display_progress(progress, detailed=args.detailed, failure_analysis=failure_analysis)
            
            # Check if processing is complete
            if progress and progress['active_workers'] == 0 and progress['files_processed'] > 0:
                print("\nProcessing appears to be complete!")
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())