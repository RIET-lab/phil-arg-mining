#!/usr/bin/env python3
"""
Parse recent run logs and estimate remaining time per run.

Features:
- Picks the n most recent log files from the logs folder (by timestamp in filename when present,
  otherwise by filesystem mtime).
- Extracts: "Total combinations to process", produced outputs (from checkpoint lines),
  and per-generation latencies (from "Generated output: ... latency=XXXs").
- Estimates remaining time = mean(latencies) * remaining_outputs.

Notes:
- Code that would read the logs directory from `config.yaml` or import local packages
  (for example, src.moralkg.config) is intentionally included but commented out.
  For now the script uses the `./.logs` directory.
  - TODO: Check if loading config and local packages actually risks interfering with generation.
"""
import argparse
import os
import re
import statistics
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


LOG_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
TOTAL_COMBINATIONS_RE = re.compile(r"Total combinations to process:\s*(\d+)")
LATENCY_RE = re.compile(r"Generated output: [^\n]*latency=([0-9.]+)s")
CHECKPOINT_PRODUCED_RE = re.compile(r"Wrote intermittent checkpoint: .*\(produced=(\d+)\)")
WRITING_CHECKPOINT_RE = re.compile(r"Writing checkpoint '.*' with (\d+) outputs")
LINE_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_filename_timestamp(name: str) -> Optional[datetime]:
    m = LOG_TIMESTAMP_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def seconds_to_hms(seconds: float) -> str:
    td = timedelta(seconds=int(round(seconds)))
    return str(td)


def parse_log(path: str) -> dict:
    """Parse a single log file and return metrics."""
    text = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    total = None
    latencies: List[float] = []
    first_ts = None
    last_ts = None

    m = TOTAL_COMBINATIONS_RE.search(text)
    if m:
        total = int(m.group(1))

    latencies = [float(x) for x in LATENCY_RE.findall(text)]

    # produced outputs should be the number of latency entries (one latency per output)
    produced = len(latencies)

    # extract timestamp from first and last non-empty lines (message beginnings)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        # first line timestamp
        m1 = LINE_TIMESTAMP_RE.match(lines[0])
        if m1:
            try:
                first_ts = datetime.strptime(m1.group(1), "%Y-%m-%d %H:%M:%S")
            except Exception:
                first_ts = None
        # last line timestamp
        m2 = LINE_TIMESTAMP_RE.match(lines[-1])
        if m2:
            try:
                last_ts = datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S")
            except Exception:
                last_ts = None

    return {
        "path": path,
        "total_combinations": total,
        "produced": produced,
        "latencies": latencies,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def estimate_remaining(total: Optional[int], produced: int, latencies: List[float]) -> Tuple[int, Optional[float]]:
    """Return (remaining_outputs, estimated_seconds)"""
    if total is None:
        return 0, None
    remaining = max(0, total - produced)
    if remaining == 0:
        return remaining, 0.0
    if not latencies:
        return remaining, None
    # Use the mean latency per generation as the estimator
    mean_latency = statistics.mean(latencies)
    est = mean_latency * remaining
    return remaining, est


def find_most_recent_logs(logdir: str, n: int) -> List[str]:
    files = []
    if not os.path.isdir(logdir):
        raise FileNotFoundError(f"Logs directory not found: {logdir}")

    for name in os.listdir(logdir):
        path = os.path.join(logdir, name)
        if not os.path.isfile(path):
            continue
        ts = parse_filename_timestamp(name)
        if ts is None:
            # fallback to mtime
            ts = datetime.fromtimestamp(os.path.getmtime(path))
        files.append((ts, path))

    files.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in files[:n]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse run logs and estimate remaining time")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of recent logs to analyze (default: 1)")
    parser.add_argument("-k", "--last-latencies", type=int, default=20, help="Show last K latency entries per log (default: 20)")
    parser.add_argument("--logs-dir", type=str, default=None, help="Logs directory (overrides config). If not set, script uses ./.logs by default")
    args = parser.parse_args()

    # Example: read logs dir from config.yaml (disabled/commented out for now)
    # import yaml
    # from src.moralkg.config import load_config  # commented; local package imports disabled for now
    # cfg = load_config("config.yaml")
    # logs_dir = cfg.general.logs.dir

    # Use provided CLI value or fallback to ./.logs
    logs_dir = args.logs_dir or os.path.join(os.getcwd(), ".logs")

    try:
        recent = find_most_recent_logs(logs_dir, args.num)
    except FileNotFoundError as e:
        print(e)
        return

    if not recent:
        print(f"No logs found in {logs_dir}")
        return

    for path in recent:
        info = parse_log(path)
        total = info["total_combinations"]
        produced = info["produced"]
        latencies = info["latencies"]
        first_ts = info.get("first_ts")
        last_ts = info.get("last_ts")

        remaining, est_seconds = estimate_remaining(total, produced, latencies)

        print("\n---")
        print(f"Log: {os.path.basename(path)}")
        if total is None:
            print("Total combinations: NOT FOUND")
        else:
            print(f"Total combinations: {total}")
        print(f"Produced outputs: {produced}")
        print(f"Remaining outputs: {remaining}")

        # elapsed time from first to last message in the file
        if first_ts and last_ts:
            elapsed = (last_ts - first_ts).total_seconds()
            print(f"Elapsed time (log messages): {seconds_to_hms(elapsed)} ({int(elapsed)}s)")
            if produced > 0:
                rate = elapsed / produced
                print(f"  avg time per produced output so far: {rate:.2f}s")
        else:
            print("Elapsed time (log messages): unknown (timestamps missing)")

        if latencies:
            mean_lat = statistics.mean(latencies)
            median_lat = statistics.median(latencies)
            stdev_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            print(f"Latencies found: {len(latencies)} (seconds)")
            print(f"  mean={mean_lat:.2f}s, median={median_lat:.2f}s, stdev={stdev_lat:.2f}s")
            to_show = latencies[-args.last_latencies :]
            print(f"  last {len(to_show)} latencies: {', '.join(f'{x:.2f}s' for x in to_show)}")
        else:
            print("No generation latency entries found in log.")

        if est_seconds is None:
            print("Estimated remaining time: unknown (missing latency data or total combinations)")
        else:
            print(f"Estimated remaining time: {seconds_to_hms(est_seconds)} ({est_seconds:.0f}s)")


if __name__ == "__main__":
    main()
