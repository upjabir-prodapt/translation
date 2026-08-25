#!/usr/bin/env python3
"""Sample a worker process's memory while an E2E translation runs.

Cloud Run bills and OOM-kills on the container's RSS, so RSS (not VSZ) is the
number that matters. The worker also spawns threads and can shell out, so the
sampler sums the process tree rather than just the root PID.

Usage:
    python scripts/sample_worker_memory.py <pid> --out mem.csv [--interval 1.0]

Writes CSV: elapsed_s,rss_mb,threads,num_procs
and prints a peak/mean summary on exit (Ctrl-C or when the process dies).
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path


def _read_rss_kb(pid: int) -> int | None:
    """RSS in kB from /proc/<pid>/statm (page count x page size)."""
    try:
        with open(f"/proc/{pid}/statm") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, IndexError, ValueError):
        return None


def _read_threads(pid: int) -> int:
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return 0


def _child_pids(pid: int) -> list[int]:
    """Direct + transitive children, via /proc/<pid>/task/*/children."""
    found: list[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        task_dir = Path(f"/proc/{current}/task")
        if not task_dir.is_dir():
            continue
        for task in task_dir.iterdir():
            children_file = task / "children"
            try:
                kids = [int(x) for x in children_file.read_text().split()]
            except (OSError, ValueError):
                continue
            for kid in kids:
                if kid not in found:
                    found.append(kid)
                    stack.append(kid)
    return found


def sample(pid: int, out_path: Path, interval: float) -> int:
    start = time.monotonic()
    peak_rss = 0.0
    peak_threads = 0
    total = 0.0
    count = 0
    stopping = {"now": False}

    def _stop(_signum, _frame):
        stopping["now"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_s", "rss_mb", "threads", "num_procs"])
        while not stopping["now"]:
            root_rss = _read_rss_kb(pid)
            if root_rss is None:
                break  # process gone
            rss_kb = root_rss
            procs = 1
            for kid in _child_pids(pid):
                kid_rss = _read_rss_kb(kid)
                if kid_rss is not None:
                    rss_kb += kid_rss
                    procs += 1

            rss_mb = rss_kb / 1024
            threads = _read_threads(pid)
            elapsed = time.monotonic() - start

            writer.writerow([f"{elapsed:.1f}", f"{rss_mb:.1f}", threads, procs])
            handle.flush()

            peak_rss = max(peak_rss, rss_mb)
            peak_threads = max(peak_threads, threads)
            total += rss_mb
            count += 1
            time.sleep(interval)

    if count:
        print(
            f"samples={count}  peak_rss={peak_rss:.0f} MB "
            f"({peak_rss / 1024:.2f} GiB)  mean_rss={total / count:.0f} MB  "
            f"peak_threads={peak_threads}",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample worker RSS during an E2E run.")
    parser.add_argument("pid", type=int, help="worker process id")
    parser.add_argument("--out", type=Path, required=True, help="CSV output path")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds")
    args = parser.parse_args()

    if not Path(f"/proc/{args.pid}").is_dir():
        print(f"error: pid {args.pid} not running", file=sys.stderr)
        return 1
    return sample(args.pid, args.out, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
