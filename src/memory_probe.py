"""Profile whole-process resident memory in an isolated, untraced interpreter."""

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import tracemalloc
from pathlib import Path

from benchmark_core import (
    ALGORITHMS,
    PATTERNS,
    WORKLOADS,
    _positive_int,
    generate_queries,
    generate_values,
)


def _current_rss_bytes() -> int:
    """Read current resident memory, independently of the high-water mark."""
    if sys.platform.startswith("linux"):
        # /proc/PID/statm's second column is resident pages (Linux proc(5)).
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "darwin":
        # Apple's ps(1) defines rss in 1024-byte units, regardless of page size.
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) * 1024
    raise RuntimeError("process RSS profiling supports macOS and Linux")


def _peak_rss_bytes() -> int:
    """Return this process's lifetime resident high-water mark in bytes."""
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # Modern XNU assigns resident_size_max (bytes) directly to ru_maxrss:
        # github.com/apple-oss-distributions/xnu, bsd/kern/kern_resource.c and
        # osfmk/mach/task_info.h. The archived Apple manpage has obsolete units.
        return int(maximum)
    if sys.platform.startswith("linux"):
        # Linux getrusage(2) documents ru_maxrss in KiB.
        return int(maximum) * 1024
    raise RuntimeError("process RSS profiling supports macOS and Linux")


def profile_process_memory(
    algorithm: str,
    length: int,
    seed: int,
    pattern: str,
    query_count: int,
    workload: str,
) -> dict[str, int]:
    """Build one structure while retaining the same shared data and queries.

    The baseline is current RSS after data generation and garbage collection.
    Retained RSS is read after the constructor returns. Peak RSS includes all
    activity since worker startup through the construction pass; OS high-water
    marks cannot be reset to exclude input generation. No queries are executed.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError("process memory worker requires tracemalloc to be stopped")
    values = generate_values(length, seed, pattern)
    queries = generate_queries(length, query_count, seed ^ 0x520, workload)
    gc.collect()
    baseline = _current_rss_bytes()
    structure = ALGORITHMS[algorithm](values)
    retained = _current_rss_bytes()
    peak = _peak_rss_bytes()
    # Explicitly keep all three alive through both readings.
    del structure, values, queries
    return {
        "process_memory_bytes": retained,
        "peak_process_memory_bytes": peak,
        "baseline_process_memory_bytes": baseline,
    }


def main() -> None:
    """Write one JSON memory profile for the benchmark parent process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--length", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pattern", choices=PATTERNS, required=True)
    parser.add_argument("--queries", type=_positive_int, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    args = parser.parse_args()
    profile = profile_process_memory(
        args.algorithm,
        args.length,
        args.seed,
        args.pattern,
        args.queries,
        args.workload,
    )
    print(json.dumps(profile))


if __name__ == "__main__":
    main()
