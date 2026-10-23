"""Reproducible RMQ construction, query, memory, and point-update benchmarks."""

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import tracemalloc
from array import array
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from math import isqrt, log2, sqrt
from pathlib import Path
from random import Random
from statistics import median, quantiles
from time import perf_counter_ns
from types import ModuleType

from _base import RangeMinimumQuery
from brute_force import BruteForce
from segment_tree import SegmentTree
from sparse_table import SparseTable
from sqrt_decomposition import SqrtDecomposition

ALGORITHMS = {
    "brute": BruteForce,
    "sqrt": SqrtDecomposition,
    "segment": SegmentTree,
    "sparse": SparseTable,
}
PATTERNS = ("random", "ascending", "descending", "constant")
WORKLOADS = ("random", "short", "wide", "singleton")

LABELS = {
    "brute": "Naive brute force",
    "sqrt": "Square root decomposition",
    "segment": "Segment tree",
    "sparse": "Sparse table",
}
COLORS = {
    "brute": "#252525",
    "sqrt": "#ce6112",
    "segment": "#1768ac",
    "sparse": "#7946a5",
}
MARKERS = {"brute": "D", "sqrt": "o", "segment": "s", "sparse": "^"}
LINESTYLES = {"brute": "--", "sqrt": "-", "segment": "-.", "sparse": ":"}
DEFAULT_SIZES = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]


@dataclass(frozen=True)
class Measurement:
    """One timing trial, buffer payload and separately profiled construction memory.

    memory_bytes/peak_memory_bytes trace an untimed constructor; process fields
    measure whole-process RSS in a fresh interpreter. Both profiles run once per
    algorithm/size and are reused across timing trials. None denotes an older
    CSV without that profile; it must never be interpreted as zero.
    """

    algorithm: str
    n: int
    queries: int
    trial: int
    seed: int
    pattern: str
    workload: str
    build_ns: int
    query_total_ns: int
    query_mean_ns: float
    storage_bytes: int
    checksum: int
    python: str
    platform: str
    memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    process_memory_bytes: int | None = None
    peak_process_memory_bytes: int | None = None
    baseline_process_memory_bytes: int | None = None


def generate_values(length: int, seed: int, pattern: str = "random") -> array:
    """Return length packed int64 values with a reproducible seed and pattern."""
    if length <= 0:
        raise ValueError("length must be positive")
    if pattern == "random":
        rng = Random(seed)
        return array("q", (rng.randint(-(10**9), 10**9) for _ in range(length)))
    if pattern == "ascending":
        return array("q", range(length))
    if pattern == "descending":
        return array("q", range(length, 0, -1))
    if pattern == "constant":
        return array("q", [-7]) * length
    raise ValueError(f"unknown pattern: {pattern}")


def generate_queries(
    length: int, count: int, seed: int, workload: str = "random"
) -> list[tuple[int, int]]:
    """Return count valid half-open ranges; vary interval lengths independently."""
    if length <= 0 or count <= 0:
        raise ValueError("length and query count must be positive")
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload}")
    rng = Random(seed)
    queries = []
    for _ in range(count):
        if workload == "random":
            first, second = rng.randrange(length), rng.randrange(length)
            start, stop = min(first, second), max(first, second) + 1
        elif workload == "wide":
            width = rng.randint(max(1, length // 2), length)
            start = rng.randrange(length - width + 1)
            stop = start + width
        else:
            start = rng.randrange(length)
            width = 1 if workload == "singleton" else rng.randint(1, min(64, length))
            stop = min(length, start + width)
        queries.append((start, stop))
    return queries


def estimate_storage_bytes(algorithm: str, length: int) -> int:
    """Return exact logical buffer bytes from n, without constructing a structure."""
    if length <= 0:
        raise ValueError("length must be positive")
    if algorithm == "brute":
        entries = length
    elif algorithm == "sqrt":
        size = isqrt(length - 1) + 1
        entries = length + (length + size - 1) // size
    elif algorithm == "segment":
        entries = 2 * (1 << (length - 1).bit_length())
    elif algorithm == "sparse":
        levels = length.bit_length()
        entries = (length + 1) * levels - (1 << levels) + 1
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return entries * array("q").itemsize


def _query_batch(
    structure: RangeMinimumQuery, queries: Sequence[tuple[int, int]]
) -> int:
    """Run pre-generated queries and return a checksum without storing all answers."""
    query = structure.query
    checksum = 0
    for start, stop in queries:
        checksum += query(start, stop)
    return checksum


def _measure_memory(
    constructor: type[RangeMinimumQuery], values: array
) -> tuple[int, int]:
    """Return retained/peak traced bytes from a separate construction pass.

    Start tracing after input generation, so shared data, queries and the
    interpreter baseline are excluded. Retained memory includes the structure's
    Python objects, array headers and allocated capacity; peak also includes
    construction temporaries. These are traced allocations, not process RSS.
    Do not alter an existing caller-owned tracing session.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError("memory profiling requires tracemalloc to be stopped")
    gc.collect()
    tracemalloc.start()
    try:
        structure = constructor(values)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del structure
    return current, peak


def measure_process_memory(
    algorithm: str,
    length: int,
    seed: int,
    pattern: str,
    query_count: int,
    workload: str,
) -> tuple[int, int, int]:
    """Return retained, peak and pre-build RSS bytes from a fresh interpreter.

    Unlike traced allocations, RSS includes Python, shared inputs and allocator
    overhead. The peak covers process startup through construction, including
    data generation. seed is the actual dataset seed, without adding length.
    This separate process never imports the plotting/reporting dependencies.
    """
    if (
        algorithm not in ALGORITHMS
        or pattern not in PATTERNS
        or workload not in WORKLOADS
    ):
        raise ValueError("unknown process memory profile configuration")
    if length <= 0 or query_count <= 0:
        raise ValueError("length and query count must be positive")
    environment = os.environ.copy()
    source_path = str(Path(__file__).resolve().parent)
    previous_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + previous_path if previous_path else source_path
    )
    # An inherited startup setting must not instrument the fresh RSS process.
    environment.pop("PYTHONTRACEMALLOC", None)
    command = [
        sys.executable,
        "-m",
        "memory_probe",
        "--algorithm",
        algorithm,
        "--length",
        str(length),
        "--seed",
        str(seed),
        "--pattern",
        pattern,
        "--queries",
        str(query_count),
        "--workload",
        workload,
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, env=environment
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "no diagnostic output").strip()
        raise RuntimeError(f"process memory worker failed: {detail}") from error
    except OSError as error:
        raise RuntimeError(f"cannot start process memory worker: {error}") from error
    try:
        payload = json.loads(result.stdout)
        names = (
            "process_memory_bytes",
            "peak_process_memory_bytes",
            "baseline_process_memory_bytes",
        )
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        values = tuple(payload[name] for name in names)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("RSS measurements must be nonnegative integers")
        retained, peak, baseline = values
        if peak < max(retained, baseline):
            raise ValueError("peak RSS is below an observed current RSS")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid process memory worker output: {error}") from error
    return retained, peak, baseline


def run_benchmarks(
    sizes: Sequence[int],
    algorithms: Sequence[str],
    query_count: int = 1000,
    repeats: int = 3,
    seed: int = 520,
    pattern: str = "random",
    workload: str = "random",
) -> Iterator[Measurement]:
    """Yield raw static trials using identical data/queries for every algorithm.

    Build timing includes the constructor's input copy. Query timing excludes
    construction, validation samples, warm-up and all data/query generation.
    Only one structure is retained at a time. storage_bytes counts logical
    array payload only. memory_bytes/peak_memory_bytes come from an extra untimed
    construction once per algorithm/size, including object overhead/capacity and
    peak construction temporaries, but excluding shared input/queries. Tracing
    is off throughout timing and query execution. A fresh subprocess separately
    measures total process RSS before/after construction and its lifetime peak,
    including Python, shared inputs and allocator overhead.
    """
    if query_count <= 0 or repeats <= 0:
        raise ValueError("query_count and repeats must be positive")
    if not sizes or not algorithms:
        raise ValueError("at least one size and algorithm are required")
    if any(name not in ALGORITHMS for name in algorithms):
        raise ValueError("unknown algorithm")
    if tracemalloc.is_tracing():
        raise RuntimeError("benchmark timings require tracemalloc to be stopped")
    python_version, system = platform.python_version(), platform.platform()
    for length in sizes:
        data_seed = seed + length
        values = generate_values(length, data_seed, pattern)
        queries = generate_queries(length, query_count, data_seed ^ 0x520, workload)
        # memoryview slices do not copy large ranges for the direct-min oracle.
        view = memoryview(values)
        samples = [(start, stop, min(view[start:stop])) for start, stop in queries[:8]]
        del view
        expected_checksum = None
        for algorithm in algorithms:
            constructor = ALGORITHMS[algorithm]
            memory_bytes, peak_memory_bytes = _measure_memory(constructor, values)
            process_memory, peak_process_memory, baseline_process_memory = (
                measure_process_memory(
                    algorithm, length, data_seed, pattern, query_count, workload
                )
            )
            for trial in range(1, repeats + 1):
                gc.collect()
                begin = perf_counter_ns()
                structure = constructor(values)
                build_ns = perf_counter_ns() - begin
                for start, stop, expected in samples:
                    if structure.query(start, stop) != expected:
                        raise RuntimeError(f"incorrect sample result from {algorithm}")
                _query_batch(structure, queries[:64])  # Untimed warm-up.
                begin = perf_counter_ns()
                checksum = _query_batch(structure, queries)
                query_ns = perf_counter_ns() - begin
                if expected_checksum is None:
                    expected_checksum = checksum
                elif checksum != expected_checksum:
                    raise RuntimeError(f"query checksum mismatch from {algorithm}")
                measurement = Measurement(
                    algorithm=algorithm,
                    n=length,
                    queries=query_count,
                    trial=trial,
                    seed=data_seed,
                    pattern=pattern,
                    workload=workload,
                    build_ns=build_ns,
                    query_total_ns=query_ns,
                    query_mean_ns=query_ns / query_count,
                    storage_bytes=structure.storage_bytes,
                    checksum=checksum,
                    python=python_version,
                    platform=system,
                    memory_bytes=memory_bytes,
                    peak_memory_bytes=peak_memory_bytes,
                    process_memory_bytes=process_memory,
                    peak_process_memory_bytes=peak_process_memory,
                    baseline_process_memory_bytes=baseline_process_memory,
                )
                del structure
                yield measurement


def _positive_int(value: str) -> int:
    """Parse a CLI integer and reject zero/negative values with a useful error."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """Return median, Q1 and Q3 with inclusive interpolation; support one trial."""
    if len(values) == 1:
        return values[0], values[0], values[0]
    first, _, third = quantiles(values, n=4, method="inclusive")
    return median(values), first, third


def _load_pyplot(output_dir: Path) -> ModuleType:
    """Load Matplotlib with a non-GUI backend and a writable local cache."""
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


UPDATE_ALGORITHMS = {
    "brute": BruteForce,
    "sqrt": SqrtDecomposition,
    "segment": SegmentTree,
}


@dataclass(frozen=True)
class UpdateMeasurement:
    """One batch timing; mean is batch duration divided by assignment count."""

    algorithm: str
    n: int
    updates: int
    trial: int
    seed: int
    update_seed: int
    update_total_ns: int
    update_mean_ns: float
    checksum: int
    python: str
    platform: str


def generate_updates(length: int, count: int, seed: int) -> list[tuple[int, int]]:
    """Generate uniform positions with replacement and random int64 assignments."""
    if length <= 0 or count <= 0:
        raise ValueError("length and update count must be positive")
    rng = Random(seed)
    return [(rng.randrange(length), rng.randint(-(10**9), 10**9)) for _ in range(count)]


def _apply_updates(structure, updates: Sequence[tuple[int, int]]) -> None:
    update = structure.update
    for position, value in updates:
        update(position, value)


def run_update_benchmarks(
    sizes: Sequence[int],
    update_count: int = 10000,
    repeats: int = 9,
    seed: int = 520,
) -> Iterator[UpdateMeasurement]:
    """Time identical assignments on fresh structures, rotating algorithm order.

    Generation, construction, separate-instance warm-up, and final-state checks
    are untimed. GC is disabled only during each batch and its state restored.
    Every updated point and nine final range minima are independently checked.
    """
    if not sizes or any(length <= 0 for length in sizes):
        raise ValueError("sizes must be nonempty and positive")
    if len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be unique")
    if update_count <= 0 or repeats <= 0:
        raise ValueError("update_count and repeats must be positive")
    python_version, system = platform.python_version(), platform.platform()
    algorithms = list(UPDATE_ALGORITHMS)
    for length in sizes:
        data_seed = seed + length
        update_seed = data_seed ^ 0xA520
        values = generate_values(length, data_seed)
        updates = generate_updates(length, update_count, update_seed)
        expected = array("q", values)
        for position, value in updates:
            expected[position] = value
        points = [
            (position, expected[position])
            for position in sorted({position for position, _ in updates})
        ]
        ranges = [(0, length)] + generate_queries(length, 8, update_seed ^ 0x520)
        expected_ranges = [
            (start, stop, min(expected[start:stop])) for start, stop in ranges
        ]
        del expected
        for constructor in UPDATE_ALGORITHMS.values():
            warmup = constructor(values)
            _apply_updates(warmup, updates[: min(256, update_count)])
            del warmup
        for trial in range(repeats):
            offset = trial % len(algorithms)
            for algorithm in algorithms[offset:] + algorithms[:offset]:
                gc.collect()
                structure = UPDATE_ALGORITHMS[algorithm](values)
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    started = perf_counter_ns()
                    _apply_updates(structure, updates)
                    elapsed = perf_counter_ns() - started
                finally:
                    if gc_enabled:
                        gc.enable()
                checksum = 0
                for position, expected_value in points:
                    actual = structure.query(position, position + 1)
                    if actual != expected_value:
                        raise AssertionError(f"incorrect final point: {algorithm}")
                    checksum += actual
                for start, stop, expected_value in expected_ranges:
                    actual = structure.query(start, stop)
                    if actual != expected_value:
                        raise AssertionError(f"incorrect final range: {algorithm}")
                    checksum += actual
                del structure
                yield UpdateMeasurement(
                    algorithm=algorithm,
                    n=length,
                    updates=update_count,
                    trial=trial + 1,
                    seed=data_seed,
                    update_seed=update_seed,
                    update_total_ns=elapsed,
                    update_mean_ns=elapsed / update_count,
                    checksum=checksum,
                    python=python_version,
                    platform=system,
                )


def summarize_updates(measurements: Sequence[UpdateMeasurement]) -> list[dict]:
    """Summarize batch averages in microseconds with inclusive quartiles."""
    if not measurements:
        raise ValueError("at least one measurement is required")
    groups = defaultdict(list)
    for row in measurements:
        groups[(row.n, row.algorithm)].append(row)
    summaries = []
    for (length, algorithm), trials in sorted(groups.items()):
        configurations = {
            (r.updates, r.seed, r.update_seed, r.python, r.platform, r.checksum)
            for r in trials
        }
        if len(configurations) != 1:
            raise ValueError(f"incompatible trials for {algorithm} at n={length}")
        middle, first, third = _quartiles([r.update_mean_ns / 1000 for r in trials])
        summaries.append(
            dict(
                algorithm=algorithm,
                n=length,
                updates=trials[0].updates,
                repeats=len(trials),
                seed=trials[0].seed,
                update_seed=trials[0].update_seed,
                update_median_us=middle,
                update_q1_us=first,
                update_q3_us=third,
            )
        )
    return summaries


def _write_update_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_update_series(axis, rows: Sequence[dict], algorithm: str, x_values) -> None:
    """Keep the same timing values, quartiles, and method styles in every view."""
    axis.plot(
        x_values,
        [row["update_median_us"] for row in rows],
        label=LABELS[algorithm],
        color=COLORS[algorithm],
        marker=MARKERS[algorithm],
        linestyle=LINESTYLES[algorithm],
    )
    axis.fill_between(
        x_values,
        [row["update_q1_us"] for row in rows],
        [row["update_q3_us"] for row in rows],
        color=COLORS[algorithm],
        alpha=0.15,
    )


def plot_updates(summaries: Sequence[dict], output_dir: Path) -> None:
    """Save three axis-scale comparisons and per-method growth-axis panels."""
    groups = {
        algorithm: sorted(
            (row for row in summaries if row["algorithm"] == algorithm),
            key=lambda row: row["n"],
        )
        for algorithm in UPDATE_ALGORITHMS
        if any(row["algorithm"] == algorithm for row in summaries)
    }
    if not groups:
        raise ValueError("at least one mutable-method summary is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "update_time_semilogy.png").unlink(missing_ok=True)
    plt = _load_pyplot(output_dir)
    views = (
        ("update_time.png", "log", "log"),
        ("update_time_linear.png", "linear", "linear"),
        ("update_time_semilogx.png", "log", "linear"),
    )
    for filename, xscale, yscale in views:
        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
        for algorithm, rows in groups.items():
            _plot_update_series(axis, rows, algorithm, [row["n"] for row in rows])
        axis.set(
            title="Update time",
            xlabel="Number of input values (n)",
            ylabel="Time per update (µs)",
            xscale=xscale,
            yscale=yscale,
        )
        if xscale == "linear":
            axis.set_xlim(left=0)
        if yscale == "linear":
            axis.set_ylim(bottom=0)
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(loc="upper left")
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    transforms = {
        "brute": (float, "Number of input values (n)"),
        "sqrt": (sqrt, "Square root of input size (√n)"),
        "segment": (log2, "Logarithm of input size (log₂(n))"),
    }
    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(4.8 * len(groups), 4.5),
        squeeze=False,
        layout="constrained",
    )
    figure.suptitle("Update time")
    for axis, (algorithm, rows) in zip(axes[0], groups.items(), strict=True):
        transform, xlabel = transforms[algorithm]
        _plot_update_series(
            axis, rows, algorithm, [transform(row["n"]) for row in rows]
        )
        axis.set(
            title=LABELS[algorithm],
            xlabel=xlabel,
            ylabel="Time per update (µs)",
            xlim=(0, None),
            ylim=(0, None),
        )
        axis.grid(True, alpha=0.2)
    figure.savefig(output_dir / "update_time_growth_axes.png", dpi=180)
    plt.close(figure)


def load_update_measurements(path: Path) -> list[UpdateMeasurement]:
    """Load saved raw trials for plot regeneration without rerunning updates."""
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            UpdateMeasurement(
                **{
                    name: kind(row[name])
                    for name, kind in UpdateMeasurement.__annotations__.items()
                }
            )
            for row in csv.DictReader(stream)
        ]


def update_main(argv: Sequence[str] | None = None) -> None:
    """Run point-update measurements or regenerate their saved plots."""
    parser = argparse.ArgumentParser(
        description="Benchmark point assignments on the three mutable RMQ structures."
    )
    parser.add_argument("--sizes", type=_positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--updates", type=_positive_int, default=10000)
    parser.add_argument("--repeats", type=_positive_int, default=9)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="regenerate summaries and plots from saved raw trials without timing",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        measurements = load_update_measurements(args.from_csv)
    else:
        measurements = []
        for row in run_update_benchmarks(
            args.sizes, args.updates, args.repeats, args.seed
        ):
            measurements.append(row)
            print(
                f"n={row.n:>8,} {row.algorithm:>7} trial={row.trial} "
                f"{row.update_mean_ns / 1000:.3f} µs/update",
                flush=True,
            )
    summaries = summarize_updates(measurements)
    if not args.from_csv:
        _write_update_csv(
            args.output_dir / "update_measurements.csv",
            [asdict(r) for r in measurements],
        )
    _write_update_csv(args.output_dir / "update_summary.csv", summaries)
    plot_updates(summaries, args.output_dir)
    print(f"Saved update summary and plots to {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run static measurements or dispatch the updates command."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["updates"]:
        update_main(arguments[1:])
        return
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="For point updates, run: python -m benchmark_core updates --help.",
    )
    parser.add_argument(
        "--sizes", type=_positive_int, nargs="+", default=[1000, 10000, 100000]
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    parser.add_argument("--queries", type=_positive_int, default=1000)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--pattern", choices=PATTERNS, default="random")
    parser.add_argument("--workload", choices=WORKLOADS, default="random")
    parser.add_argument("--output", type=Path, default=Path("results/measurements.csv"))
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="print buffer sizes without allocating data",
    )
    args = parser.parse_args(arguments)
    if args.estimate_only:
        writer = csv.writer(sys.stdout)
        writer.writerow(["algorithm", "n", "storage_bytes"])
        for length in args.sizes:
            for name in args.algorithms:
                writer.writerow([name, length, estimate_storage_bytes(name, length)])
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements = run_benchmarks(
        args.sizes,
        args.algorithms,
        args.queries,
        args.repeats,
        args.seed,
        args.pattern,
        args.workload,
    )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Measurement.__annotations__))
        writer.writeheader()
        for result in measurements:
            writer.writerow(asdict(result))
            stream.flush()
            print(
                f"{result.algorithm:8} n={result.n:,} trial={result.trial} "
                f"build={result.build_ns / 1e6:.3f} ms "
                f"query={result.query_mean_ns / 1e3:.3f} us/query "
                f"memory={result.memory_bytes / 2**20:.3f} MiB "
                f"peak-build={result.peak_memory_bytes / 2**20:.3f} MiB "
                f"process={result.process_memory_bytes / 2**20:.3f} MiB "
                f"peak-process={result.peak_process_memory_bytes / 2**20:.3f} MiB",
                file=sys.stderr,
            )
    print(f"Raw measurements: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
