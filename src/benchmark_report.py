"""Generate raw/summary benchmark CSV files and runtime/storage growth plots."""

import argparse
import csv
import platform
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from math import log2, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import ModuleType

from benchmark_core import (
    ALGORITHMS,
    COLORS,
    DEFAULT_SIZES,
    LABELS,
    LINESTYLES,
    MARKERS,
    PATTERNS,
    WORKLOADS,
    Measurement,
    _load_pyplot,
    _positive_int,
    _quartiles,
    measure_process_memory,
    run_benchmarks,
)

RETIRED_PLOT_NAMES = {
    "build_time.png",
    "memory_growth.png",
    "memory_per_element.png",
    "memory_per_element_linear.png",
    "peak_memory_growth.png",
    "query_speedup.png",
    "query_speedup_linear.png",
    "query_time_advanced.png",
    "query_time_advanced_linear.png",
    "query_time_linear.png",
    "query_time_semilogx.png",
    "query_time_sparse_linear.png",
    "space_growth.png",
    "space_growth_linear.png",
    "space_per_element.png",
    "space_per_element_linear.png",
}


@dataclass(frozen=True)
class Summary:
    """Timing statistics, buffer payload, allocation profiles, and process RSS."""

    algorithm: str
    n: int
    queries: int
    repeats: int
    seed: int
    pattern: str
    workload: str
    build_median_ms: float
    build_q1_ms: float
    build_q3_ms: float
    query_median_us: float
    query_q1_us: float
    query_q3_us: float
    storage_bytes: int
    storage_mib: float
    storage_bytes_per_value: float
    memory_bytes: int | None = None
    memory_mib: float | None = None
    memory_bytes_per_value: float | None = None
    peak_memory_bytes: int | None = None
    peak_memory_mib: float | None = None
    peak_memory_bytes_per_value: float | None = None
    process_memory_bytes: int | None = None
    process_memory_mib: float | None = None
    peak_process_memory_bytes: int | None = None
    peak_process_memory_mib: float | None = None
    baseline_process_memory_bytes: int | None = None
    baseline_process_memory_mib: float | None = None

    @property
    def build_median_ns_per_value(self) -> float:
        """Return median construction time divided by n, in ns per input value."""
        return self.build_median_ms * 1e6 / self.n

    @property
    def build_q1_ns_per_value(self) -> float:
        """Return the first construction-time quartile divided by n, in ns/value."""
        return self.build_q1_ms * 1e6 / self.n

    @property
    def build_q3_ns_per_value(self) -> float:
        """Return the third construction-time quartile divided by n, in ns/value."""
        return self.build_q3_ms * 1e6 / self.n


def summarize(measurements: Sequence[Measurement]) -> list[Summary]:
    """Aggregate raw trials into unit-labelled summaries for each algorithm/size.

    Reject mixed configurations within a group rather than combining unlike
    datasets or workloads into a single plotted point.
    """
    if not measurements:
        raise ValueError("at least one measurement is required")
    groups = defaultdict(list)
    for measurement in measurements:
        groups[(measurement.algorithm, measurement.n)].append(measurement)
    summaries = []
    for (algorithm, length), trials in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        configurations = {
            (row.queries, row.seed, row.pattern, row.workload, row.storage_bytes)
            for row in trials
        }
        if len(configurations) != 1:
            raise ValueError(f"incompatible trials for {algorithm} at n={length}")
        build, build_q1, build_q3 = _quartiles([row.build_ns / 1e6 for row in trials])
        query, query_q1, query_q3 = _quartiles(
            [row.query_mean_ns / 1e3 for row in trials]
        )
        memory_statistics = {}
        for name in (
            "memory",
            "peak_memory",
            "process_memory",
            "peak_process_memory",
            "baseline_process_memory",
        ):
            values = {getattr(row, f"{name}_bytes") for row in trials}
            if len(values) != 1:
                raise ValueError(
                    f"incompatible memory trials for {algorithm} at n={length}"
                )
            value = values.pop()
            if value is None:
                continue
            memory_statistics.update(
                {
                    f"{name}_bytes": value,
                    f"{name}_mib": value / 2**20,
                }
            )
            if name in ("memory", "peak_memory"):
                memory_statistics[f"{name}_bytes_per_value"] = value / length
        first = trials[0]
        summaries.append(
            Summary(
                algorithm=algorithm,
                n=length,
                queries=first.queries,
                repeats=len(trials),
                seed=first.seed,
                pattern=first.pattern,
                workload=first.workload,
                build_median_ms=build,
                build_q1_ms=build_q1,
                build_q3_ms=build_q3,
                query_median_us=query,
                query_q1_us=query_q1,
                query_q3_us=query_q3,
                storage_bytes=first.storage_bytes,
                storage_mib=first.storage_bytes / 2**20,
                storage_bytes_per_value=first.storage_bytes / length,
                **memory_statistics,
            )
        )
    return summaries


def load_measurements(path: Path) -> list[Measurement]:
    """Read raw trials, preserving unknown memory values in older CSV files."""
    field_types = Measurement.__annotations__
    optional_fields = {
        "memory_bytes",
        "peak_memory_bytes",
        "process_memory_bytes",
        "peak_process_memory_bytes",
        "baseline_process_memory_bytes",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(field_types) - optional_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        rss_fields = {
            "process_memory_bytes",
            "peak_process_memory_bytes",
            "baseline_process_memory_bytes",
        }
        present_rss_fields = rss_fields & set(reader.fieldnames or [])
        if present_rss_fields and present_rss_fields != rss_fields:
            raise ValueError("CSV process memory profile is missing columns")
        measurements = []
        for row in reader:
            try:
                measurement = Measurement(
                    **{
                        name: (int(row[name]) if row.get(name) else None)
                        if name in optional_fields
                        else kind(row[name])
                        for name, kind in field_types.items()
                    }
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid measurement on line {reader.line_num}"
                ) from error
            if measurement.algorithm not in ALGORITHMS:
                raise ValueError(f"unknown algorithm on line {reader.line_num}")
            current, peak = measurement.memory_bytes, measurement.peak_memory_bytes
            if (current is None) != (peak is None) or (
                current is not None and (current < 0 or peak < current)
            ):
                raise ValueError(
                    f"invalid memory measurement on line {reader.line_num}"
                )
            rss = (
                measurement.process_memory_bytes,
                measurement.peak_process_memory_bytes,
                measurement.baseline_process_memory_bytes,
            )
            if any(value is not None for value in rss) and (
                any(value is None for value in rss)
                or any(value < 0 for value in rss)
                or rss[1] < max(rss[0], rss[2])
            ):
                raise ValueError(
                    f"invalid process memory measurement on line {reader.line_num}"
                )
            measurements.append(measurement)
    return measurements


def _refresh_process_profiles(measurements: Sequence[Measurement]) -> list[Measurement]:
    """Refresh process RSS once per dataset/method configuration, preserving trials."""
    python_version, system = platform.python_version(), platform.platform()
    if any(
        row.python != python_version or row.platform != system for row in measurements
    ):
        raise ValueError(
            "process memory refresh requires the Python version and platform of "
            "the saved timing trials; use that environment or rerun the benchmark "
            "in the current environment"
        )
    profiles = {}
    refreshed = []
    for row in measurements:
        configuration = (
            row.algorithm,
            row.n,
            row.seed,
            row.pattern,
            row.queries,
            row.workload,
        )
        if configuration not in profiles:
            print(
                f"Profiling process memory: {row.algorithm:8} n={row.n:,}",
                file=sys.stderr,
            )
            profiles[configuration] = measure_process_memory(*configuration)
        current, peak, baseline = profiles[configuration]
        refreshed.append(
            replace(
                row,
                process_memory_bytes=current,
                peak_process_memory_bytes=peak,
                baseline_process_memory_bytes=baseline,
            )
        )
    return refreshed


def _write_refreshed_measurements(
    path: Path, measurements: Sequence[Measurement], source_rows: Sequence[dict]
) -> None:
    """Atomically add refreshed RSS while keeping original non-RSS CSV values exact."""
    rss_fields = (
        "process_memory_bytes",
        "peak_process_memory_bytes",
        "baseline_process_memory_bytes",
    )
    fieldnames = list(Measurement.__annotations__)
    for source_row in source_rows:
        fieldnames.extend(name for name in source_row if name not in fieldnames)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for measurement, source_row in zip(measurements, source_rows, strict=True):
                row = {**asdict(measurement), **source_row}
                row.update({name: getattr(measurement, name) for name in rss_fields})
                writer.writerow(row)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _save_normalized_query_plot(
    summaries: Sequence[Summary], output_dir: Path, pyplot: ModuleType
) -> Path | None:
    """Plot each method's query times divided by its growth model in its own panel.

    Convert medians and quartiles to nanoseconds before normalization. Segment
    tree points at n=1 are omitted because log2(1) is zero. Panel scales are
    independent, and only methods with eligible measurements are displayed.
    """
    specifications = (
        ("brute", "Query time / n (ns)", lambda n: n),
        ("sqrt", r"Query time / $\sqrt{n}$ (ns)", sqrt),
        ("segment", r"Query time / $\log_2 n$ (ns)", log2),
        ("sparse", "Query time (ns)", lambda n: 1),
    )
    panels = []
    for algorithm, ylabel, normalizer in specifications:
        rows = sorted(
            (
                row
                for row in summaries
                if row.algorithm == algorithm and (algorithm != "segment" or row.n > 1)
            ),
            key=lambda row: row.n,
        )
        if rows:
            panels.append((algorithm, ylabel, normalizer, rows))
    if not panels:
        return None
    columns = min(2, len(panels))
    rows_count = (len(panels) + columns - 1) // columns
    figure, axes = pyplot.subplots(
        rows_count, columns, figsize=(12, 4.2 * rows_count + 1), squeeze=False
    )
    figure.subplots_adjust(
        left=0.10, right=0.97, bottom=0.12, top=0.86, wspace=0.38, hspace=0.55
    )
    figure.suptitle(
        "Normalized range query time",
        x=0.10,
        y=0.96,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#172536",
    )
    for axis, (algorithm, ylabel, normalizer, rows) in zip(
        axes.flat, panels, strict=False
    ):
        sizes = [row.n for row in rows]
        factors = [normalizer(row.n) for row in rows]
        medians = [
            row.query_median_us * 1000 / factor
            for row, factor in zip(rows, factors, strict=True)
        ]
        lower = [
            row.query_q1_us * 1000 / factor
            for row, factor in zip(rows, factors, strict=True)
        ]
        upper = [
            row.query_q3_us * 1000 / factor
            for row, factor in zip(rows, factors, strict=True)
        ]
        axis.plot(
            sizes,
            medians,
            label=LABELS[algorithm],
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            linestyle=LINESTYLES[algorithm],
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.6,
        )
        axis.fill_between(
            sizes,
            lower,
            upper,
            color=COLORS[algorithm],
            alpha=0.13,
            linewidth=0,
        )
        axis.set_xscale("log")
        axis.set_ylim(bottom=0, top=max(upper) * 1.12 or 1)
        axis.set_title(
            LABELS[algorithm], loc="left", fontsize=12, weight="bold", pad=14
        )
        axis.set_xlabel("Number of input values (n)", labelpad=8)
        axis.set_ylabel(ylabel, labelpad=8)
        axis.grid(which="major", color="#d6dde5", linewidth=0.7)
        axis.grid(which="minor", color="#edf0f4", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    path = output_dir / "query_time_normalized.png"
    try:
        figure.savefig(path, dpi=180, facecolor="white")
    finally:
        pyplot.close(figure)
    return path


def save_plots(
    summaries: Sequence[Summary], output_dir: Path, pyplot: ModuleType
) -> list[Path]:
    """Save six focused comparisons using consistent colors, markers and dashes.

    Unknown legacy process RSS is omitted. Memory is measured once per
    algorithm/size; timing medians alone receive interquartile bands. Build
    comparisons use raw linear axes and a normalized ns/value view on log-x,
    linear-y axes to distinguish linear and n log n construction costs. Query
    comparisons also include independently normalized panels for each method.
    """
    if not summaries:
        raise ValueError("at least one summary is required")
    configurations = {
        (row.queries, row.pattern, row.workload, row.repeats) for row in summaries
    }
    if len(configurations) != 1:
        raise ValueError("plots require one workload and trial configuration")
    seeds_by_size = defaultdict(set)
    for row in summaries:
        seeds_by_size[row.n].add(row.seed)
    if any(len(seeds) != 1 for seeds in seeds_by_size.values()):
        raise ValueError("plots require matching dataset seeds at each input size")
    specifications = [
        (
            "build_time",
            "Construction time",
            "Build time (ms)",
            "build_median_ms",
            "build_q1_ms",
            "build_q3_ms",
            (("linear", "linear", "_linear"),),
        ),
        (
            "build_time_per_element",
            "Build time per input value",
            "Build time / n (ns per input value)",
            "build_median_ns_per_value",
            "build_q1_ns_per_value",
            "build_q3_ns_per_value",
            (("log", "linear", ""),),
        ),
        (
            "query_time",
            "Range query time",
            "Average time per query (us)",
            "query_median_us",
            "query_q1_us",
            "query_q3_us",
            (("log", "log", ""),),
        ),
        (
            "memory_growth",
            "Retained process memory after construction",
            "Retained process memory (MiB)",
            "process_memory_mib",
            None,
            None,
            (("linear", "linear", "_linear"),),
        ),
        (
            "peak_memory_growth",
            "Peak process memory through construction",
            "Peak process memory (MiB)",
            "peak_process_memory_mib",
            None,
            None,
            (("linear", "linear", "_linear"),),
        ),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_paths = {
        output_dir / f"{stem}{suffix}.png"
        for stem, _, _, _, _, _, axes in specifications
        for _, _, suffix in axes
    }
    managed_paths.update(output_dir / name for name in RETIRED_PLOT_NAMES)
    managed_paths.add(output_dir / "query_time_normalized.png")
    paths = []
    for stem, title, ylabel, metric, lower, upper, axes in specifications:
        series = {}
        for algorithm in ALGORITHMS:
            rows = sorted(
                (
                    row
                    for row in summaries
                    if row.algorithm == algorithm and getattr(row, metric) is not None
                ),
                key=lambda row: row.n,
            )
            if rows:
                series[algorithm] = rows
        if not series:
            continue
        for xscale, yscale, suffix in axes:
            figure, axis = pyplot.subplots(figsize=(9.2, 6.0))
            figure.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.86)
            figure.text(0.14, 0.94, title, fontsize=18, weight="bold", color="#172536")
            for algorithm, rows in series.items():
                sizes = [row.n for row in rows]
                values = [getattr(row, metric) for row in rows]
                axis.plot(
                    sizes,
                    values,
                    label=LABELS[algorithm],
                    color=COLORS[algorithm],
                    marker=MARKERS[algorithm],
                    linestyle=LINESTYLES[algorithm],
                    linewidth=2,
                    markersize=6,
                    markerfacecolor="white",
                    markeredgewidth=1.6,
                )
                if lower is not None:
                    axis.fill_between(
                        sizes,
                        [getattr(row, lower) for row in rows],
                        [getattr(row, upper) for row in rows],
                        color=COLORS[algorithm],
                        alpha=0.13,
                        linewidth=0,
                    )
            axis.set_xscale(xscale)
            axis.set_yscale(yscale)
            if xscale == "linear":
                axis.set_xlim(left=0)
                axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            if yscale == "linear":
                axis.set_ylim(bottom=0)
            axis.set_xlabel("Number of input values (n)", labelpad=10)
            axis.set_ylabel(ylabel, labelpad=10)
            axis.grid(which="major", color="#d6dde5", linewidth=0.7)
            if xscale == "log":
                axis.grid(which="minor", color="#edf0f4", linewidth=0.4)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            # Fixed legend position keeps labels from covering almost-flat curves.
            axis.legend(
                loc="upper left",
                bbox_to_anchor=(0, -0.18),
                ncol=2,
                fontsize=9,
                frameon=False,
                borderaxespad=0,
            )
            path = output_dir / f"{stem}{suffix}.png"
            try:
                figure.savefig(path, dpi=180, facecolor="white")
            finally:
                pyplot.close(figure)
            paths.append(path)
    normalized_path = _save_normalized_query_plot(summaries, output_dir, pyplot)
    if normalized_path is not None:
        paths.append(normalized_path)
    # A successful replot must not leave obsolete comparisons from an earlier run.
    for stale_path in managed_paths - set(paths):
        stale_path.unlink(missing_ok=True)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark with CLI options and write raw CSV, summary CSV and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=_positive_int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    parser.add_argument("--queries", type=_positive_int, default=1000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--pattern", choices=PATTERNS, default="random")
    parser.add_argument("--workload", choices=WORKLOADS, default="wide")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="regenerate summaries and plots from raw CSV without rerunning timings",
    )
    parser.add_argument(
        "--refresh-memory",
        action="store_true",
        help="refresh whole-process RSS for saved trials without rerunning timings",
    )
    args = parser.parse_args(argv)
    if args.refresh_memory and args.from_csv is None:
        parser.error("--refresh-memory requires --from-csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pyplot = _load_pyplot(args.output_dir)
    except ImportError as error:
        parser.error(
            f"Matplotlib is required for plots: {error}. "
            "Use the Nix shell or install it with 'python -m pip install matplotlib'."
        )
    sizes = sorted(set(args.sizes))
    algorithms = list(dict.fromkeys(args.algorithms))
    paths = []
    if args.from_csv is not None:
        try:
            measurements = load_measurements(args.from_csv)
            if args.refresh_memory:
                with args.from_csv.open(newline="", encoding="utf-8") as stream:
                    source_rows = list(csv.DictReader(stream))
                measurements = _refresh_process_profiles(measurements)
            summaries = summarize(measurements)
            if args.refresh_memory:
                raw_path = args.output_dir / "measurements.csv"
                _write_refreshed_measurements(raw_path, measurements, source_rows)
                paths.append(raw_path)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(f"Loaded measurements from {args.from_csv}")
        if any(row.process_memory_bytes is None for row in measurements):
            print(
                "Some saved trials have no process RSS profile; unavailable process "
                "memory points are omitted. Use --refresh-memory with --from-csv "
                "to measure them without rerunning timings.",
                file=sys.stderr,
            )
    else:
        measurements = []
        raw_path = args.output_dir / "measurements.csv"
        with raw_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(Measurement.__annotations__)
            )
            writer.writeheader()
            for row in run_benchmarks(
                sizes,
                algorithms,
                args.queries,
                args.repeats,
                args.seed,
                args.pattern,
                args.workload,
            ):
                writer.writerow(asdict(row))
                stream.flush()
                measurements.append(row)
                print(
                    f"{row.algorithm:8} n={row.n:>10,} "
                    f"trial={row.trial}/{args.repeats} "
                    f"build={row.build_ns / 1e6:>9.3f} ms "
                    f"query={row.query_mean_ns / 1e3:>7.3f} us "
                    f"storage={row.storage_bytes / 2**20:>8.3f} MiB "
                    f"retained-rss={row.process_memory_bytes / 2**20:>8.3f} MiB "
                    f"peak-rss={row.peak_process_memory_bytes / 2**20:>8.3f} MiB",
                    file=sys.stderr,
                )
        paths.append(raw_path)
        summaries = summarize(measurements)
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Summary.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in summaries)
    paths.extend([summary_path, *save_plots(summaries, args.output_dir, pyplot)])
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
