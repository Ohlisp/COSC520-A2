"""Verify workload fairness and benchmark output, without testing elapsed speeds."""

import csv
import json
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_core import (
    ALGORITHMS,
    PATTERNS,
    WORKLOADS,
    _measure_memory,
    estimate_storage_bytes,
    generate_queries,
    generate_values,
    main,
    measure_process_memory,
    run_benchmarks,
)


@pytest.fixture
def process_profiles(monkeypatch):
    """Keep ordinary workload tests independent of child processes and OS RSS."""
    calls = []

    def profile(*args):
        assert not tracemalloc.is_tracing()
        calls.append(args)
        return 100_000_000, 110_000_000, 80_000_000

    monkeypatch.setattr("benchmark_core.measure_process_memory", profile)
    return calls


def test_static_estimate_cli_remains_available(capsys):
    """Merging the update runner must preserve allocation-only static runs."""
    main(["--sizes", "3", "--algorithms", "brute", "--estimate-only"])
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert rows == [{"algorithm": "brute", "n": "3", "storage_bytes": "24"}]


def test_csv_runner_defaults_to_shared_results_folder(
    tmp_path, monkeypatch, process_profiles
):
    """The CSV-only entry point uses the same static raw file as the plot runner."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--sizes",
            "3",
            "--algorithms",
            "brute",
            "--queries",
            "1",
            "--repeats",
            "1",
        ],
    )
    main()
    with (tmp_path / "results" / "measurements.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "brute"
    assert {path.name for path in (tmp_path / "results").iterdir()} == {
        "measurements.csv"
    }


@pytest.mark.parametrize("pattern", PATTERNS)
def test_reproducible_data(pattern):
    """The saved seed and pattern must reproduce the exact packed dataset."""
    assert generate_values(19, 520, pattern) == generate_values(19, 520, pattern)


@pytest.mark.parametrize("workload", WORKLOADS)
@pytest.mark.parametrize("length", [1, 3, 65])
def test_queries_are_reproducible_and_valid(workload, length):
    """All query distributions must include only valid ranges, including at n=1."""
    queries = generate_queries(length, 200, 520, workload)
    assert queries == generate_queries(length, 200, 520, workload)
    assert all(0 <= start < stop <= length for start, stop in queries)
    if workload == "short":
        assert all(stop - start <= 64 for start, stop in queries)
    elif workload == "wide":
        assert all(stop - start >= max(1, length // 2) for start, stop in queries)
    elif workload == "singleton":
        assert all(stop - start == 1 for start, stop in queries)


@pytest.mark.parametrize("length", [1, 2, 3, 16, 17, 1000])
def test_space_estimate_matches_allocated_payload(length):
    """No-allocation estimates must match actual logical buffer sizes."""
    for name, constructor in ALGORITHMS.items():
        assert (
            estimate_storage_bytes(name, length)
            == constructor(range(length)).storage_bytes
        )


def test_comparison_records_consistent_trials(process_profiles):
    """Check all algorithms receive matching inputs and emit complete raw trials."""
    results = list(run_benchmarks([1, 33], list(ALGORITHMS), query_count=20, repeats=2))
    assert len(results) == 16
    for length in (1, 33):
        rows = [row for row in results if row.n == length]
        assert len({row.checksum for row in rows}) == 1
        assert {row.trial for row in rows} == {1, 2}
        for row in rows:
            assert row.queries == 20
            assert row.seed == 520 + length
            assert row.build_ns >= 0 and row.query_total_ns >= 0
            assert row.query_mean_ns == row.query_total_ns / row.queries
            assert row.storage_bytes == estimate_storage_bytes(row.algorithm, length)
            assert row.peak_memory_bytes >= row.memory_bytes >= row.storage_bytes
            assert row.process_memory_bytes == 100_000_000
            assert row.peak_process_memory_bytes == 110_000_000
            assert row.baseline_process_memory_bytes == 80_000_000
        for algorithm in ALGORITHMS:
            profiles = {
                (row.memory_bytes, row.peak_memory_bytes)
                for row in rows
                if row.algorithm == algorithm
            }
            assert len(profiles) == 1  # One separate memory pass per algorithm/size.
    assert process_profiles == [
        (algorithm, length, 520 + length, "random", 20, "random")
        for length in (1, 33)
        for algorithm in ALGORITHMS
    ]


def test_memory_profile_captures_constructor_temporaries():
    """A segment tree releases its packed input copy before retained memory is read."""
    values = generate_values(4096, 520)
    retained, peak = _measure_memory(ALGORITHMS["segment"], values)
    assert retained >= estimate_storage_bytes("segment", len(values))
    assert peak - retained >= len(values) * values.itemsize
    assert not tracemalloc.is_tracing()


def test_memory_profile_is_separate_from_timing(monkeypatch, process_profiles):
    """Only the extra constructor call is traced; timing queries remain untraced."""
    states = []
    base = ALGORITHMS["brute"]

    class ObservedBruteForce(base):
        def __init__(self, values):
            states.append(tracemalloc.is_tracing())
            super().__init__(values)

        def query(self, start, stop):
            assert not tracemalloc.is_tracing()
            return super().query(start, stop)

    monkeypatch.setitem(ALGORITHMS, "brute", ObservedBruteForce)
    list(run_benchmarks([10], ["brute"], query_count=3, repeats=2))
    assert states == [True, False, False]
    assert len(process_profiles) == 1


def test_memory_profile_stops_tracing_after_failure():
    """A failed construction must not leave later timing runs under tracing."""

    def fail(values):
        raise ValueError("construction failed")

    with pytest.raises(ValueError, match="construction failed"):
        _measure_memory(fail, generate_values(10, 520))
    assert not tracemalloc.is_tracing()


def test_benchmark_preserves_existing_tracing_session():
    """Do not silently stop a caller's profiler or time instrumented code."""
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="tracemalloc"):
            list(run_benchmarks([10], ["brute"]))
        with pytest.raises(RuntimeError, match="tracemalloc"):
            _measure_memory(ALGORITHMS["brute"], generate_values(10, 520))
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_count": 0},
        {"repeats": 0},
        {"sizes": []},
        {"sizes": [0]},
        {"algorithms": []},
        {"algorithms": ["unknown"]},
        {"pattern": "unknown"},
        {"workload": "unknown"},
    ],
)
def test_invalid_benchmark_configuration(kwargs, process_profiles):
    """Reject runs that would otherwise produce misleading or empty results."""
    config = {"sizes": [5], "algorithms": ["sqrt"], **kwargs}
    with pytest.raises(ValueError):
        list(run_benchmarks(**config))


def test_process_profile_uses_local_source_and_actual_seed(monkeypatch):
    """A fresh worker must import this checkout and not add n to the saved seed."""
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "process_memory_bytes": 200,
                    "peak_process_memory_bytes": 300,
                    "baseline_process_memory_bytes": 100,
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/some/other/package")
    monkeypatch.setenv("PYTHONTRACEMALLOC", "10")
    assert measure_process_memory("brute", 33, 553, "constant", 20, "wide") == (
        200,
        300,
        100,
    )
    command = observed["command"]
    assert command[:3] == [sys.executable, "-m", "memory_probe"]
    assert command[command.index("--seed") + 1] == "553"
    source = str(Path(__file__).resolve().parents[1] / "src")
    assert observed["env"]["PYTHONPATH"] == source + os.pathsep + "/some/other/package"
    assert "PYTHONTRACEMALLOC" not in observed["env"]
    assert observed["check"] and observed["capture_output"] and observed["text"]


@pytest.mark.parametrize(
    "output",
    [
        "not JSON",
        "[]",
        "{}",
        '{"process_memory_bytes": true, "peak_process_memory_bytes": 3, '
        '"baseline_process_memory_bytes": 1}',
        '{"process_memory_bytes": 2.5, "peak_process_memory_bytes": 3, '
        '"baseline_process_memory_bytes": 1}',
        '{"process_memory_bytes": -1, "peak_process_memory_bytes": 3, '
        '"baseline_process_memory_bytes": 1}',
        '{"process_memory_bytes": 4, "peak_process_memory_bytes": 3, '
        '"baseline_process_memory_bytes": 1}',
        '{"process_memory_bytes": 2, "peak_process_memory_bytes": 3, '
        '"baseline_process_memory_bytes": 4}',
    ],
)
def test_process_profile_rejects_invalid_worker_results(monkeypatch, output):
    """Do not turn failed, noninteger or inconsistent RSS readings into plots."""
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output)
    )
    with pytest.raises(RuntimeError, match="invalid process memory worker output"):
        measure_process_memory("brute", 10, 530, "random", 10, "wide")


@pytest.mark.parametrize(
    "error, message",
    [
        (
            subprocess.CalledProcessError(1, "worker", stderr="RSS reading failed"),
            "process memory worker failed: RSS reading failed",
        ),
        (OSError("cannot execute"), "cannot start process memory worker"),
    ],
)
def test_process_profile_reports_worker_failure(monkeypatch, error, message):
    """Surface the worker diagnostic instead of saving incomplete memory data."""

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match=message):
        measure_process_memory("brute", 10, 530, "random", 10, "wide")


def test_process_probe_matches_shared_workload_and_reading_order(monkeypatch):
    """Keep generated input/queries alive and read baseline, retained, then peak."""
    import memory_probe

    calls = []
    values = generate_values(10, 530)
    queries = [(0, 10)] * 3

    def data(length, seed, pattern):
        calls.append(("data", length, seed, pattern))
        return values

    def intervals(length, count, seed, workload):
        calls.append(("queries", length, count, seed, workload))
        return queries

    def constructor(data):
        assert data is values
        calls.append("build")
        return object()

    readings = iter([100, 200])

    def current():
        calls.append("current")
        return next(readings)

    def peak():
        calls.append("peak")
        return 300

    monkeypatch.setattr(memory_probe, "generate_values", data)
    monkeypatch.setattr(memory_probe, "generate_queries", intervals)
    monkeypatch.setitem(memory_probe.ALGORITHMS, "brute", constructor)
    monkeypatch.setattr(memory_probe, "_current_rss_bytes", current)
    monkeypatch.setattr(memory_probe, "_peak_rss_bytes", peak)
    assert memory_probe.profile_process_memory(
        "brute", 10, 530, "random", 3, "wide"
    ) == {
        "process_memory_bytes": 200,
        "peak_process_memory_bytes": 300,
        "baseline_process_memory_bytes": 100,
    }
    assert calls == [
        ("data", 10, 530, "random"),
        ("queries", 10, 3, 530 ^ 0x520, "wide"),
        "current",
        "build",
        "current",
        "peak",
    ]


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="RSS profiling supports macOS and Linux",
)
def test_real_process_profile_includes_interpreter_and_constructor_memory():
    """A fresh segment-tree process captures input, tree and interpreter RSS."""
    length = 1_000_000
    retained, peak, baseline = measure_process_memory(
        "segment", length, 520 + length, "constant", 10, "wide"
    )
    payload = estimate_storage_bytes("segment", length)
    assert baseline > length * 8  # Shared input plus the interpreter/allocator.
    assert retained > payload + length * 8
    # Large array copies fault in real pages and are released after construction.
    assert retained - baseline > payload // 2
    assert peak >= retained and peak >= baseline
    # Freed construction buffers may remain resident in the process allocator;
    # unlike traced allocations, RSS need not fall when their Python owners die.
