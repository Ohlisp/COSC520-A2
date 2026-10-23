"""Check reproducible update comparisons without asserting elapsed speeds."""

import csv
import sys
from dataclasses import replace

import pytest

import benchmark_core
from benchmark_core import (
    UPDATE_ALGORITHMS,
    generate_updates,
    run_update_benchmarks,
    summarize_updates,
)


@pytest.mark.parametrize(
    "arguments, expected_option",
    [(["--help"], "--queries QUERIES"), (["updates", "--help"], "--updates UPDATES")],
)
def test_benchmark_modes_show_their_own_options(arguments, expected_option, capsys):
    """The combined command must expose both original command-line interfaces."""
    with pytest.raises(SystemExit) as error:
        benchmark_core.main(arguments)
    assert error.value.code == 0
    assert expected_option in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [["updates", "--sizes", "0"], ["updates", "--queries", "1"], ["--updates", "1"]],
)
def test_benchmark_modes_reject_invalid_or_misrouted_options(
    arguments, tmp_path, monkeypatch
):
    """Invalid input must not silently start a different experiment."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        benchmark_core.main(arguments)
    assert error.value.code == 2
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("length", [1, 3, 65])
def test_assignments_are_reproducible_and_valid(length):
    """Every generated assignment must fit the common point-update API."""
    updates = generate_updates(length, 200, 520)
    assert len(updates) == 200
    assert updates == generate_updates(length, 200, 520)
    assert updates != generate_updates(length, 200, 521)
    assert all(
        0 <= position < length
        and type(value) is int
        and -(1 << 63) <= value < (1 << 63)
        for position, value in updates
    )


@pytest.mark.parametrize("length, count", [(0, 10), (-1, 10), (5, 0), (5, -1)])
def test_assignments_reject_nonpositive_configuration(length, count):
    """Invalid workloads must fail before a misleading empty timing is saved."""
    with pytest.raises(ValueError):
        generate_updates(length, count, 520)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sizes": []},
        {"sizes": [0]},
        {"sizes": [-1]},
        {"update_count": 0},
        {"update_count": -1},
        {"repeats": 0},
        {"repeats": -1},
    ],
)
def test_update_benchmark_rejects_invalid_configuration(kwargs):
    """Positive sizes, assignment counts and trial counts are required."""
    config = {"sizes": [5], "update_count": 4, "repeats": 1, **kwargs}
    with pytest.raises(ValueError):
        list(run_update_benchmarks(**config))


def test_update_comparison_only_includes_mutable_structures():
    """A sparse-table rebuild is not a point assignment."""
    assert set(UPDATE_ALGORITHMS) == {"brute", "sqrt", "segment"}
    assert all(
        hasattr(constructor, "update") for constructor in UPDATE_ALGORITHMS.values()
    )


def test_update_trials_use_matching_workloads_and_final_checksums():
    """All methods and repetitions receive identical input and assignments."""
    measurements = list(run_update_benchmarks([1, 33], update_count=20, repeats=3))
    assert len(measurements) == 18
    for length in (1, 33):
        rows = [row for row in measurements if row.n == length]
        assert len({row.checksum for row in rows}) == 1
        assert {row.algorithm for row in rows} == set(UPDATE_ALGORITHMS)
        for algorithm in UPDATE_ALGORITHMS:
            assert {row.trial for row in rows if row.algorithm == algorithm} == {
                1,
                2,
                3,
            }
        for row in rows:
            assert row.updates == 20
            assert row.seed == 520 + length
            assert row.update_seed == (520 + length) ^ 0xA520
            assert row.update_total_ns >= 0
            assert row.update_mean_ns == row.update_total_ns / row.updates
            assert row.python and row.platform


@pytest.fixture
def raw_trial():
    """Use a real trial so summary checks share the runner's record contract."""
    return next(run_update_benchmarks([1], update_count=4, repeats=1))


def test_update_summary_converts_units_and_uses_inclusive_quartiles(raw_trial):
    """Summarize trial batch averages in microseconds, retaining the workload."""
    trials = [
        replace(
            raw_trial,
            trial=trial,
            update_total_ns=mean_ns * raw_trial.updates,
            update_mean_ns=mean_ns,
        )
        for trial, mean_ns in enumerate((1000, 2000, 3000, 4000), start=1)
    ]
    rows = summarize_updates(trials)
    assert len(rows) == 1
    row = rows[0]
    assert row["algorithm"] == raw_trial.algorithm
    assert row["n"] == raw_trial.n
    assert row["updates"] == raw_trial.updates
    assert row["seed"] == raw_trial.seed
    assert row["update_seed"] == raw_trial.update_seed
    assert row["repeats"] == 4
    assert row["update_median_us"] == pytest.approx(2.5)
    assert row["update_q1_us"] == pytest.approx(1.75)
    assert row["update_q3_us"] == pytest.approx(3.25)


def test_update_summary_preserves_single_trial(raw_trial):
    """A single trial has no measured spread across repetitions."""
    row = summarize_updates([raw_trial])[0]
    mean_us = raw_trial.update_mean_ns / 1000
    assert row["repeats"] == 1
    assert row["update_median_us"] == pytest.approx(mean_us)
    assert row["update_q1_us"] == pytest.approx(mean_us)
    assert row["update_q3_us"] == pytest.approx(mean_us)


@pytest.mark.parametrize(
    "changes",
    [
        {"updates": 5},
        {"seed": 999},
        {"update_seed": 999},
        {"python": "another Python version"},
        {"platform": "another platform"},
    ],
)
def test_update_summary_rejects_mixed_workloads_or_environments(raw_trial, changes):
    """Trials at one algorithm/size must describe a single comparable experiment."""
    with pytest.raises(ValueError):
        summarize_updates([raw_trial, replace(raw_trial, trial=2, **changes)])


def test_flat_update_outputs_preserve_static_results_and_remove_semilogy(
    tmp_path, monkeypatch
):
    """Both runners can share results/ without overwriting each other's data."""
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    preserved = {
        "measurements.csv": b"static raw data",
        "summary.csv": b"static summary",
        "query_time.png": b"static plot",
        "unrelated.png": b"unrelated image",
    }
    for name, content in preserved.items():
        (output / name).write_bytes(content)
    (output / "update_time_semilogy.png").write_bytes(b"obsolete plot")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_core",
            "updates",
            "--sizes",
            "4",
            "--updates",
            "3",
            "--repeats",
            "1",
        ],
    )
    benchmark_core.main()
    raw_path = output / "update_measurements.csv"
    raw_data = raw_path.read_bytes()
    with raw_path.open(newline="") as stream:
        assert {row["algorithm"] for row in csv.DictReader(stream)} == {
            "brute",
            "sqrt",
            "segment",
        }
    assert (output / "update_summary.csv").is_file()
    assert {path.name for path in output.glob("update_time*.png")} == {
        "update_time.png",
        "update_time_linear.png",
        "update_time_semilogx.png",
        "update_time_growth_axes.png",
    }

    def unexpected_timing(*args, **kwargs):
        pytest.fail("saved-CSV regeneration must not rerun timing")

    monkeypatch.setattr(benchmark_core, "run_update_benchmarks", unexpected_timing)
    monkeypatch.setattr(
        sys, "argv", ["benchmark_core", "updates", "--from-csv", str(raw_path)]
    )
    benchmark_core.main()
    assert raw_path.read_bytes() == raw_data
    for name, content in preserved.items():
        assert (output / name).read_bytes() == content
    assert not any(
        path.is_dir() for path in output.iterdir() if path.name != ".matplotlib"
    )
