"""Check aggregation units/statistics and the complete CSV/plot output pipeline."""

import csv
import platform
from dataclasses import asdict, replace
from importlib.util import find_spec
from pathlib import Path

import pytest

from benchmark_core import Measurement
from benchmark_report import (
    load_measurements,
    main,
    save_plots,
    summarize,
)


def measurement(**changes):
    """Return a known raw trial, allowing individual measurements to be overridden."""
    row = Measurement(
        algorithm="sqrt",
        n=1000,
        queries=100,
        trial=1,
        seed=1520,
        pattern="random",
        workload="wide",
        build_ns=1_000_000,
        query_total_ns=100_000,
        query_mean_ns=1000,
        storage_bytes=8192,
        checksum=-100,
        python="3.10.0",
        platform="test",
        memory_bytes=9000,
        peak_memory_bytes=12000,
        process_memory_bytes=20 * 2**20,
        peak_process_memory_bytes=24 * 2**20,
        baseline_process_memory_bytes=16 * 2**20,
    )
    return replace(row, **changes)


def test_summary_uses_medians_quartiles_and_correct_units():
    """A slow outlier must not become the median or alter the space measurement."""
    rows = [
        measurement(trial=1, build_ns=1_000_000, query_mean_ns=1000),
        measurement(trial=2, build_ns=3_000_000, query_mean_ns=3000),
        measurement(trial=3, build_ns=9_000_000, query_mean_ns=9000),
    ]
    (result,) = summarize(rows)
    assert result.repeats == 3
    assert (result.build_median_ms, result.build_q1_ms, result.build_q3_ms) == (3, 2, 6)
    assert (
        result.build_median_ns_per_value,
        result.build_q1_ns_per_value,
        result.build_q3_ns_per_value,
    ) == (3000, 2000, 6000)
    assert (result.query_median_us, result.query_q1_us, result.query_q3_us) == (3, 2, 6)
    assert result.storage_bytes == 8192
    assert result.storage_mib == 8192 / 2**20
    assert result.storage_bytes_per_value == 8192 / 1000
    assert result.memory_bytes == 9000
    assert result.memory_mib == 9000 / 2**20
    assert result.memory_bytes_per_value == 9
    assert result.peak_memory_bytes == 12000
    assert result.peak_memory_mib == 12000 / 2**20
    assert result.peak_memory_bytes_per_value == 12
    assert result.process_memory_bytes == 20 * 2**20
    assert result.process_memory_mib == 20
    assert result.peak_process_memory_bytes == 24 * 2**20
    assert result.peak_process_memory_mib == 24
    assert result.baseline_process_memory_bytes == 16 * 2**20
    assert result.baseline_process_memory_mib == 16


def test_single_trial_and_separate_groups():
    """One repeat must work, and different algorithms/sizes must stay separate."""
    results = summarize(
        [
            measurement(),
            measurement(algorithm="segment"),
            measurement(n=10000),
        ]
    )
    assert len(results) == 3
    assert {(row.algorithm, row.n) for row in results} == {
        ("sqrt", 1000),
        ("segment", 1000),
        ("sqrt", 10000),
    }
    assert all(
        row.build_median_ms == row.build_q1_ms == row.build_q3_ms for row in results
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 99},
        {"queries": 200},
        {"workload": "short"},
        {"memory_bytes": None, "peak_memory_bytes": None},
        {"process_memory_bytes": 21 * 2**20},
    ],
)
def test_summary_rejects_incompatible_trials(changes):
    """Do not combine different datasets or query workloads into one plotted point."""
    with pytest.raises(ValueError, match="incompatible"):
        summarize([measurement(), measurement(**changes)])


def test_summary_rejects_empty_results():
    """An empty CSV must not be represented as a successful comparison."""
    with pytest.raises(ValueError, match="at least one"):
        summarize([])


def test_report_runner_defaults_to_shared_results_folder(tmp_path, monkeypatch):
    """The plot pipeline keeps its CSVs and figures directly under results/."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("benchmark_report._load_pyplot", lambda _: None)
    monkeypatch.setattr(
        "benchmark_report.run_benchmarks",
        lambda *args: iter([measurement()]),
    )
    output_dirs = []

    def record_plot_destination(summaries, output_dir, pyplot):
        output_dirs.append(output_dir)
        return []

    monkeypatch.setattr("benchmark_report.save_plots", record_plot_destination)
    main([])
    assert output_dirs == [Path("results")]
    assert {path.name for path in (tmp_path / "results").iterdir()} == {
        "measurements.csv",
        "summary.csv",
    }


def test_runner_creates_csv_and_png_outputs(tmp_path, monkeypatch):
    """Run all structures end to end and validate CSV contents and image files."""
    if find_spec("matplotlib") is None:
        pytest.skip("plotting requires optional matplotlib dependency")
    main(
        [
            "--sizes",
            "17",
            "3",
            "17",
            "--queries",
            "10",
            "--repeats",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    with (tmp_path / "measurements.csv").open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    with (tmp_path / "summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert len(raw) == len(summary) == 8
    assert {row["algorithm"] for row in raw} == {"brute", "sqrt", "segment", "sparse"}
    for length in (3, 17):
        rows = [row for row in raw if int(row["n"]) == length]
        assert len({row["checksum"] for row in rows}) == 1
    assert all(float(row["storage_mib"]) > 0 for row in summary)
    assert all(
        float(row["peak_memory_mib"]) >= float(row["memory_mib"]) > 0 for row in summary
    )
    assert all(
        float(row["peak_process_memory_mib"]) >= float(row["process_memory_mib"]) > 0
        and float(row["peak_process_memory_mib"])
        >= float(row["baseline_process_memory_mib"])
        for row in summary
    )
    assert all(
        float(row["storage_bytes_per_value"])
        == int(row["storage_bytes"]) / int(row["n"])
        for row in summary
    )
    from matplotlib.image import imread

    names = (
        "build_time_linear.png",
        "build_time_per_element.png",
        "query_time.png",
        "query_time_normalized.png",
        "memory_growth_linear.png",
        "peak_memory_growth_linear.png",
    )
    assert {path.name for path in tmp_path.glob("*.png")} == set(names)
    for name in names:
        pixels = imread(tmp_path / name)
        assert pixels.shape[0] > 500 and pixels.shape[1] > 500
        assert pixels.min() < pixels.max()  # Actual rendered content, not blank PNGs.

    # Replotting must leave the input intact and use genuine linear/log axes.
    from matplotlib.figure import Figure

    raw_path = tmp_path / "measurements.csv"
    original_csv = raw_path.read_bytes()
    assert len(load_measurements(raw_path)) == 8
    observed = {}
    normalized_panel_titles = []
    original_savefig = Figure.savefig

    def record_axes(figure, path, *args, **kwargs):
        axis = figure.axes[0]
        assert len(figure.texts) == 1
        assert figure.texts[0].get_text()
        assert all(not panel.texts for panel in figure.axes)
        if path.name == "memory_growth_linear.png":
            assert axis.get_ylabel() == "Retained process memory (MiB)"
        elif path.name == "peak_memory_growth_linear.png":
            assert axis.get_ylabel() == "Peak process memory (MiB)"
        if path.name == "query_time_normalized.png":
            visible = [panel for panel in figure.axes if panel.get_visible()]
            normalized_panel_titles.extend(
                panel.get_title(loc="left") for panel in visible
            )
            assert len(visible) == 4
            for panel in visible:
                assert panel.get_xscale() == "log"
                assert panel.get_yscale() == "linear"
                assert panel.get_ylim()[0] == 0
                assert len(panel.lines) == 1
                assert "ns" in panel.get_ylabel()
        observed[path.name] = (
            axis.get_xscale(),
            axis.get_yscale(),
            axis.get_ylim()[0],
            len(axis.lines),
            {line.get_linestyle() for line in axis.lines},
            [tuple(line.get_ydata()) for line in axis.lines],
        )
        return original_savefig(figure, path, *args, **kwargs)

    def reject_benchmark(*args, **kwargs):
        pytest.fail("replotting must not run new timing trials")

    monkeypatch.setattr(Figure, "savefig", record_axes)
    monkeypatch.setattr("benchmark_report.run_benchmarks", reject_benchmark)
    output_dir = tmp_path / "replotted"
    main(["--from-csv", str(raw_path), "--output-dir", str(output_dir)])
    assert raw_path.read_bytes() == original_csv
    assert (output_dir / "summary.csv").read_bytes() == (
        tmp_path / "summary.csv"
    ).read_bytes()
    assert set(observed) == set(names)
    for name, (xscale, yscale, bottom, lines, styles, values) in observed.items():
        expected_x = "linear" if name.endswith("_linear.png") else "log"
        expected_y = (
            "linear" if "per_element" in name or "normalized" in name else expected_x
        )
        assert xscale == expected_x and yscale == expected_y
        if expected_y == "linear":
            assert bottom == 0
        expected_lines = 1 if name == "query_time_normalized.png" else 4
        assert lines == len(styles) == expected_lines
        assert all(len(points) == 2 for points in values)
    assert normalized_panel_titles == [
        "Naive brute force",
        "Square root decomposition",
        "Segment tree",
        "Sparse table",
    ]
    lookup = {(row["algorithm"], int(row["n"])): row for row in summary}
    normalized_builds = observed["build_time_per_element.png"][-1]
    for algorithm, points in zip(
        ("brute", "sqrt", "segment", "sparse"), normalized_builds, strict=True
    ):
        assert points == pytest.approx(
            [
                float(lookup[(algorithm, n)]["build_median_ms"]) * 1e6 / n
                for n in (3, 17)
            ]
        )
    for name, metric in (
        ("memory_growth_linear.png", "process_memory_mib"),
        ("peak_memory_growth_linear.png", "peak_process_memory_mib"),
    ):
        for algorithm, points in zip(
            ("brute", "sqrt", "segment", "sparse"), observed[name][-1], strict=True
        ):
            assert points == pytest.approx(
                [float(lookup[(algorithm, n)][metric]) for n in (3, 17)]
            )


def test_loader_rejects_missing_columns(tmp_path):
    """Reject a summary or unrelated CSV instead of plotting it as raw trials."""
    path = tmp_path / "wrong.csv"
    path.write_text("algorithm,n\nsqrt,1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_measurements(path)


@pytest.mark.parametrize("has_traced_memory", [False, True])
def test_legacy_csv_keeps_memory_unknown_and_omits_memory_plots(
    tmp_path, has_traced_memory
):
    """Neither missing memory nor an older trace is presented as process RSS."""
    path = tmp_path / "legacy.csv"
    row = asdict(measurement())
    if not has_traced_memory:
        del row["memory_bytes"], row["peak_memory_bytes"]
    for name in (
        "process_memory_bytes",
        "peak_process_memory_bytes",
        "baseline_process_memory_bytes",
    ):
        del row[name]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (loaded,) = load_measurements(path)
    assert loaded.memory_bytes == (9000 if has_traced_memory else None)
    assert loaded.peak_memory_bytes == (12000 if has_traced_memory else None)
    assert loaded.process_memory_bytes is loaded.peak_process_memory_bytes is None
    assert loaded.baseline_process_memory_bytes is None
    (summary,) = summarize([loaded])
    assert summary.process_memory_mib is summary.peak_process_memory_mib is None
    assert summary.baseline_process_memory_mib is None
    if find_spec("matplotlib") is None:
        return
    from matplotlib import pyplot

    paths = save_plots([summary], tmp_path / "plots", pyplot)
    assert all("memory" not in path.name for path in paths)


@pytest.mark.parametrize(
    "current,peak", [(100, 99), (-1, 100), (None, 100), (100, None)]
)
def test_loader_rejects_invalid_memory_profiles(tmp_path, current, peak):
    """Missing halves, negative values, and impossible peaks are invalid profiles."""
    path = tmp_path / "invalid.csv"
    row = asdict(measurement(memory_bytes=current, peak_memory_bytes=peak))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="invalid memory"):
        load_measurements(path)


@pytest.mark.parametrize(
    "current,peak,baseline",
    [
        (-1, 20, 5),
        (10, 9, 5),
        (10, 20, -1),
        (10, 20, 21),
        (None, 20, 5),
        (10, None, 5),
        (10, 20, None),
    ],
)
def test_loader_rejects_invalid_process_memory_profiles(
    tmp_path, current, peak, baseline
):
    """Process RSS must be a complete nonnegative triple with a consistent peak."""
    path = tmp_path / "invalid-rss.csv"
    row = asdict(
        measurement(
            process_memory_bytes=current,
            peak_process_memory_bytes=peak,
            baseline_process_memory_bytes=baseline,
        )
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="invalid process memory"):
        load_measurements(path)


def test_loader_rejects_incomplete_process_memory_columns(tmp_path):
    """An incomplete RSS schema is distinct from a legacy CSV with no RSS fields."""
    path = tmp_path / "incomplete-rss.csv"
    row = asdict(measurement())
    del row["baseline_process_memory_bytes"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="process memory.*missing columns"):
        load_measurements(path)


@pytest.mark.parametrize("in_place", [False, True])
def test_refresh_memory_deduplicates_profiles_and_preserves_existing_csv_values(
    tmp_path, monkeypatch, in_place
):
    """Refresh RSS alone, retaining timings and traced/payload values verbatim."""
    path = tmp_path / "measurements.csv"
    rows = [
        asdict(
            measurement(
                algorithm=algorithm,
                trial=trial,
                python=platform.python_version(),
                platform=platform.platform(),
                process_memory_bytes=None,
                peak_process_memory_bytes=None,
                baseline_process_memory_bytes=None,
            )
        )
        for algorithm in ("sqrt", "segment")
        for trial in (1, 2)
    ]
    for row in rows:
        row["query_mean_ns"] = "1.000e3"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = path.read_bytes()
    with path.open(newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    calls = []

    def profile(*configuration):
        # Even an in-place refresh may replace the source only after all profiles.
        assert path.read_bytes() == original
        calls.append(configuration)
        current = (20 + len(calls)) * 2**20
        return current, current + 2**20, 16 * 2**20

    def reject_timing(*args, **kwargs):
        pytest.fail("refreshing process RSS must not rerun timing trials")

    monkeypatch.setattr("benchmark_report.measure_process_memory", profile)
    monkeypatch.setattr("benchmark_report.run_benchmarks", reject_timing)
    monkeypatch.setattr("benchmark_report._load_pyplot", lambda _: None)
    monkeypatch.setattr("benchmark_report.save_plots", lambda *args: [])
    output = tmp_path if in_place else tmp_path / "refreshed"
    main(["--from-csv", str(path), "--refresh-memory", "--output-dir", str(output)])
    assert calls == [
        ("sqrt", 1000, 1520, "random", 100, "wide"),
        ("segment", 1000, 1520, "random", 100, "wide"),
    ]
    rss_fields = {
        "process_memory_bytes",
        "peak_process_memory_bytes",
        "baseline_process_memory_bytes",
    }
    with (output / "measurements.csv").open(newline="") as stream:
        refreshed_rows = list(csv.DictReader(stream))
    assert len(refreshed_rows) == 4
    for before, after in zip(original_rows, refreshed_rows, strict=True):
        assert {
            key: value for key, value in before.items() if key not in rss_fields
        } == {key: value for key, value in after.items() if key not in rss_fields}
        current = 21 * 2**20 if after["algorithm"] == "sqrt" else 22 * 2**20
        assert int(after["process_memory_bytes"]) == current
        assert int(after["peak_process_memory_bytes"]) == current + 2**20
        assert int(after["baseline_process_memory_bytes"]) == 16 * 2**20
    if not in_place:
        assert path.read_bytes() == original
    assert not list(output.glob(".measurements.csv.*.tmp"))


def test_refresh_memory_rejects_different_timing_environment(
    tmp_path, monkeypatch, capsys
):
    """RSS collected under another interpreter must not inherit old timing metadata."""
    path = tmp_path / "measurements.csv"
    row = asdict(measurement(python="different-python-version"))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    original = path.read_bytes()

    def reject_profile(*args):
        pytest.fail("environment checks must precede process memory profiling")

    monkeypatch.setattr("benchmark_report.measure_process_memory", reject_profile)
    monkeypatch.setattr("benchmark_report._load_pyplot", lambda _: None)
    with pytest.raises(SystemExit) as error:
        main(
            ["--from-csv", str(path), "--refresh-memory", "--output-dir", str(tmp_path)]
        )
    assert error.value.code == 2
    assert "saved timing trials" in capsys.readouterr().err
    assert path.read_bytes() == original


def test_refresh_memory_requires_saved_trials(capsys):
    """A missing source must fail before launching a new benchmark."""
    with pytest.raises(SystemExit) as error:
        main(["--refresh-memory"])
    assert error.value.code == 2
    assert "--refresh-memory requires --from-csv" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["profile", "replace"])
def test_failed_memory_refresh_preserves_source(tmp_path, monkeypatch, failure):
    """A failed probe or atomic replacement leaves an in-place source CSV intact."""
    path = tmp_path / "measurements.csv"
    row = asdict(
        measurement(python=platform.python_version(), platform=platform.platform())
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    original = path.read_bytes()

    def profile(*args):
        if failure == "profile":
            raise RuntimeError("memory worker failed")
        return 20 * 2**20, 24 * 2**20, 16 * 2**20

    def reject_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr("benchmark_report.measure_process_memory", profile)
    monkeypatch.setattr("benchmark_report._load_pyplot", lambda _: None)
    if failure == "replace":
        monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(SystemExit) as error:
        main(
            ["--from-csv", str(path), "--refresh-memory", "--output-dir", str(tmp_path)]
        )
    assert error.value.code == 2
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".measurements.csv.*.tmp"))


def test_plots_reject_comparisons_with_different_dataset_seeds(tmp_path):
    """The same input size alone does not make speedup comparisons equivalent."""
    rows = summarize([measurement(), measurement(algorithm="brute", seed=99)])
    with pytest.raises(ValueError, match="matching dataset seeds"):
        save_plots(rows, tmp_path, None)


@pytest.mark.parametrize("legacy_memory", [False, True])
def test_plots_support_algorithm_subset_without_brute(
    tmp_path, monkeypatch, legacy_memory
):
    """Subset/legacy replots remove stale managed plots and preserve other files."""
    if find_spec("matplotlib") is None:
        pytest.skip("plotting requires optional matplotlib dependency")
    from matplotlib import pyplot
    from matplotlib.figure import Figure

    observed = {}
    stale_names = (
        "build_time.png",
        "query_time_linear.png",
        "query_time_semilogx.png",
        "space_growth.png",
        "space_growth_linear.png",
        "space_per_element.png",
        "space_per_element_linear.png",
        "query_time_sparse_linear.png",
        "query_time_advanced.png",
        "query_time_advanced_linear.png",
        "memory_growth.png",
        "peak_memory_growth.png",
        "memory_per_element.png",
        "memory_per_element_linear.png",
        "query_speedup.png",
        "query_speedup_linear.png",
        "memory_growth_linear.png",
        "peak_memory_growth_linear.png",
    )
    preserved_names = (
        "unrelated.png",
        "update_time.png",
        "update_time_linear.png",
        "update_time_semilogx.png",
        "update_time_growth_axes.png",
        "update_measurements.csv",
        "update_summary.csv",
    )
    for name in (*stale_names, *preserved_names):
        (tmp_path / name).write_bytes(b"existing image")

    def record_axes(figure, path, *args, **kwargs):
        observed[path.name] = (
            [
                panel.get_title(loc="left")
                for panel in figure.axes
                if panel.get_visible()
            ]
            if path.name == "query_time_normalized.png"
            else [line.get_label() for line in figure.axes[0].lines]
        )
        path.write_bytes(b"new generated image")

    monkeypatch.setattr(Figure, "savefig", record_axes)
    memory = (
        {
            "process_memory_bytes": None,
            "peak_process_memory_bytes": None,
            "baseline_process_memory_bytes": None,
        }
        if legacy_memory
        else {}
    )
    rows = summarize(
        [measurement(**memory), measurement(algorithm="segment", **memory)]
    )
    paths = save_plots(rows, tmp_path, pyplot)
    assert {path.name for path in paths} == set(observed)
    expected_names = {
        "build_time_linear.png",
        "build_time_per_element.png",
        "query_time.png",
        "query_time_normalized.png",
    }
    if not legacy_memory:
        expected_names.update(
            {"memory_growth_linear.png", "peak_memory_growth_linear.png"}
        )
    assert set(observed) == expected_names
    assert all(
        labels == ["Square root decomposition", "Segment tree"]
        for labels in observed.values()
    )
    for name in stale_names:
        if name in observed:
            assert (tmp_path / name).read_bytes() == b"new generated image"
        else:
            assert not (tmp_path / name).exists()
    for name in preserved_names:
        assert (tmp_path / name).read_bytes() == b"existing image"


def test_build_graph_normalizes_linear_growth_and_iqr(tmp_path, monkeypatch):
    """A tenfold raw build increase remains flat after division by tenfold n."""
    if find_spec("matplotlib") is None:
        pytest.skip("plotting requires optional matplotlib dependency")
    from matplotlib import pyplot
    from matplotlib.figure import Figure

    rows = [
        measurement(n=n, trial=trial, build_ns=duration * factor)
        for n, factor in ((1000, 1), (10000, 10))
        for trial, duration in enumerate((1_000_000, 3_000_000, 9_000_000), start=1)
    ]
    observed = {}

    def record_build_axes(figure, path, *args, **kwargs):
        if not path.name.startswith("build_time"):
            return
        axis = figure.axes[0]
        observed[path.name] = (
            axis.get_xscale(),
            axis.get_yscale(),
            tuple(axis.lines[0].get_ydata()),
        )
        if path.name == "build_time_per_element.png":
            vertices = axis.collections[0].get_paths()[0].vertices
            for n in (1000, 10000):
                bounds = [y for x, y in vertices if x == n]
                assert min(bounds) == pytest.approx(2000)
                assert max(bounds) == pytest.approx(6000)

    monkeypatch.setattr(Figure, "savefig", record_build_axes)
    save_plots(summarize(rows), tmp_path, pyplot)
    assert observed == {
        "build_time_linear.png": ("linear", "linear", (3, 30)),
        "build_time_per_element.png": ("log", "linear", (3000, 3000)),
    }


def test_query_normalization_flattens_matching_growth_and_scales_iqr(
    tmp_path, monkeypatch
):
    """Each method's expected query growth becomes flat in its normalized panel."""
    if find_spec("matplotlib") is None:
        pytest.skip("plotting requires optional matplotlib dependency")
    from matplotlib import pyplot
    from matplotlib.figure import Figure

    # At n=16 and n=256, the four raw curves follow n, sqrt(n), log2(n), and 1.
    raw_medians = {
        "brute": (16000, 256000),
        "sqrt": (12000, 48000),
        "segment": (8000, 16000),
        "sparse": (500, 500),
    }
    rows = [
        measurement(
            algorithm=algorithm, n=n, trial=trial, query_mean_ns=duration * factor
        )
        for algorithm, durations in raw_medians.items()
        for n, duration in zip((16, 256), durations, strict=True)
        for trial, factor in enumerate((0.5, 1, 1.5), start=1)
    ]
    observed = {}

    def record_normalized_axes(figure, path, *args, **kwargs):
        if path.name != "query_time_normalized.png":
            return
        assert len(figure.texts) == 1
        visible = [axis for axis in figure.axes if axis.get_visible()]
        assert len(visible) == 4
        for axis in visible:
            assert not axis.texts
            assert (axis.get_xscale(), axis.get_yscale()) == ("log", "linear")
            assert axis.get_ylim()[0] == 0
            assert len(axis.lines) == len(axis.collections) == 1
            assert "ns" in axis.get_ylabel()
            curve = axis.lines[0]
            assert tuple(curve.get_xdata()) == (16, 256)
            vertices = axis.collections[0].get_paths()[0].vertices
            bounds = []
            for n in (16, 256):
                values = [y for x, y in vertices if x == n]
                bounds.append((min(values), max(values)))
            observed[axis.get_title(loc="left")] = (tuple(curve.get_ydata()), bounds)

    monkeypatch.setattr(Figure, "savefig", record_normalized_axes)
    paths = save_plots(summarize(rows), tmp_path, pyplot)
    assert tmp_path / "query_time_normalized.png" in paths
    expected = {
        "Naive brute force": 1000,
        "Square root decomposition": 3000,
        "Segment tree": 2000,
        "Sparse table": 500,
    }
    assert list(observed) == list(expected)
    for title, value in expected.items():
        curve, bounds = observed[title]
        assert curve == pytest.approx((value, value))
        for lower, upper in bounds:
            assert (lower, upper) == pytest.approx((value * 0.75, value * 1.25))


@pytest.mark.parametrize(
    "algorithms,expected_titles",
    [
        (("segment",), []),
        (
            ("brute", "sqrt", "segment", "sparse"),
            ["Naive brute force", "Square root decomposition", "Sparse table"],
        ),
    ],
)
def test_query_normalization_handles_singleton_segment_and_odd_panels(
    tmp_path, monkeypatch, algorithms, expected_titles
):
    """log2(1) is omitted safely, and an odd number of useful panels hides extras."""
    if find_spec("matplotlib") is None:
        pytest.skip("plotting requires optional matplotlib dependency")
    from matplotlib import pyplot
    from matplotlib.figure import Figure

    path = tmp_path / "query_time_normalized.png"
    path.write_bytes(b"old normalized plot")
    observed_titles = []

    def record_normalized_axes(figure, saved_path, *args, **kwargs):
        if saved_path.name != "query_time_normalized.png":
            return
        visible = [axis for axis in figure.axes if axis.get_visible()]
        observed_titles.extend(axis.get_title(loc="left") for axis in visible)
        assert len(figure.axes) == 4
        assert len(visible) == 3
        for axis in visible:
            assert tuple(axis.lines[0].get_ydata()) == (1000,)
        saved_path.write_bytes(b"new normalized plot")

    monkeypatch.setattr(Figure, "savefig", record_normalized_axes)
    rows = summarize([measurement(algorithm=name, n=1) for name in algorithms])
    paths = save_plots(rows, tmp_path, pyplot)
    assert observed_titles == expected_titles
    if expected_titles:
        assert path in paths
        assert path.read_bytes() == b"new normalized plot"
    else:
        assert path not in paths
        assert not path.exists()
