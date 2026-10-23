"""Verify the public benchmark command orchestrates both comparisons safely."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def entrypoint(monkeypatch):
    """Load the root script without relying on the working directory."""
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location(
        "public_benchmark_entrypoint", PROJECT_ROOT / "benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preset(monkeypatch):
    """Load the fixed experiment launcher with its sibling module available."""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(
        "preset_benchmark_entrypoint", PROJECT_ROOT / "run_all_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option_values(command, option):
    """Read argument values without depending on the order of named options."""
    values = []
    for argument in command[command.index(option) + 1 :]:
        if argument.startswith("--"):
            break
        values.append(argument)
    return values


def test_entrypoint_forwards_settings_and_local_source(
    entrypoint, monkeypatch, tmp_path
):
    """Both modules receive the requested workload and can import this checkout."""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing/source")
    monkeypatch.setenv("BENCHMARK_ENTRYPOINT_MARKER", "preserved")
    output_dir = tmp_path / "exported"
    entrypoint.main(
        [
            "--sizes",
            "11",
            "23",
            "--queries",
            "7",
            "--updates",
            "8",
            "--static-repeats",
            "2",
            "--update-repeats",
            "3",
            "--seed",
            "77",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert [command[:3] for command, _ in calls] == [
        [sys.executable, "-m", "benchmark_report"],
        [sys.executable, "-m", "benchmark_core"],
    ]
    for command, kwargs in calls:
        assert kwargs["check"] is True
        assert _option_values(command, "--sizes") == ["11", "23"]
        assert _option_values(command, "--seed") == ["77"]
        assert _option_values(command, "--output-dir") == [str(output_dir)]
        environment = kwargs["env"]
        assert environment["PYTHONPATH"].split(os.pathsep) == [
            str(PROJECT_ROOT / "src"),
            "/existing/source",
        ]
        assert environment["BENCHMARK_ENTRYPOINT_MARKER"] == "preserved"
    static_command, update_command = (command for command, _ in calls)
    assert update_command[3] == "updates"
    assert _option_values(static_command, "--queries") == ["7"]
    assert _option_values(static_command, "--repeats") == ["2"]
    assert _option_values(update_command, "--updates") == ["8"]
    assert _option_values(update_command, "--repeats") == ["3"]
    assert os.environ["PYTHONPATH"] == "/existing/source"


def test_chosen_sizes_use_optional_workload_defaults(entrypoint, monkeypatch):
    """Custom sizes can use the documented counts, repeats and output directory."""
    commands = []
    environments = []

    def run(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    entrypoint.main(["--sizes", "11", "23"])
    assert len(commands) == 2
    for command in commands:
        assert _option_values(command, "--sizes") == ["11", "23"]
        assert _option_values(command, "--seed") == ["520"]
        assert _option_values(command, "--output-dir") == ["results"]
    assert _option_values(commands[0], "--queries") == ["1000"]
    assert _option_values(commands[0], "--repeats") == ["5"]
    assert _option_values(commands[1], "--updates") == ["10000"]
    assert _option_values(commands[1], "--repeats") == ["9"]
    assert all(
        environment["PYTHONPATH"] == str(PROJECT_ROOT / "src")
        for environment in environments
    )


def test_custom_benchmark_requires_explicit_sizes(entrypoint, monkeypatch):
    """Omitting sizes must not accidentally start the full experiment."""
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: commands.append(args)
    )
    with pytest.raises(SystemExit) as error:
        entrypoint.main([])
    assert error.value.code == 2
    assert commands == []


def test_static_failure_prevents_update_run(entrypoint, monkeypatch):
    """A failed first comparison must propagate instead of starting the second."""
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        entrypoint.main(["--sizes", "5"])
    assert len(commands) == 1
    assert commands[0][:3] == [
        sys.executable,
        "-m",
        "benchmark_report",
    ]


@pytest.mark.parametrize(
    "option",
    ["--sizes", "--queries", "--updates", "--static-repeats", "--update-repeats"],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_counts_never_launch_children(entrypoint, monkeypatch, option, value):
    """Reject invalid shared and per-comparison sizes before starting any work."""
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: commands.append(args)
    )
    arguments = [] if option == "--sizes" else ["--sizes", "5"]
    with pytest.raises(SystemExit) as error:
        entrypoint.main([*arguments, option, value])
    assert error.value.code == 2
    assert commands == []


def test_preset_forwards_complete_configuration_to_both_comparisons(
    preset, entrypoint, monkeypatch
):
    """The no-argument preset forwards its configured sizes and workload."""
    forwarded = []
    commands = []

    def benchmark_main(arguments):
        forwarded.append(arguments)
        entrypoint.main(arguments)

    monkeypatch.setattr(preset, "benchmark_main", benchmark_main)
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    preset.main([])
    assert len(forwarded) == 1
    arguments = forwarded[0]
    sizes = [str(size) for size in preset.PRESET_SIZES]
    expected_options = {
        "--sizes": sizes,
        "--queries": ["1000"],
        "--updates": ["10000"],
        "--static-repeats": ["5"],
        "--update-repeats": ["9"],
        "--seed": ["520"],
        "--output-dir": ["results"],
    }
    assert {argument for argument in arguments if argument.startswith("--")} == set(
        expected_options
    )
    for option, values in expected_options.items():
        assert _option_values(arguments, option) == values
    assert [command[:3] for command in commands] == [
        [sys.executable, "-m", "benchmark_report"],
        [sys.executable, "-m", "benchmark_core"],
    ]
    assert commands[1][3] == "updates"
    assert all(_option_values(command, "--sizes") == sizes for command in commands)


def test_preset_help_does_not_start_benchmarks(preset, monkeypatch, capsys):
    """Asking about the preset must not launch its expensive experiment."""
    calls = []
    monkeypatch.setattr(
        preset, "benchmark_main", lambda arguments: calls.append(arguments)
    )
    with pytest.raises(SystemExit) as error:
        preset.main(["--help"])
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert calls == []


@pytest.mark.parametrize(
    "arguments",
    [["--sizes", "5"], ["--queries", "2"], ["--output-dir", "elsewhere"]],
)
def test_preset_rejects_custom_arguments(preset, monkeypatch, arguments):
    """Custom experiment options belong to benchmark.py, not the fixed preset."""
    calls = []
    monkeypatch.setattr(
        preset, "benchmark_main", lambda arguments: calls.append(arguments)
    )
    with pytest.raises(SystemExit) as error:
        preset.main(arguments)
    assert error.value.code == 2
    assert calls == []
