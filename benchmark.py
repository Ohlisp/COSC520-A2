"""Run customized static and update benchmarks using explicit input sizes."""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark_core import _positive_int  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Run both experiments sequentially, stopping if either experiment fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_positive_int,
        nargs="+",
        required=True,
        help="input sizes for this custom run; use run_all_benchmark.py for the preset",
    )
    parser.add_argument("--queries", type=_positive_int, default=1000)
    parser.add_argument("--updates", type=_positive_int, default=10000)
    parser.add_argument("--static-repeats", type=_positive_int, default=5)
    parser.add_argument("--update-repeats", type=_positive_int, default=9)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    common = [
        "--sizes",
        *(str(size) for size in sorted(set(args.sizes))),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
    ]
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    previous_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + previous_path if previous_path else source_path
    )

    print("Running construction, query and memory benchmarks...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark_report",
            *common,
            "--queries",
            str(args.queries),
            "--repeats",
            str(args.static_repeats),
        ],
        check=True,
        env=environment,
    )
    print("Running point-update benchmarks...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark_core",
            "updates",
            *common,
            "--updates",
            str(args.updates),
            "--repeats",
            str(args.update_repeats),
        ],
        check=True,
        env=environment,
    )
    print(f"All benchmark CSVs and plots saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
