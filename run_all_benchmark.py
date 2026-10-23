"""Run the preset static and update benchmarks through ten million values."""

import argparse
from collections.abc import Sequence

from benchmark import main as benchmark_main

PRESET_SIZES = (
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Pass the complete preset explicitly to the customizable benchmark runner."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Preset: sizes from 1,000 through 1,000,000; 1,000 queries; "
            "Preset is different from the pdf report benchmark as 10 million values with trials take a long time to run and large amounts of memory."
            "10,000 updates; five static trials; nine update trials; seed 520; "
            "output in results/. Use benchmark.py --sizes ... for custom runs."
        ),
    )
    parser.parse_args(argv)
    benchmark_main(
        [
            "--sizes",
            *(str(size) for size in PRESET_SIZES),
            "--queries",
            "1000",
            "--updates",
            "10000",
            "--static-repeats",
            "5",
            "--update-repeats",
            "9",
            "--seed",
            "520",
            "--output-dir",
            "results",
        ]
    )


if __name__ == "__main__":
    main()
