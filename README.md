# COSC 520 Assignment 2: range minimum queries

Python implementations of brute force, square root decomposition, a segment
tree, and a sparse table for range minimum queries. Queries return the minimum
over a zero-based, half-open range `[start, stop)`. The first three structures
also support point assignments; the sparse table is static.

## Project layout

```text
.
├── README.md
├── pyproject.toml
├── flake.nix
├── flake.lock
├── .envrc
├── demo.py                  # Examples of queries and updates
├── benchmark.py             # Custom benchmarks
├── run_all_benchmark.py     # Full benchmark preset
├── src/
│   ├── _base.py
│   ├── brute_force.py
│   ├── sqrt_decomposition.py
│   ├── segment_tree.py
│   ├── sparse_table.py
│   ├── benchmark_core.py
│   ├── benchmark_report.py
│   └── memory_probe.py
├── tests/                   # Unit tests
├── docs/
│   ├── BENCHMARKING.md
│   └── UPDATE_COMPARISON.md
└── results/                 # Benchmark outputs
```

## Setup

Run these commands from the project root. Choose either setup below.

### Nix

With Nix with flakes enabled:

```sh
nix develop
```

This provides Python, dependencies, and the import path. You can also run the
tests directly with `nix run path:.#test`.

### Python virtual environment

Use Python 3.14. In a macOS,  Linux shell, or Windows WSL2 terminal:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Dataset

No dataset download is required. Benchmarks generate a range of reproducible inputs in
memory as packed signed 64-bit integers. 
## Tests and demo

After either setup:

```sh
python -m pytest -q
python demo.py
```

## Benchmark

Run the full preset:

```sh
python run_all_benchmark.py
```

It runs runtime benchmark and memory measurements, writing the csv and plots to `results/`.
Your can also run a smaller custom benchmark:

```sh
python benchmark.py --sizes 1000 10000 --queries 100 --updates 1000 \
  --static-repeats 3 --update-repeats 3 --output-dir results
```

For more details, run `python benchmark.py --help`  and `python run_all_benchmark.py --help`.

you can also edit the `PRESET_SIZES` in `run_all_benchmark.py` to change the sizes used in the full benchmark.
