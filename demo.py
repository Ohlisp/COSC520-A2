"""Show each range-minimum structure's public API with a small example.

Run with ``python demo.py`` from the repository root; no installation is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from brute_force import BruteForce  # noqa: E402
from segment_tree import SegmentTree  # noqa: E402
from sparse_table import SparseTable  # noqa: E402
from sqrt_decomposition import SqrtDecomposition  # noqa: E402

VALUES = [8, 3, 7, -2, 5, 1, 9]


def show_query(structure, values: list[int], start: int, stop: int) -> None:
    """Print a method call, the values covered, and its returned minimum."""
    result = structure.query(start, stop)
    print(f"  query({start}, {stop}) -> {result}; covered values: {values[start:stop]}")


def demonstrate(structure_type) -> None:
    """Construct one structure and show its queries and supported changes."""
    values = VALUES.copy()
    structure = structure_type(values)
    print(f"\n{structure_type.__name__}")
    print(f"  Construct: {structure_type.__name__}({values})")
    print(f"  len(structure) -> {len(structure)}")
    print(f"  storage_bytes -> {structure.storage_bytes} (packed buffers only)")
    show_query(structure, values, 0, len(values))
    show_query(structure, values, 1, 5)
    show_query(structure, values, 2, 3)

    if isinstance(structure, SparseTable):
        print("  SparseTable has no update() method; rebuild after changing values.")
        values[3] = 6
        print(f"  Change the source list: values[3] = 6 -> {values}")
        print(f"  Original table: query(0, 7) -> {structure.query(0, 7)}")
        rebuilt = SparseTable(values)
        print("  Rebuild: rebuilt = SparseTable(values)")
        print(f"  Rebuilt table: query(0, 7) -> {rebuilt.query(0, 7)}")
        print("  The original table keeps its own copy of the earlier values.")
        return

    print("  Raise the old minimum from -2 to 6:")
    result = structure.update(3, 6)
    values[3] = 6
    print(f"  update(3, 6) -> {result}; current values: {values}")
    show_query(structure, values, 0, len(values))
    show_query(structure, values, 1, 5)

    print("  Assign a new minimum at position 0:")
    result = structure.update(0, -4)
    values[0] = -4
    print(f"  update(0, -4) -> {result}; current values: {values}")
    show_query(structure, values, 0, len(values))
    show_query(structure, values, 0, 1)


def main() -> None:
    print("Range minimum query demo")
    print("Indices start at 0. query(start, stop) includes start and excludes stop.")
    print("update(position, value) replaces one value and returns None.")
    for structure_type in (BruteForce, SqrtDecomposition, SegmentTree, SparseTable):
        demonstrate(structure_type)


if __name__ == "__main__":
    main()
