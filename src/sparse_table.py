"""Static range minima using overlapping power-of-two intervals."""

from array import array
from collections.abc import Iterable

from _base import RangeMinimumQuery, pack_values


class SparseTable(RangeMinimumQuery):
    """O(n log n) build/space and O(1) queries; rebuild to change values."""

    __slots__ = ("_table",)

    def __init__(self, values: Iterable[int]) -> None:
        """Build minimum for every valid interval of each power-of-two length."""
        packed = pack_values(values)
        self._length = len(packed)
        self._table = [packed]
        for level in range(1, self._length.bit_length()):
            half = 1 << (level - 1)
            previous = self._table[-1]
            row = array(
                "q",
                (
                    min(previous[start], previous[start + half])
                    for start in range(self._length - 2 * half + 1)
                ),
            )
            self._table.append(row)

    def query(self, start: int, stop: int) -> int:
        """Combine two possibly overlapping intervals covering [start, stop)."""
        start, stop = self._check_range(start, stop)
        level = (stop - start).bit_length() - 1
        width = 1 << level
        row = self._table[level]
        return min(row[start], row[stop - width])

    @property
    def storage_bytes(self) -> int:
        """Return packed level payload bytes, excluding Python object overhead."""
        return sum(len(row) * row.itemsize for row in self._table)
