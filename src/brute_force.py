"""Naive range minima by scanning every value in the requested interval."""

from collections.abc import Iterable

from _base import RangeMinimumQuery, pack_values


class BruteForce(RangeMinimumQuery):
    """O(n) build/space, O(stop - start) queries and O(1) point assignments.

    Queries use constant auxiliary space: they do not copy the requested range
    or retain precomputed summaries.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Iterable[int]) -> None:
        """Copy nonempty int64 input without building any range summaries."""
        self._values = pack_values(values)
        self._length = len(self._values)

    def query(self, start: int, stop: int) -> int:
        """Return a range minimum by directly comparing its individual values."""
        start, stop = self._check_range(start, stop)
        result = self._values[start]
        for position in range(start + 1, stop):
            value = self._values[position]
            if value < result:
                result = value
        return result

    def update(self, position: int, value: int) -> None:
        """Assign an int64 value at a validated position; return None."""
        position = self._check_index(position)
        self._values[position] = value

    @property
    def storage_bytes(self) -> int:
        """Return the copied values' array payload size in bytes."""
        return len(self._values) * self._values.itemsize
