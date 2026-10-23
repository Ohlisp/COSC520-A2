"""Range minima using square root decomposition and explicit block summaries."""

from array import array
from collections.abc import Iterable
from math import isqrt

from _base import MAX_INT64, RangeMinimumQuery, pack_values


class SqrtDecomposition(RangeMinimumQuery):
    """O(n) build/space, O(sqrt(n)) queries and point assignments."""

    __slots__ = ("_values", "_blocks", "_block_size")

    def __init__(self, values: Iterable[int]) -> None:
        """Copy nonempty int64 input and build one minimum per ceil(sqrt(n)) block."""
        self._values = pack_values(values)
        self._length = len(self._values)
        self._block_size = isqrt(self._length - 1) + 1
        count = (self._length + self._block_size - 1) // self._block_size
        self._blocks = array("q", [MAX_INT64]) * count
        # store the minimum of each block in _blocks
        for position, value in enumerate(self._values):
            block = position // self._block_size
            if value < self._blocks[block]:
                self._blocks[block] = value

    def query(self, start: int, stop: int) -> int:
        """Return a range minimum by scanning two tails and complete block minima."""
        start, stop = self._check_range(start, stop)
        size = self._block_size
        result = MAX_INT64
        # The first tail ends at a block boundary or at the query's stop.
        tail_stop = min(stop, ((start + size - 1) // size) * size)
        for position in range(start, tail_stop):
            value = self._values[position]
            if value < result:
                result = value
        start = tail_stop
        while start + size <= stop:
            value = self._blocks[start // size]
            if value < result:
                result = value
            start += size
        for position in range(start, stop):
            value = self._values[position]
            if value < result:
                result = value
        return result

    def update(self, position: int, value: int) -> None:
        """Assign an int64 value at position and rescan its block; return None.

        Rescanning is necessary when an assignment removes the previous minimum.
        Invalid positions or values raise before any summaries are changed.
        """
        position = self._check_index(position)
        self._values[position] = value
        block = position // self._block_size
        start = block * self._block_size
        stop = min(start + self._block_size, self._length)
        result = MAX_INT64
        for offset in range(start, stop):
            candidate = self._values[offset]
            if candidate < result:
                result = candidate
        self._blocks[block] = result

    @property
    def storage_bytes(self) -> int:
        """Return the combined values/block array payload size in bytes."""
        return (len(self._values) + len(self._blocks)) * self._values.itemsize
