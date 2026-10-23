"""Shared input contract; no range query algorithms are implemented here."""

from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable
from operator import index

MAX_INT64 = (1 << 63) - 1


def pack_values(values: Iterable[int]) -> array:
    """Copy integer input into a nonempty signed 64-bit buffer and return it.

    Raise ValueError for empty input, TypeError for noninteger elements, and
    OverflowError for elements outside [-2**63, 2**63 - 1].
    """
    packed = array("q", values)
    if not packed:
        raise ValueError("at least one value is required")
    return packed


class RangeMinimumQuery(ABC):
    """Common interface for minima of nonempty, half-open ranges [start, stop)."""

    __slots__ = ("_length",)

    def __len__(self) -> int:
        """Return the number of input values in O(1) time; take no arguments."""
        return self._length

    def _check_range(self, start: int, stop: int) -> tuple[int, int]:
        """Return validated integer endpoints; reject bounds and empty ranges."""
        start, stop = index(start), index(stop)
        if not 0 <= start <= self._length or not 0 <= stop <= self._length:
            raise IndexError("range endpoints must be between 0 and len(structure)")
        if start >= stop:
            raise ValueError("range must be nonempty: start < stop")
        return start, stop

    def _check_index(self, position: int) -> int:
        """Return a valid integer element index; raise IndexError out of bounds."""
        position = index(position)
        if not 0 <= position < self._length:
            raise IndexError("position must be between 0 and len(structure) - 1")
        return position

    @abstractmethod
    def query(self, start: int, stop: int) -> int:
        """Return min(values[start:stop]) for a valid nonempty half-open range."""

    @property
    @abstractmethod
    def storage_bytes(self) -> int:
        """Return owned array payload bytes, excluding Python object overhead."""
