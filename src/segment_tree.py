"""An iterative binary segment tree for range minima and point assignments."""

from array import array
from collections.abc import Iterable

from _base import MAX_INT64, RangeMinimumQuery, pack_values


class SegmentTree(RangeMinimumQuery):
    """O(n) build/space, O(log(n)) queries and point assignments."""

    __slots__ = ("_tree", "_leaf_count")

    def __init__(self, values: Iterable[int]) -> None:
        """Copy nonempty int64 input into leaves and build parent minima bottom-up."""
        packed = pack_values(values)
        self._length = len(packed)
        self._leaf_count = 1 << (self._length - 1).bit_length()
        self._tree = array("q", [MAX_INT64]) * (2 * self._leaf_count)
        self._tree[self._leaf_count : self._leaf_count + self._length] = packed
        for node in range(self._leaf_count - 1, 0, -1):
            left = self._tree[2 * node]
            right = self._tree[2 * node + 1]
            self._tree[node] = left if left < right else right

    def query(self, start: int, stop: int) -> int:
        """Return a range minimum by merging disjoint nodes while climbing the tree."""
        start, stop = self._check_range(start, stop)
        start += self._leaf_count
        stop += self._leaf_count
        result = MAX_INT64
        while start < stop:
            if start & 1:
                value = self._tree[start]
                if value < result:
                    result = value
                start += 1
            if stop & 1:
                stop -= 1
                value = self._tree[stop]
                if value < result:
                    result = value
            start //= 2
            stop //= 2
        return result

    def update(self, position: int, value: int) -> None:
        """Assign an int64 leaf and recompute its ancestors; return None."""
        node = self._check_index(position) + self._leaf_count
        self._tree[node] = value
        node //= 2
        while node:
            left = self._tree[2 * node]
            right = self._tree[2 * node + 1]
            self._tree[node] = left if left < right else right
            node //= 2

    @property
    def storage_bytes(self) -> int:
        """Return the tree array payload size, including padding, in bytes."""
        return len(self._tree) * self._tree.itemsize
