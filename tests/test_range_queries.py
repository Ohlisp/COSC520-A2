"""Differential tests against Python's direct minimum, independent of internals."""

from array import array
from random import Random

import pytest

from brute_force import BruteForce
from segment_tree import SegmentTree
from sparse_table import SparseTable
from sqrt_decomposition import SqrtDecomposition

STRUCTURES = (BruteForce, SqrtDecomposition, SegmentTree, SparseTable)
MUTABLE = (BruteForce, SqrtDecomposition, SegmentTree)
MIN_INT64, MAX_INT64 = -(1 << 63), (1 << 63) - 1


@pytest.mark.parametrize("structure_type", STRUCTURES)
@pytest.mark.parametrize(
    "values",
    [
        [42],
        [3, -1],
        [5, 5, 5, 5, 5],
        [8, 3, 7, -2, 5, 1, 9],
        list(range(17)),
        list(range(17, 0, -1)),
        [MAX_INT64, MIN_INT64, 0, MAX_INT64],
        [MAX_INT64] * 9,
    ],
)
def test_every_range(structure_type, values):
    """Check singleton, full, aligned, partial, negative and repeated-value ranges."""
    structure = structure_type(values)
    assert len(structure) == len(values)
    for start in range(len(values)):
        for stop in range(start + 1, len(values) + 1):
            assert structure.query(start, stop) == min(values[start:stop])


@pytest.mark.parametrize("structure_type", STRUCTURES)
def test_random_lengths_and_ranges(structure_type):
    """Exhaust small lengths and sample around square/power-of-two boundaries."""
    rng = Random(520)
    for length in [*range(1, 35), 63, 64, 65, 80, 81, 82, 127, 128, 129, 257]:
        values = [rng.randint(-1000, 1000) for _ in range(length)]
        structure = structure_type(values)
        for _ in range(150):
            start = rng.randrange(length)
            stop = rng.randrange(start + 1, length + 1)
            assert structure.query(start, stop) == min(values[start:stop])


@pytest.mark.parametrize("structure_type", STRUCTURES)
def test_input_is_copied_and_iterators_are_accepted(structure_type):
    """External list/array mutations must not change stored data or summaries."""
    for values in ([8, -4, 7], array("q", [8, -4, 7])):
        structure = structure_type(values)
        values[1] = 99
        assert structure.query(0, 3) == -4
        assert structure.query(1, 2) == -4
    assert structure_type(iter([3, 1, 2])).query(0, 3) == 1


@pytest.mark.parametrize("structure_type", STRUCTURES)
@pytest.mark.parametrize(
    ("values", "error"),
    [
        ([], ValueError),
        ([1.5], TypeError),
        (["1"], TypeError),
        ([1 << 63], OverflowError),
        ([-(1 << 63) - 1], OverflowError),
    ],
)
def test_invalid_input(structure_type, values, error):
    """Check the documented nonempty signed-integer input contract."""
    with pytest.raises(error):
        structure_type(values)


@pytest.mark.parametrize("structure_type", STRUCTURES)
@pytest.mark.parametrize(
    ("start", "stop", "error"),
    [
        (-1, 2, IndexError),
        (0, -1, IndexError),
        (0, 4, IndexError),
        (4, 5, IndexError),
        (0, 0, ValueError),
        (3, 3, ValueError),
        (2, 1, ValueError),
        (0.0, 2, TypeError),
        (0, "2", TypeError),
    ],
)
def test_invalid_ranges(structure_type, start, stop, error):
    """Reject invalid ranges consistently rather than accepting Python slicing."""
    with pytest.raises(error):
        structure_type([3, 1, 2]).query(start, stop)


@pytest.mark.parametrize("structure_type", MUTABLE)
def test_updates_against_list(structure_type):
    """Check mixed queries/assignments, including removing an existing minimum."""
    rng = Random(2026)
    for length in (1, 2, 7, 16, 17, 31, 65):
        values = [rng.randint(-100, 100) for _ in range(length)]
        structure = structure_type(values)
        for _ in range(120):
            position = rng.randrange(length)
            value = rng.choice([MIN_INT64, MAX_INT64, rng.randint(-100, 100)])
            assert structure.update(position, value) is None
            values[position] = value
            assert structure.query(position, position + 1) == value
            assert structure.query(0, length) == min(values)
            start = rng.randrange(length)
            stop = rng.randrange(start + 1, length + 1)
            assert structure.query(start, stop) == min(values[start:stop])


@pytest.mark.parametrize("structure_type", MUTABLE)
def test_removing_minimum_and_updating_last_partial_block(structure_type):
    """Increasing the minimum and updating either endpoint must repair summaries."""
    values = [-9, 2, 3, 4, 5, 6, -7]
    structure = structure_type(values)
    for position, value in [(0, 10), (6, 12), (6, -20), (6, -20), (0, -30)]:
        structure.update(position, value)
        values[position] = value
        for start in range(len(values)):
            for stop in range(start + 1, len(values) + 1):
                assert structure.query(start, stop) == min(values[start:stop])


@pytest.mark.parametrize("structure_type", MUTABLE)
@pytest.mark.parametrize(
    ("position", "value", "error"),
    [
        (-1, 0, IndexError),
        (3, 0, IndexError),
        (0.5, 0, TypeError),
        (1, 1.5, TypeError),
        (1, 1 << 63, OverflowError),
        (1, -(1 << 63) - 1, OverflowError),
    ],
)
def test_invalid_updates_leave_structure_unchanged(
    structure_type, position, value, error
):
    """Failed updates must preserve original values and aggregate answers."""
    structure = structure_type([3, -1, 2])
    with pytest.raises(error):
        structure.update(position, value)
    assert [structure.query(i, i + 1) for i in range(3)] == [3, -1, 2]
    assert structure.query(0, 3) == -1


@pytest.mark.parametrize("structure_type", STRUCTURES)
def test_storage_is_compact(structure_type):
    """Packed storage must remain within the documented payload bounds."""
    length = 1000
    structure = structure_type(range(length))
    if structure_type is BruteForce:
        assert structure.storage_bytes == array("q").itemsize * length
    elif structure_type is SqrtDecomposition:
        assert 8 * length < structure.storage_bytes < 9 * length
    elif structure_type is SegmentTree:
        assert 16 * length <= structure.storage_bytes < 32 * length
    else:
        assert structure.storage_bytes == 8 * sum(
            length - (1 << k) + 1 for k in range(length.bit_length())
        )
