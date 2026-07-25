"""
Correctness tests for true Karmarkar-Karp largest-differencing partitioning.

Covers the Rust extension's `karmarkar_karp` (index-based, `equal_size`
support) and `get_seqlen_balanced_partitions` (validation + sorted indices).

Run with:
    uv run pytest test_differencing.py -v
(after `uv run maturin develop`)
"""

from __future__ import annotations

import heapq
import random

import pytest


# ---------------------------------------------------------------------------
# Pure-Python reference implementation (greedy LPT, for spread comparison)
# ---------------------------------------------------------------------------


def lpt_spread(seqlen_list: list[int], k: int) -> int:
    """Greedy longest-processing-time spread, as an upper bound to compare against."""
    sums = [0] * k
    heap = [(0, i) for i in range(k)]
    heapq.heapify(heap)
    for length in sorted(seqlen_list, reverse=True):
        s, idx = heapq.heappop(heap)
        sums[idx] = s + length
        heapq.heappush(heap, (sums[idx], idx))
    return max(sums) - min(sums)


def partition_spread(seqlen_list: list[int], partitions: list[list[int]]) -> int:
    sums = [sum(seqlen_list[i] for i in p) for p in partitions]
    return max(sums) - min(sums)


def assert_full_coverage(seqlen_list: list[int], partitions: list[list[int]]) -> None:
    seen: set[int] = set()
    for p in partitions:
        for idx in p:
            assert idx not in seen, f"index {idx} assigned twice"
            seen.add(idx)
    assert seen == set(range(len(seqlen_list))), "not every index was assigned"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def kk():
    try:
        from seq_len_balance import karmarkar_karp  # noqa: PLC0415

        return karmarkar_karp
    except ImportError:
        pytest.skip(
            "seq_len_balance extension not found — run `uv run maturin develop` first"
        )


@pytest.fixture(scope="session")
def balanced_partitions():
    try:
        from seq_len_balance import get_seqlen_balanced_partitions  # noqa: PLC0415

        return get_seqlen_balanced_partitions
    except ImportError:
        pytest.skip(
            "seq_len_balance extension not found — run `uv run maturin develop` first"
        )


# ---------------------------------------------------------------------------
# karmarkar_karp
# ---------------------------------------------------------------------------


class TestKarmarkarKarp:
    def test_tiny_example(self, kk):
        items = [8, 7, 6, 5, 4]
        partitions = kk(items, 3)
        assert len(partitions) == 3
        assert_full_coverage(items, partitions)

    def test_single_bin(self, kk):
        items = [3, 1, 4, 1, 5]
        partitions = kk(items, 1)
        assert len(partitions) == 1
        assert_full_coverage(items, partitions)

    def test_k_equals_n(self, kk):
        items = list(range(1, 9))
        partitions = kk(items, len(items))
        assert_full_coverage(items, partitions)
        for p in partitions:
            assert len(p) == 1

    def test_uniform_weights_perfect_balance(self, kk):
        items = [5] * 100
        partitions = kk(items, 4)
        assert_full_coverage(items, partitions)
        assert partition_spread(items, partitions) == 0

    def test_no_empty_partition(self, kk):
        items = [100, 1, 1, 1, 1, 1]
        partitions = kk(items, 3)
        assert len(partitions) == 3
        for p in partitions:
            assert len(p) > 0

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_spread_at_most_lpt(self, kk, k: int):
        """True KK differencing should be at least as good as greedy LPT."""
        rng = random.Random(2024)
        items = [rng.randint(1, 1000) for _ in range(500)]
        partitions = kk(items, k)
        assert_full_coverage(items, partitions)
        kk_spread = partition_spread(items, partitions)
        assert kk_spread <= lpt_spread(items, k)

    def test_equal_size_splits_evenly(self, kk):
        items = [rng for rng in range(1, 13)]  # 12 items
        partitions = kk(items, 4, True)
        assert len(partitions) == 4
        assert_full_coverage(items, partitions)
        for p in partitions:
            assert len(p) == 3

    def test_equal_size_rejects_non_divisible(self, kk):
        items = list(range(1, 11))  # 10 items, k=3 doesn't divide
        with pytest.raises(ValueError):
            kk(items, 3, True)

    def test_k_greater_than_n_raises(self, kk):
        with pytest.raises(ValueError):
            kk([1, 2], 5, True)

    @pytest.mark.parametrize("n", [1_000, 10_000])
    def test_large_random_lossless(self, kk, n: int):
        rng = random.Random(7)
        items = [rng.randint(1, 10_000) for _ in range(n)]
        partitions = kk(items, 8)
        assert_full_coverage(items, partitions)


# ---------------------------------------------------------------------------
# get_seqlen_balanced_partitions
# ---------------------------------------------------------------------------


class TestGetSeqlenBalancedPartitions:
    def test_returns_sorted_indices(self, balanced_partitions):
        items = [8, 7, 6, 5, 4, 3, 2, 1]
        partitions = balanced_partitions(items, 3)
        assert len(partitions) == 3
        assert_full_coverage(items, partitions)
        for p in partitions:
            assert p == sorted(p)

    def test_all_partitions_non_empty(self, balanced_partitions):
        items = [1, 2, 3, 4, 5]
        partitions = balanced_partitions(items, 5)
        assert len(partitions) == 5
        for p in partitions:
            assert len(p) > 0

    def test_equal_size(self, balanced_partitions):
        items = list(range(1, 13))
        partitions = balanced_partitions(items, 4, True)
        assert len(partitions) == 4
        assert_full_coverage(items, partitions)
        for p in partitions:
            assert len(p) == 3

    def test_raises_when_fewer_items_than_partitions(self, balanced_partitions):
        with pytest.raises(ValueError):
            balanced_partitions([1, 2], 5)

    def test_raises_on_non_divisible_equal_size(self, balanced_partitions):
        with pytest.raises(ValueError):
            balanced_partitions(list(range(1, 11)), 3, True)
