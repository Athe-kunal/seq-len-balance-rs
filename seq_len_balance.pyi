"""Type hints for the compiled ``seq_len_balance`` extension module."""

from typing import Sequence


def karmarkar_karp_partition(items: Sequence[float], k: int) -> list[list[float]]:
    """Partition items into ``k`` bins using the Karmarkar-Karp heuristic."""


def first_fit_decreasing_pack(items: Sequence[float], k: float) -> list[list[float]]:
    """Pack items into bins of capacity ``k`` using first-fit decreasing."""


def best_fit_decreasing_pack(items: Sequence[float], k: float) -> list[list[float]]:
    """Pack items into bins of capacity ``k`` using best-fit decreasing."""


def kk(items: Sequence[float], k: int) -> list[list[float]]:
    """Backward-compatible alias for :func:`karmarkar_karp_partition`."""


def ffd(items: Sequence[float], k: float) -> list[list[float]]:
    """Backward-compatible alias for :func:`first_fit_decreasing_pack`."""


def bfd(items: Sequence[float], k: float) -> list[list[float]]:
    """Backward-compatible alias for :func:`best_fit_decreasing_pack`."""


def karmarkar_karp(
    seqlen_list: Sequence[int], k_partitions: int, equal_size: bool = False
) -> list[list[int]]:
    """True Karmarkar-Karp differencing; returns original indices per partition.

    When ``equal_size`` is True, every partition has exactly
    ``len(seqlen_list) // k_partitions`` items (requires exact divisibility).
    """


def get_seqlen_balanced_partitions(
    seqlen_list: Sequence[int], k_partitions: int, equal_size: bool = False
) -> list[list[int]]:
    """``karmarkar_karp`` plus validation and bookkeeping.

    Guarantees exactly ``k_partitions`` non-empty partitions, each index of
    ``seqlen_list`` assigned exactly once, and each partition's indices sorted
    ascending.
    """
