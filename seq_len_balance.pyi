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
