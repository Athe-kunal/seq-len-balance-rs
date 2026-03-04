"""
Correctness tests for the Karmarkar-Karp (greedy LPT) partition algorithm.

Two implementations are compared:
  - `kk_python`: pure-Python reference using the `heapq` stdlib
  - `seq_len_balance.kk`: Rust extension built with maturin

Both use the same greedy strategy: sort items descending, then repeatedly
assign each item to the bin with the current minimum sum (min-heap).

Run with:
    uv run pytest test_kk.py -v
or (after `uv run maturin develop`):
    uv run python test_kk.py
"""

from __future__ import annotations

import heapq
import random
from collections import Counter
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Pure-Python reference implementation
# ---------------------------------------------------------------------------


def kk_python(items: list[float], k: int) -> list[list[float]]:
    """Greedy LPT partition into k bins using Python's heapq min-heap."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    sorted_items = sorted(items, reverse=True)
    # heap entries: (current_bin_sum, bin_index)
    heap: list[tuple[float, int]] = [(0.0, i) for i in range(k)]
    heapq.heapify(heap)
    bins: list[list[float]] = [[] for _ in range(k)]
    for item in sorted_items:
        min_sum, idx = heapq.heappop(heap)
        bins[idx].append(item)
        heapq.heappush(heap, (min_sum + item, idx))
    return bins


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def bin_sums(bins: list[list[float]]) -> list[float]:
    return [sum(b) for b in bins]


def imbalance(bins: list[list[float]]) -> float:
    """max_bin_sum - min_bin_sum."""
    sums = bin_sums(bins)
    return max(sums) - min(sums)


def assert_valid_partition(items: list[float], bins: list[list[float]], k: int) -> None:
    """Verify the partition is complete and consistent."""
    # Correct number of bins
    assert len(bins) == k, f"Expected {k} bins, got {len(bins)}"

    # Every item appears exactly once (multiset equality)
    original_counts = Counter(round(x, 10) for x in items)
    partitioned_counts: Counter = Counter()
    for b in bins:
        for x in b:
            partitioned_counts[round(x, 10)] += 1
    assert original_counts == partitioned_counts, "Items were lost or duplicated"

    # All items are non-negative (the algorithm only makes sense for non-negative inputs)
    for b in bins:
        for x in b:
            assert x >= 0.0, f"Unexpected negative item: {x}"


def compare_implementations(
    items: list[float],
    k: int,
    *,
    rust_kk,
    atol: float = 1e-6,
) -> None:
    """Run both implementations and assert they produce the same imbalance."""
    py_bins = kk_python(items, k)
    rs_bins = rust_kk(items, k)

    assert_valid_partition(items, py_bins, k)
    assert_valid_partition(items, rs_bins, k)

    py_imbalance = imbalance(py_bins)
    rs_imbalance = imbalance(rs_bins)

    # Both implementations should reach the same imbalance (deterministic greedy)
    assert (
        abs(py_imbalance - rs_imbalance) <= atol
    ), f"Imbalance mismatch: Python={py_imbalance:.6f}, Rust={rs_imbalance:.6f}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIZES = [10_000, 100_000, 1_000_000]
K_VALUES = [2, 4, 8, 16]


@pytest.fixture(scope="session")
def rust_kk():
    """Import the Rust extension; skip all tests if it isn't built yet."""
    try:
        from seq_len_balance import kk  # noqa: PLC0415

        return kk
    except ImportError:
        pytest.skip(
            "seq_len_balance extension not found — run `uv run maturin develop` first"
        )


# ---------------------------------------------------------------------------
# Correctness: Python-only (no Rust dependency)
# ---------------------------------------------------------------------------


class TestKKPythonCorrectness:
    """Validate the pure-Python reference implementation."""

    def test_tiny_example(self):
        """Reproduce the worked example from kk.rs comments."""
        items = [8, 7, 6, 5, 4]
        bins = kk_python(items, 3)
        assert_valid_partition(items, bins, 3)
        # Greedy LPT on [8,7,6,5,4] into 3 bins → [8],[7,4]=11,[6,5]=11
        # imbalance = 3; it must be strictly less than the largest item (8)
        assert imbalance(bins) < max(items)

    def test_single_bin(self):
        items = [3.0, 1.0, 4.0, 1.0, 5.0]
        bins = kk_python(items, 1)
        assert len(bins) == 1
        assert abs(sum(bins[0]) - sum(items)) < 1e-9

    def test_k_equals_n(self):
        items = list(range(1, 9))
        bins = kk_python(items, len(items))
        assert_valid_partition(items, bins, len(items))
        for b in bins:
            assert len(b) == 1

    def test_uniform_weights(self):
        """All-equal weights → perfect balance (imbalance == 0)."""
        items = [5.0] * 100
        bins = kk_python(items, 4)
        assert_valid_partition(items, bins, 4)
        assert imbalance(bins) == 0.0

    def test_all_zeros(self):
        items = [0.0] * 50
        bins = kk_python(items, 5)
        assert_valid_partition(items, bins, 5)
        assert imbalance(bins) == 0.0

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("k", [2, 8])
    def test_large_random_integer(self, n: int, k: int):
        """
        For large random integer inputs the greedy LPT imbalance should be
        much smaller than the total sum.  We check that imbalance / total < 0.01.
        """
        rng = random.Random(42)
        items = [float(rng.randint(1, 1000)) for _ in range(n)]
        bins = kk_python(items, k)
        assert_valid_partition(items, bins, k)
        ratio = imbalance(bins) / sum(items)
        assert (
            ratio < 0.01
        ), f"n={n}, k={k}: imbalance ratio={ratio:.4%} exceeds 1% threshold"

    @pytest.mark.parametrize("n", SIZES)
    def test_large_random_float(self, n: int):
        """Float weights — partition must remain lossless (multiset equality)."""
        rng = random.Random(7)
        items = [rng.uniform(0.1, 100.0) for _ in range(n)]
        bins = kk_python(items, 4)
        assert_valid_partition(items, bins, 4)

    @pytest.mark.parametrize("n", SIZES)
    def test_imbalance_shrinks_with_n(self, n: int):
        """
        As n grows the absolute imbalance per item should decrease (law of large
        numbers effect on greedy LPT).
        We simply check imbalance / n < 10 (very loose but always true for
        items drawn from [1, 1000]).
        """
        rng = random.Random(99)
        items = [float(rng.randint(1, 1000)) for _ in range(n)]
        bins = kk_python(items, 4)
        per_item_imbalance = imbalance(bins) / n
        assert (
            per_item_imbalance < 10.0
        ), f"n={n}: per-item imbalance={per_item_imbalance:.4f} seems too large"


# ---------------------------------------------------------------------------
# Cross-validation: Python vs Rust
# ---------------------------------------------------------------------------


class TestKKRustVsPython:
    """Compare the Rust and Python implementations head-to-head."""

    def test_tiny_example(self, rust_kk):
        items = [8.0, 7.0, 6.0, 5.0, 4.0]
        compare_implementations(items, 3, rust_kk=rust_kk)

    @pytest.mark.parametrize("k", K_VALUES)
    def test_medium_random(self, rust_kk, k: int):
        rng = random.Random(2024)
        items = [float(rng.randint(1, 500)) for _ in range(1_000)]
        compare_implementations(items, k, rust_kk=rust_kk)

    @pytest.mark.parametrize("n", SIZES)
    def test_large_integer_k4(self, rust_kk, n: int):
        rng = random.Random(1337)
        items = [float(rng.randint(1, 1000)) for _ in range(n)]
        compare_implementations(items, 4, rust_kk=rust_kk)

    @pytest.mark.parametrize("n", SIZES)
    def test_large_float_k8(self, rust_kk, n: int):
        rng = random.Random(555)
        items = [rng.uniform(0.5, 200.0) for _ in range(n)]
        compare_implementations(items, 8, rust_kk=rust_kk)

    def test_rust_no_items_lost_1m(self, rust_kk):
        """Dedicated 1 M element lossless check for the Rust binding."""
        rng = random.Random(0)
        items = [float(rng.randint(1, 10_000)) for _ in range(1_000_000)]
        rs_bins = rust_kk(items, 16)
        assert_valid_partition(items, rs_bins, 16)


# ---------------------------------------------------------------------------
# Standalone runner + benchmark bar-plot
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import time

    import matplotlib

    matplotlib.use("Agg")  # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    try:
        from seq_len_balance import kk as rust_kk

        has_rust = True
    except ImportError:
        has_rust = False
        print("[WARN] Rust extension not found; Rust bars will be omitted.\n")

    BENCHMARK_DIR = pathlib.Path(__file__).parent / "benchmark"
    BENCHMARK_DIR.mkdir(exist_ok=True)

    SIZES = [10_000, 100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 10_000_000]
    K = 4
    REPEATS = 3  # average over this many runs to reduce noise

    print("=" * 65)
    print(f"KK (greedy LPT) benchmark  k={K}  repeats={REPEATS}")
    print("=" * 65)

    py_times: list[float] = []
    rs_times: list[float] = []

    for n in SIZES:
        rng = random.Random(42)
        items = [float(rng.randint(1, 1000)) for _ in range(n)]

        # --- Python ---
        elapsed = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            kk_python(items, K)
            elapsed.append(time.perf_counter() - t0)
        py_ms = min(elapsed) * 1_000  # best-of-N in ms
        py_times.append(py_ms)

        # --- Rust ---
        if has_rust:
            elapsed = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                rust_kk(items, K)
                elapsed.append(time.perf_counter() - t0)
            rs_ms = min(elapsed) * 1_000
            rs_times.append(rs_ms)
        else:
            rs_times.append(0.0)

        row = f"n={n:>9,}  Python: {py_ms:8.1f} ms" + (
            f"   Rust: {rs_times[-1]:8.1f} ms" if has_rust else ""
        )
        print(row)

    # ------------------------------------------------------------------
    # Bar chart
    # ------------------------------------------------------------------
    labels = [
        f"{n // 1_000}k" if n < 1_000_000 else f"{n // 1_000_000}M" for n in SIZES
    ]
    x = np.arange(len(SIZES))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    color_py = "#4c9be8"
    color_rs = "#e87c4c"

    if has_rust:
        bars_py = ax.bar(
            x - bar_w / 2,
            py_times,
            bar_w,
            label="Python (heapq)",
            color=color_py,
            zorder=3,
        )
        bars_rs = ax.bar(
            x + bar_w / 2,
            rs_times,
            bar_w,
            label="Rust (seq_len_balance)",
            color=color_rs,
            zorder=3,
        )
        for bar in (*bars_py, *bars_rs):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(py_times + rs_times) * 0.01,
                f"{h:.0f}" if h >= 10 else f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
            )
    else:
        bars_py = ax.bar(
            x, py_times, bar_w * 1.4, label="Python (heapq)", color=color_py, zorder=3
        )
        for bar in bars_py:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(py_times) * 0.01,
                f"{h:.0f}" if h >= 10 else f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f} ms"))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.yaxis.label.set_color("white")
    ax.set_xlabel("Sequence length (n)", color="white", fontsize=12)
    ax.set_ylabel("Time (ms, best of 3)", color="white", fontsize=12)
    ax.set_title(
        f"KK Greedy LPT Partition  —  k={K}  (lower is better)",
        color="white",
        fontsize=14,
        pad=14,
    )
    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.7, zorder=0)
    ax.legend(facecolor="#1e2130", edgecolor="#555", labelcolor="white", fontsize=10)

    out_path = BENCHMARK_DIR / "kk_benchmark.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\nBar chart saved → {out_path}")
