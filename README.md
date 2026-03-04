# seq-len-balance-rs

Fast sequence-length balancing: KK partition, FFD, and BFD bin packing (Rust + Python).

## Setup

```bash
uv sync
pip install maturin
maturin develop
```

## Usage

```python
from seq_len_balance import kk, ffd, bfd

print(kk([8, 7, 6, 5, 4], 2))   # -> [[8, 5], [7, 4], [6]]
print(ffd([6, 3, 4, 5, 2, 7, 1], 10))  # -> [[7, 3], [6, 4], [5, 2, 1]]
print(bfd([6, 3, 4, 5, 2, 7, 1], 10))  # -> [[7, 2, 1], [6, 4], [5, 3]]
```

## Benchmark

```bash
uv sync --group benchmark
uv run python test_kk.py
```

```
=================================================================
KK (greedy LPT) benchmark  k=4  repeats=3
=================================================================
n=   10,000  Python:      2.4 ms   Rust:      0.4 ms
n=  100,000  Python:     25.5 ms   Rust:      2.9 ms
n=  500,000  Python:    131.4 ms   Rust:     16.3 ms
n=1,000,000  Python:    294.7 ms   Rust:     33.1 ms
n=2,000,000  Python:    663.9 ms   Rust:     69.6 ms
n=4,000,000  Python:   1405.9 ms   Rust:    144.9 ms
n=10,000,000  Python:   3577.8 ms   Rust:    377.6 ms
```

## Rust in Jupyter (optional)

```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```
