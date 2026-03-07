from __future__ import annotations

import asyncio
import math
from collections import defaultdict, deque
from typing import Any

from seq_len_balance import bfd, ffd, kk
from zmq_server import datamodels as zmq_datamodels
from zmq_server import system_specs as zmq_system_specs

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None


def _length_of_sequence(sequence: Any) -> int:
    try:
        return len(sequence)
    except TypeError as exc:
        raise TypeError(f"Sequence {sequence!r} does not define len()") from exc


def balance_sequences(
    sequences: list[Any],
    worker_count: int,
    algorithm: zmq_datamodels.Algorithm = "karmarkar_karp",
) -> list[list[zmq_datamodels.JsonDict]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be >= 1")
    if not sequences:
        return [[] for _ in range(worker_count)]

    lengths = [float(_length_of_sequence(s)) for s in sequences]

    if algorithm == "karmarkar_karp":
        bins: list[list[float]] = kk(lengths, worker_count)
    else:
        # ffd/bfd use bin capacity; target = ceil(total / k) so we get at most k bins
        capacity = math.ceil(sum(lengths) / worker_count)
        bins = ffd(lengths, capacity) if algorithm == "first_fit_decreasing" else bfd(lengths, capacity)
        # pad to worker_count so every worker receives a partition (possibly empty)
        while len(bins) < worker_count:
            bins.append([])

    seqs_by_length: dict[float, deque[zmq_datamodels.JsonDict]] = defaultdict(deque)
    for sequence_id, sequence in enumerate(sequences):
        seq_len = float(_length_of_sequence(sequence))
        seqs_by_length[seq_len].append(
            {
                "sequence_id": sequence_id,
                "length": int(seq_len),
                "sequence": sequence,
            }
        )

    partitions: list[list[zmq_datamodels.JsonDict]] = []
    for length_bin in bins:
        shard: list[zmq_datamodels.JsonDict] = []
        for seq_len in length_bin:
            shard.append(seqs_by_length[float(seq_len)].popleft())
        partitions.append(shard)
    return partitions


async def process_partition(
    partition: list[zmq_datamodels.JsonDict],
    *,
    worker_rank: int,
    device: str,
) -> zmq_datamodels.ProcessPartitionResult:
    token_lengths = [item["length"] for item in partition]
    total_tokens = sum(token_lengths)

    if device.startswith("cuda") and torch is not None and token_lengths:
        device_index = int(device.split(":", 1)[1])
        torch.cuda.set_device(device_index)
        _ = torch.tensor(token_lengths, device=device)
        torch.cuda.synchronize(device_index)

    await asyncio.sleep(0)

    return zmq_datamodels.ProcessPartitionResult(
        worker_rank=worker_rank,
        device=device,
        num_sequences=len(partition),
        total_tokens=total_tokens,
        sequence_ids=[item["sequence_id"] for item in partition],
        memory=zmq_system_specs.worker_memory_snapshot(device),
    )
