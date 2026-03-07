import dataclasses
from typing import Any, Literal


JsonDict = dict[str, Any]

Algorithm = Literal["karmarkar_karp", "first_fit_decreasing", "best_fit_decreasing"]


@dataclasses.dataclass(slots=True)
class PendingRequest:
    client_prefix: list[bytes]
    expected_parts: int
    partitions: dict[int, JsonDict] = dataclasses.field(default_factory=dict)
    worker_memory: dict[int, JsonDict] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class ProcessPartitionResult:
    worker_rank: int
    device: str
    num_sequences: int
    total_tokens: int
    sequence_ids: list[int]
    memory: JsonDict
