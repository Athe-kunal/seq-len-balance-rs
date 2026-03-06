from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

import zmq
import zmq.asyncio
from seq_len_balance import kk as rust_kk  # type: ignore[import-untyped]

try:
    import psutil  # type: ignore[import-not-found]
except ImportError:
    psutil = None

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None

try:
    from mpi4py import MPI  # type: ignore[import-not-found]
except ImportError:
    MPI = None


JsonDict = dict[str, Any]


@dataclass(slots=True)
class PendingRequest:
    client_prefix: list[bytes]
    expected_parts: int
    partitions: dict[int, JsonDict] = field(default_factory=dict)
    worker_memory: dict[int, JsonDict] = field(default_factory=dict)


def _is_running_under_mpi() -> bool:
    return any(
        key in os.environ
        for key in (
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_SIZE",
        )
    )


def _cuda_device_names() -> list[str]:
    if torch is None or not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def _resolve_worker_devices(
    requested_workers: int | None,
    mpi_worker_count: int | None,
) -> tuple[list[str], str]:
    cuda_devices = _cuda_device_names()
    if cuda_devices:
        limit = len(cuda_devices)
        if mpi_worker_count is not None:
            limit = min(limit, mpi_worker_count)
        if requested_workers is not None:
            limit = min(limit, requested_workers)
        if limit <= 0:
            raise ValueError("No usable CUDA workers were available")
        return cuda_devices[:limit], "cuda"

    worker_count = requested_workers
    if worker_count is None:
        worker_count = mpi_worker_count if mpi_worker_count is not None else (os.cpu_count() or 1)
    if mpi_worker_count is not None:
        worker_count = min(worker_count, mpi_worker_count)
    if worker_count <= 0:
        raise ValueError("worker_count must be >= 1")
    return ["cpu"] * worker_count, "mpi" if mpi_worker_count is not None else "cpu"


def _process_rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(rss)
    return int(rss * 1024)


def _system_available_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.virtual_memory().available)


def _worker_memory_snapshot(device: str) -> JsonDict:
    snapshot: JsonDict = {
        "rss_bytes": _process_rss_bytes(),
        "system_available_bytes": _system_available_bytes(),
    }
    if device.startswith("cuda") and torch is not None:
        device_index = int(device.split(":", 1)[1])
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        snapshot.update(
            {
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                "cuda_used_bytes": int(total_bytes - free_bytes),
            }
        )
    return snapshot


def _length_of_sequence(sequence: Any) -> int:
    try:
        return len(sequence)
    except TypeError as exc:
        raise TypeError(f"Sequence {sequence!r} does not define len()") from exc


def balance_sequences(sequences: list[Any], worker_count: int) -> list[list[JsonDict]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be >= 1")
    if not sequences:
        return [[] for _ in range(worker_count)]

    lengths = [float(_length_of_sequence(sequence)) for sequence in sequences]
    bins = rust_kk(lengths, worker_count)

    seqs_by_length: dict[float, deque[JsonDict]] = defaultdict(deque)
    for sequence_id, sequence in enumerate(sequences):
        seq_len = float(_length_of_sequence(sequence))
        seqs_by_length[seq_len].append(
            {
                "sequence_id": sequence_id,
                "length": int(seq_len),
                "sequence": sequence,
            }
        )

    partitions: list[list[JsonDict]] = []
    for length_bin in bins:
        shard: list[JsonDict] = []
        for seq_len in length_bin:
            shard.append(seqs_by_length[float(seq_len)].popleft())
        partitions.append(shard)
    return partitions


async def process_partition(
    partition: list[JsonDict],
    *,
    worker_rank: int,
    device: str,
) -> JsonDict:
    token_lengths = [item["length"] for item in partition]
    total_tokens = sum(token_lengths)

    if device.startswith("cuda") and torch is not None and token_lengths:
        device_index = int(device.split(":", 1)[1])
        torch.cuda.set_device(device_index)
        _ = torch.tensor(token_lengths, device=device)
        torch.cuda.synchronize(device_index)

    await asyncio.sleep(0)

    return {
        "worker_rank": worker_rank,
        "device": device,
        "num_sequences": len(partition),
        "total_tokens": total_tokens,
        "sequence_ids": [item["sequence_id"] for item in partition],
        "memory": _worker_memory_snapshot(device),
    }


async def worker_loop(
    *,
    backend_endpoint: str,
    worker_rank: int,
    device: str,
) -> None:
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.IDENTITY, f"worker-{worker_rank}".encode())
    socket.connect(backend_endpoint)

    await socket.send_json(
        {
            "kind": "worker_ready",
            "worker_rank": worker_rank,
            "device": device,
            "memory": _worker_memory_snapshot(device),
        }
    )

    try:
        while True:
            message = await socket.recv_json()
            if message.get("kind") == "shutdown":
                return
            if message.get("kind") != "task":
                continue

            partition = message["partition"]
            result = await process_partition(
                partition,
                worker_rank=worker_rank,
                device=device,
            )
            await socket.send_json(
                {
                    "kind": "task_result",
                    "request_id": message["request_id"],
                    "partition_idx": message["partition_idx"],
                    "partition": partition,
                    "result": result,
                }
            )
    finally:
        socket.close(0)


async def router_loop(
    *,
    frontend_endpoint: str,
    backend_endpoint: str,
    worker_devices: list[str],
    mode_label: str,
) -> None:
    if not worker_devices:
        raise ValueError("At least one worker is required")

    context = zmq.asyncio.Context.instance()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.DEALER)
    frontend.setsockopt(zmq.LINGER, 0)
    backend.setsockopt(zmq.LINGER, 0)
    frontend.bind(frontend_endpoint)
    backend.bind(backend_endpoint)

    pending: dict[str, PendingRequest] = {}
    ready_workers: dict[int, JsonDict] = {}

    async def handle_frontend() -> None:
        while True:
            frames = await frontend.recv_multipart()
            if not frames:
                continue

            payload = json.loads(frames[-1].decode("utf-8"))
            request_id = payload.get("request_id", uuid.uuid4().hex)
            sequences = payload.get("sequences")
            if not isinstance(sequences, list):
                reply: JsonDict = {
                    "error": "payload must contain a list-valued `sequences` field"
                }
                await frontend.send_multipart(
                    [*frames[:-1], json.dumps(reply).encode("utf-8")]
                )
                continue

            partitions = balance_sequences(sequences, len(worker_devices))
            request = PendingRequest(
                client_prefix=frames[:-1],
                expected_parts=len(partitions),
            )
            pending[request_id] = request

            if not partitions:
                reply = {
                    "request_id": request_id,
                    "mode": mode_label,
                    "world_size": len(worker_devices),
                    "partitions": [],
                    "worker_memory": [],
                }
                await frontend.send_multipart(
                    [*frames[:-1], json.dumps(reply).encode("utf-8")]
                )
                pending.pop(request_id, None)
                continue

            for partition_idx, partition in enumerate(partitions):
                await backend.send_json(
                    {
                        "kind": "task",
                        "request_id": request_id,
                        "partition_idx": partition_idx,
                        "partition": partition,
                    }
                )

    async def handle_backend() -> None:
        while True:
            message = await backend.recv_json()
            kind = message.get("kind")

            if kind == "worker_ready":
                ready_workers[message["worker_rank"]] = {
                    "worker_rank": message["worker_rank"],
                    "device": message["device"],
                    "memory": message["memory"],
                }
                continue

            if kind != "task_result":
                continue

            request_id = message["request_id"]
            request = pending.get(request_id)
            if request is None:
                continue

            partition_idx = int(message["partition_idx"])
            result = message["result"]
            request.partitions[partition_idx] = {
                "partition_idx": partition_idx,
                "sequence_ids": result["sequence_ids"],
                "num_sequences": result["num_sequences"],
                "total_tokens": result["total_tokens"],
                "worker_rank": result["worker_rank"],
                "device": result["device"],
            }
            request.worker_memory[result["worker_rank"]] = {
                "worker_rank": result["worker_rank"],
                "device": result["device"],
                **result["memory"],
            }

            if len(request.partitions) != request.expected_parts:
                continue

            ordered_partitions = [
                request.partitions[idx] for idx in sorted(request.partitions)
            ]
            worker_memory = [
                request.worker_memory[idx] for idx in sorted(request.worker_memory)
            ]

            reply = {
                "request_id": request_id,
                "world_size": len(worker_devices),
                "mode": mode_label,
                "partitions": ordered_partitions,
                "worker_memory": worker_memory,
                "ready_workers": [ready_workers[idx] for idx in sorted(ready_workers)],
            }
            await frontend.send_multipart(
                [*request.client_prefix, json.dumps(reply).encode("utf-8")]
            )
            pending.pop(request_id, None)

    try:
        await asyncio.gather(handle_frontend(), handle_backend())
    finally:
        frontend.close(0)
        backend.close(0)


async def run_client(
    *,
    frontend_endpoint: str,
    sequences: list[Any],
) -> JsonDict:
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(frontend_endpoint)

    try:
        request_id = uuid.uuid4().hex
        await socket.send_json({"request_id": request_id, "sequences": sequences})
        return await socket.recv_json()
    finally:
        socket.close(0)


def _spawn_local_workers(
    *,
    worker_devices: list[str],
    backend_endpoint: str,
) -> list[Any]:
    ctx = get_context("spawn")
    processes = []
    for rank, device in enumerate(worker_devices, start=1):
        process = ctx.Process(
            target=_run_worker_process,
            kwargs={
                "backend_endpoint": backend_endpoint,
                "worker_rank": rank,
                "device": device,
            },
        )
        process.start()
        processes.append(process)
    return processes


def _run_worker_process(*, backend_endpoint: str, worker_rank: int, device: str) -> None:
    asyncio.run(
        worker_loop(
            backend_endpoint=backend_endpoint,
            worker_rank=worker_rank,
            device=device,
        )
    )


async def run_local_cluster(args: argparse.Namespace) -> None:
    worker_devices, mode_label = _resolve_worker_devices(args.workers, None)
    processes = _spawn_local_workers(
        worker_devices=worker_devices,
        backend_endpoint=args.backend_bind,
    )
    try:
        await router_loop(
            frontend_endpoint=args.frontend_bind,
            backend_endpoint=args.backend_bind,
            worker_devices=worker_devices,
            mode_label=mode_label,
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)


async def run_mpi_cluster(args: argparse.Namespace) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required for MPI mode")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise RuntimeError("MPI mode requires at least 2 ranks (1 router + 1 worker)")

    worker_devices, mode_label = _resolve_worker_devices(args.workers, size - 1)
    usable_workers = len(worker_devices)
    if rank == 0:
        await router_loop(
            frontend_endpoint=args.frontend_bind,
            backend_endpoint=args.backend_bind,
            worker_devices=worker_devices,
            mode_label=mode_label,
        )
        return

    if rank <= usable_workers:
        device = worker_devices[rank - 1]
        await worker_loop(
            backend_endpoint=args.backend_bind,
            worker_rank=rank,
            device=device,
        )
        return

    await asyncio.Event().wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Async ROUTER/DEALER ZeroMQ balancer using seq_len_balance.kk "
            "to partition sequences by length."
        )
    )
    parser.add_argument(
        "mode",
        choices=["serve", "client"],
        help="Run the balancer server/cluster or submit one demo client request.",
    )
    parser.add_argument(
        "--frontend-bind",
        default="tcp://127.0.0.1:5555",
        help="ROUTER endpoint for client traffic.",
    )
    parser.add_argument(
        "--backend-bind",
        default="tcp://127.0.0.1:5556",
        help="DEALER endpoint used internally by worker processes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Requested worker count. If CUDA is available this caps the GPU world size; "
            "otherwise it controls the MPI/local CPU worker count."
        ),
    )
    parser.add_argument(
        "--sequences-json",
        default='["AAAA", "A", "BBBBBBBB", "CCC", "DDDDDD"]',
        help="JSON array of sequences for the demo client mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "client":
        sequences = json.loads(args.sequences_json)
        response = asyncio.run(
            run_client(
                frontend_endpoint=args.frontend_bind,
                sequences=sequences,
            )
        )
        print(json.dumps(response, indent=2))
        return

    try:
        if _is_running_under_mpi():
            asyncio.run(run_mpi_cluster(args))
        else:
            asyncio.run(run_local_cluster(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
