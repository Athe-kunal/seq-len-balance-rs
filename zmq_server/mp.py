from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import uuid

import zmq
import zmq.asyncio
from zmq_server import (
    balancer as zmq_balancer,
    datamodels as zmq_datamodels,
    system_specs as zmq_system_specs,
)

try:
    from mpi4py import MPI  # type: ignore[import-not-found]
except ImportError:
    MPI = None


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
            "memory": zmq_system_specs.worker_memory_snapshot(device),
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
            result = await zmq_balancer.process_partition(
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
                    "result": dataclasses.asdict(result),
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
    algorithm: zmq_datamodels.Algorithm = "karmarkar_karp",
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

    pending: dict[str, zmq_datamodels.PendingRequest] = {}
    ready_workers: dict[int, zmq_datamodels.JsonDict] = {}

    async def handle_frontend() -> None:
        while True:
            frames = await frontend.recv_multipart()
            if not frames:
                continue

            payload = json.loads(frames[-1].decode("utf-8"))
            request_id = payload.get("request_id", uuid.uuid4().hex)
            sequences = payload.get("sequences")
            if not isinstance(sequences, list):
                reply: zmq_datamodels.JsonDict = {
                    "error": "payload must contain a list-valued `sequences` field"
                }
                await frontend.send_multipart(
                    [*frames[:-1], json.dumps(reply).encode("utf-8")]
                )
                continue

            partitions = zmq_balancer.balance_sequences(sequences, len(worker_devices), algorithm)
            request = zmq_datamodels.PendingRequest(
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

            reply: zmq_datamodels.JsonDict = {
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


async def run_mpi_cluster(args: argparse.Namespace) -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is required for MPI mode")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise RuntimeError("MPI mode requires at least 2 ranks (1 router + 1 worker)")

    worker_devices, mode_label = zmq_system_specs.resolve_worker_devices(
        args.workers, size - 1
    )
    usable_workers = len(worker_devices)
    if rank == 0:
        await router_loop(
            frontend_endpoint=args.frontend_bind,
            backend_endpoint=args.backend_bind,
            worker_devices=worker_devices,
            mode_label=mode_label,
            algorithm=args.algorithm,
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
