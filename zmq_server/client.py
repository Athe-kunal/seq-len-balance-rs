import zmq
import zmq.asyncio
import json
import uuid
import asyncio
import argparse
from multiprocessing import get_context
from typing import Any
from zmq_server import (
    mp as zmq_mp,
    system_specs as zmq_system_specs,
    datamodels as zmq_datamodels,
)

try:
    from mpi4py import MPI  # type: ignore[import-not-found]
except ImportError:
    MPI = None


async def run_client(
    *,
    frontend_endpoint: str,
    sequences: list[Any],
) -> zmq_datamodels.JsonDict:
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
    processes: list[Any] = []
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


def _run_worker_process(
    *, backend_endpoint: str, worker_rank: int, device: str
) -> None:
    asyncio.run(
        zmq_mp.worker_loop(
            backend_endpoint=backend_endpoint,
            worker_rank=worker_rank,
            device=device,
        )
    )


async def run_local_cluster(args: argparse.Namespace) -> None:
    worker_devices, mode_label = zmq_system_specs.resolve_worker_devices(
        args.workers, None
    )
    processes = _spawn_local_workers(
        worker_devices=worker_devices,
        backend_endpoint=args.backend_bind,
    )
    try:
        await zmq_mp.router_loop(
            frontend_endpoint=args.frontend_bind,
            backend_endpoint=args.backend_bind,
            worker_devices=worker_devices,
            mode_label=mode_label,
            algorithm=args.algorithm,
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Async ROUTER/DEALER ZeroMQ balancer that partitions sequences by length "
            "using karmarkar_karp, first_fit_decreasing, or best_fit_decreasing."
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
        default="[512, 128, 1024, 256, 768]",
        help="JSON array of integer lengths or sequences for the demo client mode.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["karmarkar_karp", "first_fit_decreasing", "best_fit_decreasing"],
        default="karmarkar_karp",
        help="Balancing algorithm used to partition sequences across workers.",
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
        if zmq_system_specs.is_running_under_mpi():
            asyncio.run(zmq_mp.run_mpi_cluster(args))
        else:
            asyncio.run(run_local_cluster(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
