import os
import platform
import resource

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

from zmq_server import datamodels as zmq_datamodels


def is_running_under_mpi() -> bool:
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


def worker_memory_snapshot(device: str) -> zmq_datamodels.JsonDict:
    snapshot: zmq_datamodels.JsonDict = {
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


def resolve_worker_devices(
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
        worker_count = (
            mpi_worker_count if mpi_worker_count is not None else (os.cpu_count() or 1)
        )
    if mpi_worker_count is not None:
        worker_count = min(worker_count, mpi_worker_count)
    if worker_count <= 0:
        raise ValueError("worker_count must be >= 1")
    return ["cpu"] * worker_count, "mpi" if mpi_worker_count is not None else "cpu"
