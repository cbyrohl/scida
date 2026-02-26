"""
Memory management and distributed computing initialization for scida.
"""

import logging
import os
from typing import Optional, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

from scida.helpers_misc import parse_humansize

log = logging.getLogger(__name__)

_auto_init_done = False  # module-level flag to only auto-init/warn once per session


def _ensure_distributed_if_needed():
    """Auto-init distributed scheduler on TNGLab, log debug hint elsewhere.

    Called automatically at the top of ``scida.load()``.  On TNGLab the
    distributed scheduler is started with default settings (2 GB per
    worker, 4 workers) so that ``da.histogram2d`` and similar operations
    don't OOM.  Outside TNGLab a DEBUG-level message nudges users toward
    ``scida.init_resources()`` for large datasets.

    The function is guarded by a module-level flag so it only runs once
    per Python session.
    """
    global _auto_init_done
    if _auto_init_done:
        return

    # Check if a distributed client is already active
    try:
        from dask.distributed import get_client

        get_client()
        _auto_init_done = True
        return  # client already exists, nothing to do
    except (ImportError, ValueError):
        pass  # no client active

    if _detect_tnglab_environment():
        log.info(
            "TNGLab detected: auto-initializing distributed scheduler "
            "with memory limits. Call scida.init_resources() explicitly "
            "to customize settings."
        )
        init_resources()  # uses TNGLab defaults (2GB, 4 workers)
    else:
        log.debug(
            "No dask distributed client found. For large datasets, consider "
            "calling scida.init_resources(memory_limit=...) to prevent "
            "out-of-memory errors. See https://scida.io/largedatasets/"
        )

    _auto_init_done = True


def init_resources(
    memory_limit: Optional[Union[str, int]] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    dashboard_port: Optional[int] = 8787,
    **cluster_kwargs,
) -> Optional[object]:
    """
    Initialize scida with optional distributed computing and memory management.
    This function sets up dask cluster as requested.

    Parameters
    ----------
    memory_limit : str or int, optional
        Memory limit per worker. Can be specified as:
        - String with units: "4GB", "2GiB", "500MB"
        - Integer in bytes: 4000000000
        If None and n_workers is specified, use half of total system memory.
        If None and n_workers is None, we use the dask default scheduler, which does not impose limits.
    n_workers : int, optional
        Number of worker processes. If None and memory_limit is not None, set to cpu count.
    threads_per_worker : int, optional
        Number of threads per worker. If None, defaults to 1.
    dashboard_port : int, optional
        Port for the dask dashboard. Set to None to disable dashboard.
        Default is 8787.
    **cluster_kwargs
        Additional keyword arguments passed to LocalCluster.

    Returns
    -------
    Client or None
        Returns the dask Client if distributed computing is enabled, None otherwise.


    Notes
    -----
    - Once initialized, all subsequent dask operations will use the configured scheduler
    - Call this function before loading datasets for results
    - On TNGLab, we default to a memory limit of 2GB and 4 workers (can be overridden).
    - The dashboard can be accessed at http://localhost:{dashboard_port} (if enabled)
    - To reset or change configuration, restart your Python session
    """

    # Handle TNGLab environment first
    is_tnglab = _detect_tnglab_environment()
    if is_tnglab:
        log.info(
            "TNGLab environment detected, defaulting to memory_limit=2GB and n_workers=4"
        )
        memory_limit = memory_limit or "2GB"
        n_workers = n_workers or 4

    # Use distributed if:
    # - memory_limit is specified, OR
    # - n_workers is specified, OR
    # - we're in TNGLab environment
    use_distributed = memory_limit is not None or n_workers is not None or is_tnglab

    if not use_distributed:
        log.info(
            "Using default dask scheduler (no memory limits or worker constraints)."
        )
        return None

    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        log.error(
            "dask.distributed not available. Install with: pip install dask[distributed]"
        )
        return None

    # Set defaults based on what's specified
    if memory_limit is None and n_workers is not None:
        # memory_limit=None but n_workers specified -> use half system memory
        memory_limit = _get_default_memory_limit()
        log.info(
            f"No memory limit specified but n_workers={n_workers}, using half system memory: {memory_limit}"
        )

    if n_workers is None and memory_limit is not None:
        # n_workers=None but memory_limit specified -> use CPU count
        n_workers = os.cpu_count() or 1
        log.info(
            f"No n_workers specified but memory_limit={memory_limit}, using CPU count: {n_workers}"
        )

    if threads_per_worker is None:
        threads_per_worker = 1  # Conservative for memory-intensive tasks

    # Prepare cluster arguments
    cluster_args = {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_limit": memory_limit,
        **cluster_kwargs,
    }

    if dashboard_port is not None:
        cluster_args["dashboard_address"] = f":{dashboard_port}"
    else:
        cluster_args["dashboard_address"] = None

    # Handle integer memory_limit (already in bytes)
    if isinstance(memory_limit, int):
        memory_limit_bytes = memory_limit
    else:
        memory_limit_bytes = parse_humansize(memory_limit)

    try:
        log.info(
            f"Initializing LocalCluster with {n_workers} workers, "
            f"{memory_limit} memory limit per worker"
        )

        cluster = LocalCluster(**cluster_args)
        client = Client(cluster)

        log.info("Dask distributed client initialized successfully")
        if dashboard_port is not None:
            log.info(f"Dashboard available at: http://localhost:{dashboard_port}")

        # Log cluster info
        log.info(
            f"Total memory available: {n_workers * memory_limit_bytes / 1e9:.1f} GB"
        )

        return client

    except Exception as e:
        log.error(f"Failed to initialize distributed client: {e}")
        log.info("Falling back to default dask scheduler")
        return None


def _detect_tnglab_environment() -> bool:
    """
    Detect if we're running in TNGLab environment.
    """
    # Check for TNGLab-specific environment variables
    return (
        os.environ.get("NB_USER") == "tnguser"
        or os.environ.get("IS_TNGLAB") is not None
    )


def _get_default_memory_limit() -> str:
    """
    Get a reasonable default memory limit based on system resources.
    """
    if not PSUTIL_AVAILABLE:
        log.warning("psutil not available for memory detection, using 4GB default")
        return "4GB"

    try:
        total_memory = psutil.virtual_memory().total
        # Use about 50% of system memory
        memory_per_worker = total_memory * 0.5
        return f"{int(memory_per_worker)}B"
    except Exception:
        log.warning("Could not detect system memory, using 4GB default")
        return "4GB"
