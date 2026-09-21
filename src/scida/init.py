"""
Memory management and distributed computing initialization for scida.
"""

import logging
import os
from typing import Optional, Union

import dask
from dask.system import cpu_count

from scida.misc import parse_size

log = logging.getLogger(__name__)

_auto_init_done = False  # module-level flag to only auto-init/warn once per session


def _ensure_distributed_if_needed():
    """Auto-init distributed scheduler on TNGLab, log debug hint elsewhere.

    Called automatically at the top of ``scida.load()``.  On TNGLab the
    distributed scheduler is started with default settings (2 GB per
    worker, 4 workers) to manage memory during ``da.histogram2d`` and
    similar operations. Outside TNGLab a DEBUG-level message points to
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
            "calling scida.init_resources(use_distributed=True) to configure "
            "worker memory limits. See https://scida.io/largedatasets/"
        )

    _auto_init_done = True


def init_resources(
    memory_limit: Optional[Union[str, int]] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    dashboard_port: Optional[int] = 8787,
    *,
    use_distributed: Optional[bool] = None,
    **cluster_kwargs,
) -> Optional[object]:
    """
    Configure local Dask resources, optionally using a distributed cluster.

    Parameters
    ----------
    memory_limit : str or int, optional
        Memory limit per distributed worker. Requires use_distributed=True
        outside TNGLab. Can be specified as:
        - String with units: "4GB", "2GiB", "500MB"
        - Integer in bytes: 4000000000
        If None, divide the full detected system/container memory limit
        equally among the workers. On TNGLab, default to "2GB" per worker.
    n_workers : int, optional
        Number of local threads, or worker processes in distributed mode.
        If None, use the available CPU count; in distributed mode, divide
        it by threads_per_worker, with a minimum of one worker.
        CPU detection respects affinity and container quotas. Distributed
        mode on TNGLab defaults to four workers.
    threads_per_worker : int, optional
        Number of threads per distributed worker. Requires distributed
        mode. If None, defaults to 1 in that mode.
    dashboard_port : int, optional
        Port for the dask dashboard. Set to None to disable dashboard.
        Default is 8787. Only used in distributed mode.
    use_distributed : bool, optional
        If True, start a local distributed cluster with worker memory limits.
        If False, configure the threaded scheduler without a memory limit.
        If None, use distributed mode on TNGLab and threads elsewhere.
    **cluster_kwargs
        Additional keyword arguments passed to LocalCluster. Requires
        distributed mode.

    Returns
    -------
    Client or None
        The client connected to the local cluster, or None when using the
        threaded scheduler, distributed is unavailable, or cluster startup fails.


    Notes
    -----
    - Calling with no arguments configures the threaded scheduler locally.
    - Subsequent Dask operations use the configured scheduler.
    - Call this function before loading datasets.
    - On TNGLab, we default to distributed mode with 2GB per worker and 4 workers.
      Pass use_distributed=False to opt out, including during later load() calls.
    - Memory limits use Dask's best-effort worker memory management.
    - The dashboard can be accessed at http://localhost:{dashboard_port} (if enabled)
    - To reset or change configuration, restart your Python session
    """

    global _auto_init_done

    if use_distributed is not None and not isinstance(use_distributed, bool):
        raise ValueError("use_distributed must be True, False, or None")

    for name, value in {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
    }.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")

    is_tnglab = _detect_tnglab_environment()
    if use_distributed is None:
        use_distributed = is_tnglab

    if not use_distributed:
        if memory_limit is not None:
            raise ValueError("memory_limit requires use_distributed=True")
        if threads_per_worker is not None or cluster_kwargs:
            raise ValueError(
                "threads_per_worker and LocalCluster options require "
                "use_distributed=True; use n_workers to set the local thread count"
            )
        if n_workers is None:
            n_workers = cpu_count()
        dask.config.set(scheduler="threads", num_workers=n_workers)
        _auto_init_done = True  # Respect this scheduler choice during later load().
        log.info("Using the local threaded scheduler with %s threads", n_workers)
        return None

    # Keep TNGLab's distributed defaults unless explicitly overridden.
    if is_tnglab:
        log.info(
            "TNGLab environment detected, defaulting to memory_limit=2GB and n_workers=4"
        )
        if memory_limit is None:
            memory_limit = "2GB"
        if n_workers is None:
            n_workers = 4

    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        log.error(
            "dask.distributed not available. Install with: pip install dask[distributed]"
        )
        return None

    if threads_per_worker is None:
        threads_per_worker = 1

    if n_workers is None:
        n_workers = max(1, cpu_count() // threads_per_worker)
        log.info(
            f"Using {n_workers} local workers based on available CPUs "
            f"and {threads_per_worker} threads per worker"
        )

    if memory_limit is None:
        memory_limit = _get_default_memory_limit(n_workers)

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
        memory_limit_bytes = parse_size(memory_limit)

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
            f"Combined worker memory limit: {n_workers * memory_limit_bytes / 1e9:.1f} GB"
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


def _get_default_memory_limit(n_workers: int) -> int:
    """Divide the detected system/container memory limit among all workers."""
    from distributed.system import memory_limit

    per_worker = memory_limit() // n_workers
    if per_worker < 1:
        raise ValueError("Detected memory limit is too small for the requested workers")
    return per_worker
