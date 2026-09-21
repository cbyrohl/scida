"""Integration tests for memory-limited Dask clusters."""

import os
import threading
from unittest.mock import patch

import dask
import pytest

from scida.init import init_resources

pytestmark = pytest.mark.integration


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestInitIntegration:
    """Integration tests with real dask distributed clusters."""

    def test_local_defaults_use_threads_in_current_process(self, monkeypatch):
        monkeypatch.delenv("IS_TNGLAB", raising=False)
        monkeypatch.delenv("NB_USER", raising=False)
        monkeypatch.setattr("scida.init._auto_init_done", False)
        barrier = threading.Barrier(2)

        @dask.delayed
        def identify_worker(i):
            barrier.wait(timeout=10)
            return os.getpid(), threading.get_ident()

        with dask.config.set(scheduler=None, num_workers=None):
            with patch("scida.init.cpu_count", return_value=2):
                result = init_resources()
            assert result is None
            workers = dask.compute(identify_worker(0), identify_worker(1))

        assert {pid for pid, _ in workers} == {os.getpid()}
        assert len({thread for _, thread in workers}) == 2

    def test_distributed_detected_resources_reach_real_workers(self, monkeypatch):
        monkeypatch.delenv("IS_TNGLAB", raising=False)
        monkeypatch.delenv("NB_USER", raising=False)
        total_memory = 1024 * 10**6
        client = None
        try:
            with (
                patch("scida.init.cpu_count", return_value=2),
                patch("distributed.system.memory_limit", return_value=total_memory),
            ):
                client = init_resources(use_distributed=True, dashboard_port=None)
            assert client is not None
            workers = client.scheduler_info()["workers"]
            assert len(workers) == 2
            assert all(
                worker["memory_limit"] == total_memory // 2
                for worker in workers.values()
            )
            assert dask.base.get_scheduler() == client.get
            assert client.submit(os.getpid).result(timeout=30) != os.getpid()
        finally:
            if client is not None:
                client.close()
                client.cluster.close()

    def test_histogram2d_with_memory_limited_cluster(self):
        """Test that da.histogram2d works with a memory-limited cluster.

        Verifies the configured per-worker limits and compares a chunked
        histogram with NumPy. The limit leaves room for the worker runtime;
        this checks configuration and correctness, not peak memory usage.
        """
        import dask.array as da
        import numpy as np
        from dask.distributed import Client, LocalCluster

        memory_limit_mb = 512
        memory_limit_bytes = memory_limit_mb * 10**6
        cluster = None
        client = None
        try:
            cluster = LocalCluster(
                n_workers=2,
                threads_per_worker=1,
                memory_limit=f"{memory_limit_mb}MB",
                dashboard_address=None,
            )
            client = Client(cluster)

            # Verify memory limits are actually applied to workers
            info = client.scheduler_info()
            for worker_info in info["workers"].values():
                assert worker_info["memory_limit"] == memory_limit_bytes, (
                    f"Worker memory limit {worker_info['memory_limit']} != "
                    f"expected {memory_limit_bytes}"
                )

            rng = np.random.default_rng(42)
            n = 5_000_000
            x_np = rng.standard_normal(n)
            y_np = rng.standard_normal(n)

            x_da = da.from_array(x_np, chunks=500_000)
            y_da = da.from_array(y_np, chunks=500_000)

            bins = 50
            data_range = [
                [float(x_np.min()), float(x_np.max())],
                [float(y_np.min()), float(y_np.max())],
            ]
            h_dask, xedges_dask, yedges_dask = da.histogram2d(
                x_da, y_da, bins=bins, range=data_range
            )
            h_dask = client.compute(h_dask).result(timeout=30)

            h_np, xedges_np, yedges_np = np.histogram2d(
                x_np, y_np, bins=bins, range=data_range
            )

            np.testing.assert_array_equal(h_dask, h_np)
            np.testing.assert_allclose(xedges_dask, xedges_np)
            np.testing.assert_allclose(yedges_dask, yedges_np)

            assert h_dask.shape == (bins, bins)
            assert h_dask.sum() == n
        finally:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()

    def test_init_resources_creates_working_cluster(self):
        """Test that init_resources returns a functional dask client
        with the requested memory limits applied to workers."""
        import dask.array as da

        memory_limit_mb = 512
        memory_limit_bytes = memory_limit_mb * 10**6
        client = None
        try:
            client = init_resources(
                use_distributed=True,
                memory_limit=f"{memory_limit_mb}MB",
                n_workers=2,
                dashboard_port=None,
            )
            assert client is not None

            # Verify the cluster is functional
            result = client.compute(da.ones(1000).sum()).result(timeout=30)
            assert result == 1000.0

            # Verify worker count and memory limits
            info = client.scheduler_info()
            assert len(info["workers"]) == 2
            for worker_info in info["workers"].values():
                assert worker_info["memory_limit"] == memory_limit_bytes
        finally:
            if client is not None:
                client.close()
                client.cluster.close()
