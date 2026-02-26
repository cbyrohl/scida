"""
Tests for scida.init() function
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

from scida.helpers_misc import parse_humansize
from scida.init import (
    _detect_tnglab_environment,
    _ensure_distributed_if_needed,
    _get_default_memory_limit,
    init_resources,
)


class TestMemoryParsing:
    """Test memory string parsing functions."""

    def test_parse_humansize_basic(self):
        """Test parsing with the existing parse_humansize function."""
        assert parse_humansize("1GIB") == 1024**3
        assert parse_humansize("1.5GIB") == int(1.5 * 1024**3)
        assert parse_humansize("500MIB") == 500 * 1024**2

    def test_parse_humansize_decimal_units(self):
        """Test parsing decimal (SI) units: GB, MB, KB, TB, B."""
        assert parse_humansize("4GB") == 4 * 10**9
        assert parse_humansize("100MB") == 100 * 10**6
        assert parse_humansize("512KB") == 512 * 10**3
        assert parse_humansize("2TB") == 2 * 10**12
        assert parse_humansize("1024B") == 1024

    def test_parse_humansize_fractional_decimal(self):
        """Test fractional values with decimal units."""
        assert parse_humansize("1.5GB") == int(1.5 * 10**9)
        assert parse_humansize("2.5MB") == int(2.5 * 10**6)


class TestTNGLabDetection:
    """Test TNGLab environment detection."""

    def test_detect_tnglab_with_nb_user(self):
        """Test detection with NB_USER environment variable set to tnguser."""
        with patch.dict(os.environ, {"NB_USER": "tnguser"}):
            assert _detect_tnglab_environment() is True

    def test_detect_tnglab_with_wrong_nb_user(self):
        """Test detection with NB_USER set to different value."""
        with patch.dict(os.environ, {"NB_USER": "someotheruser"}):
            assert _detect_tnglab_environment() is False

    def test_detect_tnglab_with_is_tnglab(self):
        """Test detection with IS_TNGLAB environment variable."""
        with patch.dict(os.environ, {"IS_TNGLAB": "1"}):
            assert _detect_tnglab_environment() is True

    def test_detect_tnglab_with_is_tnglab_empty(self):
        """Test detection with IS_TNGLAB set to empty value."""
        with patch.dict(os.environ, {"IS_TNGLAB": ""}):
            assert _detect_tnglab_environment() is True

    def test_detect_tnglab_no_indicators(self):
        """Test when no TNGLab indicators are present."""
        with patch.dict(os.environ, {}, clear=True):
            assert _detect_tnglab_environment() is False


class TestDefaultMemoryLimit:
    """Test default memory limit calculation."""

    def test_get_default_memory_limit_with_psutil(self):
        """Test with psutil available."""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB

        with patch("scida.init.psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value = mock_memory
            result = _get_default_memory_limit()
            # Should be 50% of 16GB = 8GB
            assert result == "8589934592B"  # 8GB in bytes

    def test_get_default_memory_limit_without_psutil(self):
        """Test fallback when psutil not available."""
        with patch("scida.init.PSUTIL_AVAILABLE", False):
            result = _get_default_memory_limit()
            assert result == "4GB"

    def test_get_default_memory_limit_large_system(self):
        """Test with very large system memory."""
        mock_memory = Mock()
        mock_memory.total = 64 * 1024**3  # 64GB

        with patch("scida.init.psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value = mock_memory
            result = _get_default_memory_limit()
            # Should be 50% of 64GB = 32GB
            assert result == "34359738368B"  # 32GB in bytes


class TestInit:
    """Test scida.init_resources() function."""

    def test_init_disabled(self):
        """Test init with no parameters returns None (uses default scheduler)."""
        with patch("scida.init._detect_tnglab_environment", return_value=False):
            result = init_resources()
            assert result is None

    def test_init_no_params_no_tnglab(self):
        """Test init with no parameters outside TNGLab uses default scheduler."""
        with patch("scida.init._detect_tnglab_environment", return_value=False):
            result = init_resources()
            assert result is None

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_n_workers_only(self, mock_client, mock_cluster):
        """Test init with only n_workers specified."""
        with patch("scida.init._detect_tnglab_environment", return_value=False):
            init_resources(n_workers=2)
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            assert kwargs["n_workers"] == 2
            # Should set memory_limit to half system memory
            assert "memory_limit" in kwargs

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_memory_limit_only(self, mock_client, mock_cluster):
        """Test init with only memory_limit specified."""
        with patch("scida.init._detect_tnglab_environment", return_value=False):
            with patch("os.cpu_count", return_value=8):
                init_resources(memory_limit="4GB")
                mock_cluster.assert_called_once()
                args, kwargs = mock_cluster.call_args
                assert kwargs["memory_limit"] == "4GB"
                assert kwargs["n_workers"] == 8  # Should use CPU count

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_tnglab_defaults(self, mock_client, mock_cluster):
        """Test TNGLab environment forces specific defaults."""
        with patch("scida.init._detect_tnglab_environment", return_value=True):
            init_resources()  # No parameters
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            assert kwargs["memory_limit"] == "2GB"
            assert kwargs["n_workers"] == 4

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_tnglab_override(self, mock_client, mock_cluster):
        """Test TNGLab defaults can be overridden."""
        with patch("scida.init._detect_tnglab_environment", return_value=True):
            init_resources(memory_limit="8GB", n_workers=2)
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            assert kwargs["memory_limit"] == "8GB"  # Should keep user-specified value
            assert kwargs["n_workers"] == 2  # Should keep user-specified value

    def test_init_no_dask_distributed(self):
        """Test init when dask.distributed is not available."""
        # Patch sys.modules so that `from dask.distributed import ...` raises ImportError
        with patch.dict(sys.modules, {"dask.distributed": None}):
            result = init_resources(memory_limit="4GB")
            assert result is None

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_success(self, mock_client, mock_cluster):
        """Test successful initialization."""
        # Mock successful cluster creation
        mock_cluster_instance = Mock()
        mock_cluster.return_value = mock_cluster_instance
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        result = init_resources(memory_limit="4GB", n_workers=2)

        # Check that LocalCluster was called with correct arguments
        mock_cluster.assert_called_once()
        args, kwargs = mock_cluster.call_args
        assert kwargs["memory_limit"] == "4GB"
        assert kwargs["n_workers"] == 2
        assert kwargs["threads_per_worker"] == 1

        # Check that Client was created
        mock_client.assert_called_once_with(mock_cluster_instance)
        assert result == mock_client_instance

    @patch("dask.distributed.LocalCluster", side_effect=Exception("Cluster failed"))
    @patch("dask.distributed.Client")
    def test_init_cluster_failure(self, mock_client, mock_cluster):
        """Test handling of cluster creation failure."""
        result = init_resources(memory_limit="4GB")
        assert result is None

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_auto_enable_with_memory_limit(self, mock_client, mock_cluster):
        """Test that distributed is auto-enabled when memory limit is specified."""
        # Even without explicitly setting use_distributed=True,
        # it should be enabled when memory_limit is provided
        init_resources(memory_limit="4GB")
        mock_cluster.assert_called_once()

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_auto_enable_in_tnglab(self, mock_client, mock_cluster):
        """Test that distributed is auto-enabled in TNGLab environment."""
        with patch("scida.init._detect_tnglab_environment", return_value=True):
            init_resources()  # No explicit memory limit or use_distributed
            mock_cluster.assert_called_once()

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_custom_dashboard_port(self, mock_client, mock_cluster):
        """Test custom dashboard port setting."""
        init_resources(memory_limit="4GB", dashboard_port=9999)

        args, kwargs = mock_cluster.call_args
        assert kwargs["dashboard_address"] == ":9999"

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_no_dashboard(self, mock_client, mock_cluster):
        """Test disabling dashboard."""
        init_resources(memory_limit="4GB", dashboard_port=None)

        args, kwargs = mock_cluster.call_args
        assert kwargs["dashboard_address"] is None

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_integer_memory_limit(self, mock_client, mock_cluster):
        """Test init with integer memory_limit (bytes)."""
        init_resources(memory_limit=4_000_000_000, n_workers=2)

        args, kwargs = mock_cluster.call_args
        assert kwargs["memory_limit"] == 4_000_000_000
        assert kwargs["n_workers"] == 2

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_cluster_kwargs_forwarded(self, mock_client, mock_cluster):
        """Test that extra **cluster_kwargs are forwarded to LocalCluster."""
        init_resources(
            memory_limit="4GB", n_workers=2, silence_logs=True, protocol="tcp"
        )

        args, kwargs = mock_cluster.call_args
        assert kwargs["silence_logs"] is True
        assert kwargs["protocol"] == "tcp"

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_explicit_threads_per_worker(self, mock_client, mock_cluster):
        """Test that explicit threads_per_worker overrides the default of 1."""
        init_resources(memory_limit="4GB", n_workers=2, threads_per_worker=4)

        args, kwargs = mock_cluster.call_args
        assert kwargs["threads_per_worker"] == 4


class TestAutoInit:
    """Test _ensure_distributed_if_needed() auto-init behavior."""

    @pytest.fixture(autouse=True)
    def reset_auto_init_flag(self):
        """Reset the module-level _auto_init_done flag before each test."""
        import scida.init as init_module

        init_module._auto_init_done = False
        yield
        init_module._auto_init_done = False

    @patch("scida.init.init_resources")
    @patch("scida.init._detect_tnglab_environment", return_value=True)
    @patch("dask.distributed.get_client", side_effect=ValueError)
    def test_auto_init_on_tnglab(
        self, mock_get_client, mock_detect, mock_init_resources
    ):
        """On TNGLab with no active client, init_resources() should be called."""
        _ensure_distributed_if_needed()
        mock_init_resources.assert_called_once()

    @patch("scida.init.init_resources")
    @patch("scida.init._detect_tnglab_environment", return_value=True)
    @patch("dask.distributed.get_client")
    def test_auto_init_skipped_when_client_exists(
        self, mock_get_client, mock_detect, mock_init_resources
    ):
        """When a distributed client already exists, init_resources() should NOT be called."""
        mock_get_client.return_value = Mock()
        _ensure_distributed_if_needed()
        mock_init_resources.assert_not_called()

    @patch("scida.init.init_resources")
    @patch("scida.init._detect_tnglab_environment", return_value=False)
    @patch("dask.distributed.get_client", side_effect=ValueError)
    def test_auto_init_informs_outside_tnglab(
        self, mock_get_client, mock_detect, mock_init_resources
    ):
        """Outside TNGLab with no client, should log debug hint but NOT call init_resources()."""
        with patch("scida.init.log") as mock_log:
            _ensure_distributed_if_needed()
        mock_init_resources.assert_not_called()
        # Check that a debug-level message was logged with the hint
        mock_log.debug.assert_called()
        logged_msg = mock_log.debug.call_args[0][0]
        assert "init_resources" in logged_msg

    @patch("scida.init.init_resources")
    @patch("scida.init._detect_tnglab_environment", return_value=True)
    @patch("dask.distributed.get_client", side_effect=ValueError)
    def test_auto_init_runs_only_once(
        self, mock_get_client, mock_detect, mock_init_resources
    ):
        """Calling _ensure_distributed_if_needed() twice should only init once."""
        _ensure_distributed_if_needed()
        _ensure_distributed_if_needed()
        mock_init_resources.assert_called_once()

    @patch("scida.init.init_resources")
    @patch("scida.init._detect_tnglab_environment", return_value=True)
    def test_auto_init_handles_missing_distributed(
        self, mock_detect, mock_init_resources
    ):
        """When dask.distributed is not installed, should not raise."""
        with patch.dict(sys.modules, {"dask.distributed": None}):
            _ensure_distributed_if_needed()
        # ImportError from get_client means no client → should try to init on TNGLab
        mock_init_resources.assert_called_once()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestInitIntegration:
    """Integration tests with real dask distributed clusters."""

    def test_histogram2d_with_memory_limited_cluster(self):
        """Test that da.histogram2d works with a memory-limited cluster.

        Creates a cluster with strict per-worker memory limits, verifies
        the limits are applied, then runs histogram2d on data that exceeds
        a single worker's memory to confirm chunked processing works.
        """
        import dask.array as da
        import numpy as np
        from dask.distributed import Client, LocalCluster

        memory_limit_mb = 200
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
            h_dask = h_dask.compute()

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

        memory_limit_mb = 200
        memory_limit_bytes = memory_limit_mb * 10**6
        client = None
        try:
            client = init_resources(
                memory_limit=f"{memory_limit_mb}MB",
                n_workers=2,
                dashboard_port=None,
            )
            assert client is not None

            # Verify the cluster is functional
            result = da.ones(1000).sum().compute()
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
