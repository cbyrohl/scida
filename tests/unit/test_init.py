"""
Unit tests for scida.init_resources()
"""

import os
import sys
from unittest.mock import Mock, patch

import dask
import pytest

from scida.init import (
    _detect_tnglab_environment,
    _ensure_distributed_if_needed,
    _get_default_memory_limit,
    init_resources,
)
from scida.misc import parse_size

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolate_resource_config(monkeypatch):
    import scida.init as init_module

    monkeypatch.delenv("IS_TNGLAB", raising=False)
    monkeypatch.delenv("NB_USER", raising=False)
    monkeypatch.setattr(init_module, "_auto_init_done", False)
    with dask.config.set(scheduler=None, num_workers=None):
        yield


class TestMemoryParsing:
    """Test memory string parsing functions."""

    def test_parse_size_basic(self):
        """Test parsing with the existing parse_size function."""
        assert parse_size("1GIB") == 1024**3
        assert parse_size("1.5GIB") == int(1.5 * 1024**3)
        assert parse_size("500MIB") == 500 * 1024**2

    def test_parse_size_decimal_units(self):
        """Test parsing decimal (SI) units: GB, MB, KB, TB, B."""
        assert parse_size("4GB") == 4 * 10**9
        assert parse_size("100MB") == 100 * 10**6
        assert parse_size("512KB") == 512 * 10**3
        assert parse_size("2TB") == 2 * 10**12
        assert parse_size("1024B") == 1024

    def test_parse_size_fractional_decimal(self):
        """Test fractional values with decimal units."""
        assert parse_size("1.5GB") == int(1.5 * 10**9)
        assert parse_size("2.5MB") == int(2.5 * 10**6)


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

    @pytest.mark.parametrize(
        "total_memory, workers",
        [(16 * 1024**3, 4), (10 * 1024**3, 3), (2 * 1024**3, 4)],
    )
    def test_full_detected_budget_is_shared(self, total_memory, workers):
        """Share the effective host/container limit without reserving a fraction."""
        with patch("distributed.system.memory_limit", return_value=total_memory):
            per_worker = _get_default_memory_limit(workers)

        assert per_worker == total_memory // workers
        assert 0 <= total_memory - per_worker * workers < workers

    def test_insufficient_memory_does_not_disable_limits(self):
        """A zero-byte Dask limit would disable memory management."""
        with patch("distributed.system.memory_limit", return_value=1):
            with pytest.raises(ValueError, match="too small"):
                _get_default_memory_limit(2)


class TestInit:
    """Test scida.init_resources() function."""

    @patch("dask.distributed.LocalCluster")
    @patch("scida.init.cpu_count", return_value=6)
    def test_init_no_params_configures_local_threads(self, mock_cpus, mock_cluster):
        result = init_resources()
        assert result is None
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 6
        mock_cluster.assert_not_called()

    @patch("dask.distributed.LocalCluster")
    def test_init_local_thread_count(self, mock_cluster):
        init_resources(n_workers=3)
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 3
        mock_cluster.assert_not_called()

    @pytest.mark.parametrize("use_distributed", [None, False])
    def test_local_memory_limit_requires_explicit_distributed(self, use_distributed):
        with pytest.raises(
            ValueError, match="memory_limit requires use_distributed=True"
        ):
            init_resources(memory_limit="4GB", use_distributed=use_distributed)
        assert dask.config.get("scheduler") is None

    @pytest.mark.parametrize(
        "options", [{"threads_per_worker": 2}, {"processes": False}]
    )
    def test_local_rejects_cluster_options(self, options):
        with pytest.raises(ValueError, match="require use_distributed=True"):
            init_resources(**options)

    @pytest.mark.parametrize("parameter", ["n_workers", "threads_per_worker"])
    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_invalid_worker_counts(self, parameter, value):
        with pytest.raises(ValueError, match="positive integer"):
            init_resources(use_distributed=True, **{parameter: value})

    @pytest.mark.parametrize("value", ["false", 0, 1])
    def test_invalid_distributed_flag(self, value):
        with pytest.raises(ValueError, match="use_distributed must be"):
            init_resources(use_distributed=value)

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    @patch("scida.init.cpu_count", return_value=4)
    @patch("distributed.system.memory_limit", return_value=16 * 1024**3)
    def test_distributed_detected_defaults(
        self, mock_memory, mock_cpus, mock_client, mock_cluster
    ):
        result = init_resources(use_distributed=True)
        assert result is mock_client.return_value
        assert mock_cluster.call_args.kwargs["n_workers"] == 4
        assert mock_cluster.call_args.kwargs["threads_per_worker"] == 1
        assert mock_cluster.call_args.kwargs["memory_limit"] == 4 * 1024**3

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    @patch("scida.init.cpu_count", return_value=8)
    @patch("distributed.system.memory_limit", return_value=8 * 1024**3)
    def test_default_workers_account_for_threads(
        self, mock_memory, mock_cpus, mock_client, mock_cluster
    ):
        init_resources(use_distributed=True, threads_per_worker=4)
        assert mock_cluster.call_args.kwargs["n_workers"] == 2
        assert mock_cluster.call_args.kwargs["memory_limit"] == 4 * 1024**3

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_n_workers_only(self, mock_client, mock_cluster):
        """Test init with only n_workers specified."""
        with patch("scida.init._detect_tnglab_environment", return_value=False):
            init_resources(use_distributed=True, n_workers=2)
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            assert kwargs["n_workers"] == 2
            # The complete detected memory budget is shared among workers
            assert "memory_limit" in kwargs

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_memory_limit_only(self, mock_client, mock_cluster):
        """Test init with only memory_limit specified."""
        with patch("scida.init._detect_tnglab_environment", return_value=False):
            with patch("scida.init.cpu_count", return_value=8):
                init_resources(use_distributed=True, memory_limit="4GB")
                mock_cluster.assert_called_once()
                args, kwargs = mock_cluster.call_args
                assert kwargs["memory_limit"] == "4GB"
                assert kwargs["n_workers"] == 8  # Should use CPU count

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_tnglab_defaults(self, mock_client, mock_cluster):
        """Test TNGLab environment selects specific defaults."""
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
            init_resources(use_distributed=True, memory_limit="8GB", n_workers=2)
            mock_cluster.assert_called_once()
            args, kwargs = mock_cluster.call_args
            assert kwargs["memory_limit"] == "8GB"  # Should keep user-specified value
            assert kwargs["n_workers"] == 2  # Should keep user-specified value

    @patch("scida.init._detect_tnglab_environment", return_value=True)
    @patch("dask.distributed.LocalCluster")
    def test_tnglab_explicit_local_mode_is_respected(self, mock_cluster, mock_detect):
        init_resources(use_distributed=False, n_workers=2)
        _ensure_distributed_if_needed()
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 2
        mock_cluster.assert_not_called()

    def test_local_mode_does_not_require_distributed(self):
        with patch.dict(sys.modules, {"dask.distributed": None}):
            init_resources(n_workers=2)
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 2

    def test_init_no_dask_distributed(self):
        """Test init when dask.distributed is not available."""
        # Patch sys.modules so that `from dask.distributed import ...` raises ImportError
        with patch.dict(sys.modules, {"dask.distributed": None}):
            result = init_resources(use_distributed=True, memory_limit="4GB")
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

        result = init_resources(use_distributed=True, memory_limit="4GB", n_workers=2)

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
        result = init_resources(use_distributed=True, memory_limit="4GB")
        assert result is None

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_explicit_distributed_with_memory_limit(
        self, mock_client, mock_cluster
    ):
        """An explicit opt-in starts distributed workers with the requested limit."""
        init_resources(use_distributed=True, memory_limit="4GB")
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
        init_resources(use_distributed=True, memory_limit="4GB", dashboard_port=9999)

        args, kwargs = mock_cluster.call_args
        assert kwargs["dashboard_address"] == ":9999"

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_no_dashboard(self, mock_client, mock_cluster):
        """Test disabling dashboard."""
        init_resources(use_distributed=True, memory_limit="4GB", dashboard_port=None)

        args, kwargs = mock_cluster.call_args
        assert kwargs["dashboard_address"] is None

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_integer_memory_limit(self, mock_client, mock_cluster):
        """Test init with integer memory_limit (bytes)."""
        init_resources(use_distributed=True, memory_limit=4_000_000_000, n_workers=2)

        args, kwargs = mock_cluster.call_args
        assert kwargs["memory_limit"] == 4_000_000_000
        assert kwargs["n_workers"] == 2

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_cluster_kwargs_forwarded(self, mock_client, mock_cluster):
        """Test that extra **cluster_kwargs are forwarded to LocalCluster."""
        init_resources(
            use_distributed=True,
            memory_limit="4GB",
            n_workers=2,
            silence_logs=True,
            protocol="tcp",
        )

        args, kwargs = mock_cluster.call_args
        assert kwargs["silence_logs"] is True
        assert kwargs["protocol"] == "tcp"

    @patch("dask.distributed.LocalCluster")
    @patch("dask.distributed.Client")
    def test_init_explicit_threads_per_worker(self, mock_client, mock_cluster):
        """Test that explicit threads_per_worker overrides the default of 1."""
        init_resources(
            use_distributed=True, memory_limit="4GB", n_workers=2, threads_per_worker=4
        )

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
