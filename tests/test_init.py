"""
Tests for scida.init() function
"""

import os
from unittest.mock import Mock, patch

from scida.helpers_misc import parse_humansize
from scida.init import (
    _detect_tnglab_environment,
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

        with patch("psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value = mock_memory
            result = _get_default_memory_limit()
            # Should be 50% of 16GB = 8GB, which is at the cap
            assert result == "8589934592B"  # 8GB in bytes

    def test_get_default_memory_limit_without_psutil(self):
        """Test fallback when psutil not available."""
        with patch.dict("sys.modules", {"psutil": None}):
            result = _get_default_memory_limit()
            assert result == "4GB"

    def test_get_default_memory_limit_large_system(self):
        """Test with very large system memory."""
        mock_memory = Mock()
        mock_memory.total = 64 * 1024**3  # 64GB

        with patch("psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value = mock_memory
            result = _get_default_memory_limit()
            # Should be capped at 8GB per worker
            assert result == "8589934592B"  # 8GB in bytes


class TestInit:
    """Test scida.init_resources() function."""

    def test_init_disabled(self):
        """Test init with distributed computing disabled."""
        result = init_resources(use_distributed=False)
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
            assert kwargs["memory_limit"] == "4GB"
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

    @patch("dask.distributed.Client", side_effect=ImportError)
    def test_init_no_dask_distributed(self, mock_client):
        """Test init when dask.distributed is not available."""
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
