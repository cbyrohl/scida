import math
import os
import pathlib
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import h5py
import pytest
import requests
import yaml

from scida.config import get_config
from scida.convenience import _download, load
from scida.interface import Dataset
from scida.misc import check_config_for_dataset
from tests.testdata_properties import require_testdata, require_testdata_path


@pytest.mark.external
@require_testdata_path("interface")
def test_load(testdatapath):
    obj = load(testdatapath)
    print("type: ", type(obj))
    assert obj.file is not None
    assert obj.data is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_resource(tmp_path, monkeypatch, testdatapath):
    p = tmp_path / "conf.yaml"
    config = dict(resources=dict(dummyresource=dict(dataset=dict(path=testdatapath))))
    with open(p, "w") as file:
        yaml.dump(config, file)
    monkeypatch.setenv("SCIDA_CONFIG_PATH", str(p))
    get_config(reload=True)
    ds = load("dummyresource://dataset")
    assert isinstance(ds, Dataset)


@pytest.mark.external
@require_testdata_path("interface", only=["Illustris-3_snapshot"])
def test_load_areposnap(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
    assert obj.data is not None
    assert all([k in obj.data for k in ["PartType%i" % i for i in [0, 1, 3, 4, 5]]])
    assert "Coordinates" in obj.data["PartType0"]


@pytest.mark.external
@require_testdata_path("series", only=["TNGvariation_simulation"])
def test_load_areposim(testdatapath):
    obj = load(testdatapath, units="cgs")
    assert obj is not None


@pytest.mark.external
@require_testdata_path("interface", exclude=["gaia"], exclude_substring=True)
def test_load_units_codeunits(testdatapath):
    obj = load(testdatapath, units="code")
    assert "UnitMixin" in type(obj).__name__


@pytest.mark.external
@require_testdata_path("interface", exclude=["gaia"], exclude_substring=True)
def test_load_units_cgs(testdatapath):
    obj = load(testdatapath, units="cgs")
    assert "UnitMixin" in type(obj).__name__


@pytest.mark.external
@require_testdata_path("interface", only=["EAGLEsmall_snapshot"])
def test_load_units_manualpath(testdatapath):
    obj = load(testdatapath, units="cgs", unitfile="units/eagle.yaml")
    assert "UnitMixin" in type(obj).__name__


@pytest.mark.external
@require_testdata("interface", only=["TNG50-4_snapshot", "Illustris-3_snapshot"])
def test_guess_nameddataset(testdata_interface):
    snp = testdata_interface
    metadata = snp._metadata_raw
    candidates = check_config_for_dataset(metadata, path=snp.path)
    assert len(candidates) == 1
    print("cds", candidates)


@pytest.mark.external
@require_testdata_path("interface", only=["gaia-dr3_minimal"])
def test_gaia_hdf5(testdatapath):
    ds = load(testdatapath, units=False)
    assert ds.__class__ == Dataset


@pytest.mark.external
@require_testdata_path("interface", only=["gaia-dr3_minimal"])
def test_gaia_hdf5_withunits(testdatapath):
    ds = load(testdatapath, units=True)
    assert ds is not None
    print(ds.data["ra"].units)


@pytest.mark.external
@require_testdata_path("interface", only=["tng_halocutout_zarr"])
def test_load_zarrcutout(testdatapath):
    obj = load(testdatapath)
    assert hasattr(obj, "header")
    assert obj.data["Group"]["uid"][0] == 0


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_cachefail(cachedir, testdatapath, caplog):
    """Test recovery from failure during cache creation."""
    import signal
    import time

    class TimeoutException(Exception):
        pass

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out.")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    loadkwargs = dict(catalog="none")
    dt = 0.3
    dt_int = int(math.ceil(dt))
    with pytest.raises(Exception):
        with time_limit(dt_int):
            time.sleep(dt_int - dt)
            load(testdatapath, **loadkwargs)

    nfiles = 0
    for _, _, files in os.walk(cachedir):
        nfiles += len(files)
    assert nfiles == 0

    ds = load(testdatapath, **loadkwargs)
    nfiles = 0
    for root, dirs, files in os.walk(cachedir):
        nfiles += len(files)
    assert nfiles > 0

    fp = ds.file.filename
    ds.file.close()
    with h5py.File(fp, "r+") as f:
        del f.attrs["_cachingcomplete"]
    load(testdatapath, **loadkwargs)
    assert "Invalid cache file, attempting to create new one." in caplog.text


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_cachefail_oserror(cachedir, testdatapath, caplog):
    """Test recovery from failure during cache creation. Do so by keeping the file is kept open."""
    loadkwargs = dict(catalog="none")
    ds = load(testdatapath, **loadkwargs)
    filename = ds.file.filename
    ds.file.close()
    with open(filename, "r+b") as f:
        f.seek(0)
        f.write(b"destroyheader")
    ds = load(testdatapath, **loadkwargs)
    print(ds.info())


@pytest.mark.integration
@pytest.mark.network
def test_load_https():
    url = "https://github.com/cbyrohl/scida/releases/download/testdata-v1/minimal_TNG50-4_snapshot_z0.hdf5"
    obj = load(url)
    assert obj.file is not None
    assert obj.data is not None


def _make_mock_response(status_code=200, content=b"data"):
    """Create a mock response that works as a context manager."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = {"content-length": str(len(content))}
    resp.iter_content.return_value = [content]
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)

    if status_code >= 400:
        http_error = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_error
    else:
        resp.raise_for_status.return_value = None

    return resp


class TestDownloadRetry:
    """Tests for _download retry logic."""

    @patch("scida.convenience.time.sleep")
    @patch("scida.convenience.requests.get")
    def test_retry_on_502(self, mock_get, mock_sleep, tmp_path):
        """Retries on 502 and succeeds on second attempt."""
        fail_resp = _make_mock_response(502)
        ok_resp = _make_mock_response(200, content=b"file content")
        mock_get.side_effect = [fail_resp, ok_resp]

        target = tmp_path / "output.hdf5"
        _download("http://example.com/file", target, progressbar=False)

        assert target.read_bytes() == b"file content"
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch("scida.convenience.time.sleep")
    @patch("scida.convenience.requests.get")
    def test_retry_exhausted_raises(self, mock_get, mock_sleep, tmp_path):
        """Raises after all retries are exhausted."""
        fail_resp = _make_mock_response(502)
        mock_get.return_value = fail_resp

        target = tmp_path / "output.hdf5"
        with pytest.raises(requests.exceptions.HTTPError):
            _download("http://example.com/file", target, progressbar=False)

        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("scida.convenience.time.sleep")
    @patch("scida.convenience.requests.get")
    def test_no_retry_on_404(self, mock_get, mock_sleep, tmp_path):
        """Client errors (4xx) are not retried."""
        fail_resp = _make_mock_response(404)
        mock_get.return_value = fail_resp

        target = tmp_path / "output.hdf5"
        with pytest.raises(requests.exceptions.HTTPError):
            _download("http://example.com/file", target, progressbar=False)

        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @patch("scida.convenience.time.sleep")
    @patch("scida.convenience.requests.get")
    def test_retry_on_connection_error(self, mock_get, mock_sleep, tmp_path):
        """Retries on ConnectionError and succeeds."""
        ok_resp = _make_mock_response(200, content=b"ok")
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("connection reset"),
            ok_resp,
        ]

        target = tmp_path / "output.hdf5"
        _download("http://example.com/file", target, progressbar=False)

        assert target.read_bytes() == b"ok"
        assert mock_get.call_count == 2

    @patch("scida.convenience.time.sleep")
    @patch("scida.convenience.requests.get")
    def test_retry_on_timeout(self, mock_get, mock_sleep, tmp_path):
        """Retries on Timeout and succeeds."""
        ok_resp = _make_mock_response(200, content=b"ok")
        mock_get.side_effect = [
            requests.exceptions.Timeout("timed out"),
            ok_resp,
        ]

        target = tmp_path / "output.hdf5"
        _download("http://example.com/file", target, progressbar=False)

        assert target.read_bytes() == b"ok"
        assert mock_get.call_count == 2

    @patch("scida.convenience.time.sleep")
    @patch("scida.convenience.requests.get")
    def test_cleans_partial_download_on_retry(self, mock_get, mock_sleep, tmp_path):
        """Partial files are removed before retrying."""
        target = tmp_path / "output.hdf5"
        target.write_bytes(b"partial")

        ok_resp = _make_mock_response(200, content=b"complete")
        fail_resp = _make_mock_response(502)
        mock_get.side_effect = [fail_resp, ok_resp]

        _download("http://example.com/file", target, progressbar=False, overwrite=True)

        assert target.read_bytes() == b"complete"

    def test_existing_file_no_overwrite_raises(self, tmp_path):
        """Raises ValueError if target exists and overwrite=False."""
        target = tmp_path / "output.hdf5"
        target.write_bytes(b"existing")

        with pytest.raises(ValueError, match="already exists"):
            _download("http://example.com/file", target, progressbar=False)
