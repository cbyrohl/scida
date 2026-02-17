import math
import os
from contextlib import contextmanager

import h5py
import pytest
import yaml

from scida.config import get_config
from scida.convenience import load
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
