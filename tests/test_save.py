import pathlib

from scida import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_save(testdatapath, tmp_path):
    p = str(pathlib.Path(tmp_path) / "test.zarr")
    ds = load(testdatapath, units=False)
    ds.save(p)
    ds = load(p)
    assert ds is not None


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_save_withunits(testdatapath, tmp_path):
    p = str(pathlib.Path(tmp_path) / "test.zarr")
    ds = load(testdatapath, units=True)
    ds.save(p)
    ds = load(p)
    assert ds is not None
