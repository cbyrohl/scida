import yaml

from astrodask.config import get_config
from astrodask.convenience import load
from astrodask.interface import Dataset
from astrodask.misc import check_config_for_dataset
from tests.testdata_properties import require_testdata, require_testdata_path

# eventually replaces 'testdata_path'
# def test_load_pathdiscovery():
#    obj = load("TNG50-4_snapshot")
#    assert obj.file is not None
#    assert obj.data is not None


def test_load_https():
    # TNG50-4 snapshot
    # url = "https://heibox.uni-heidelberg.de/f/dc65a8c75220477eb62d/?dl=1"
    # TNG50-4 group
    url = "https://heibox.uni-heidelberg.de/f/ff27fb6975fb4dc391ef/?dl=1"
    obj = load(url)
    assert obj.file is not None
    assert obj.data is not None


def test_searchpath():
    foldername = "TNG50-4_snapshot"
    obj = load(foldername)
    assert obj.file is not None
    assert obj.data is not None


def test_searchpath_withcatalog():
    foldername = "TNG50-4_snapshot"
    foldername_catalog = "TNG50-4_group"
    obj = load(foldername, catalog=foldername_catalog)
    assert obj.file is not None


@require_testdata_path("interface")
def test_load(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
    assert obj.data is not None


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_resource(tmp_path, monkeypatch, testdatapath):
    p = tmp_path / "conf.yaml"
    config = dict(resources=dict(dummyresource=dict(dataset=dict(path=testdatapath))))
    with open(p, "w") as file:
        yaml.dump(config, file)
    monkeypatch.setenv("ASTRODASK_CONFIG_PATH", str(p))
    get_config(reload=True)
    ds = load("dummyresource://dataset")
    assert isinstance(ds, Dataset)


# TODO: Need to write test for TNG50-4_snapshot+TNG50-4_group using require_testdata(..., only=...)
@require_testdata_path("interface", only=["Illustris-3_snapshot"])
def test_load_areposnap(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
    assert obj.data is not None
    assert all([k in obj.data for k in ["PartType%i" % i for i in [0, 1, 3, 4, 5]]])
    assert "Coordinates" in obj.data["PartType0"]
    # assert obj.header["BoxSize"] == 35000.0


@require_testdata_path("series", only=["TNGvariation_simulation"])
def test_load_areposim(testdatapath):
    obj = load(testdatapath, units="cgs")
    assert obj is not None


@require_testdata_path("interface", exclude=["gaia"], exclude_substring=True)
def test_load_units_codeunits(testdatapath):
    obj = load(testdatapath, units="code")
    assert "UnitMixin" in type(obj).__name__


@require_testdata_path("interface", exclude=["gaia"], exclude_substring=True)
def test_load_units_cgs(testdatapath):
    obj = load(testdatapath, units="cgs")
    assert "UnitMixin" in type(obj).__name__


@require_testdata_path("interface", only=["EAGLEsmall_snapshot"])
def test_load_units_manualpath(testdatapath):
    obj = load(testdatapath, units="cgs", unitfile="units/eagle.yaml")
    assert "UnitMixin" in type(obj).__name__


@require_testdata("interface", only=["TNG50-4_snapshot", "Illustris-3_snapshot"])
def test_guess_nameddataset(testdata_interface):
    snp = testdata_interface
    metadata = snp._metadata_raw
    candidates = check_config_for_dataset(metadata, path=snp.path)
    assert len(candidates) == 1
    print("cds", candidates)


@require_testdata_path("interface", only=["gaia-dr3_minimal"])
def test_gaia_hdf5(testdatapath):
    ds = load(testdatapath, units=False)
    assert ds.__class__ == Dataset  # using most abstract class in load()


@require_testdata_path("interface", only=["gaia-dr3_minimal"])
def test_gaia_hdf5_withunits(testdatapath):
    ds = load(testdatapath, units=True)
    assert ds is not None
    print(ds.data["ra"].units)


# check that zarr cutouts from TNG have uids?
@require_testdata_path("interface", only=["tng_halocutout_zarr"])
def test_load_zarrcutout(testdatapath):
    obj = load(testdatapath)
    assert hasattr(obj, "header")
    assert obj.data["Group"]["uid"][0] == 0
