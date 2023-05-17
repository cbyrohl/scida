import pathlib

import numpy as np
import pytest

from astrodask import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_redshift_mismatch(testdatapath):
    # this should work
    grp_path = testdatapath.replace("snapshot", "group")
    load(testdatapath, catalog=grp_path)

    # this should fail
    grp_path = grp_path.replace("z3", "z0")
    with pytest.raises(Exception) as exc_info:
        load(testdatapath, catalog=grp_path)
    assert str(exc_info.value).startswith("Redshift mismatch")


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_scalefactor_in_unitreg(testdatapath):
    ds = load(testdatapath, units=True)
    a = ds.ureg("a").to_base_units().magnitude
    h = ds.ureg("h").to_base_units().magnitude
    assert a == pytest.approx(1.0 / (1.0 + ds.redshift))
    assert h == pytest.approx(0.6774)


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_scalefactor_in_units(testdatapath):
    ds_nounits = load(testdatapath, units=False)
    ds = load(testdatapath, units=True)
    a = ds.ureg("a").to_base_units().magnitude
    h = ds.ureg("h").to_base_units().magnitude

    # check for velocities (only has "a" dependency)
    vel = ds.data["PartType0"]["Velocities"][0, 0].compute()
    vel_nounits = ds_nounits.data["PartType0"]["Velocities"][0, 0].compute()
    vel_cgs = vel.to_base_units().magnitude
    vel_nounits_cgs = 1e5 * np.sqrt(a) * vel_nounits
    assert vel_cgs == pytest.approx(vel_nounits_cgs)

    # check for positions (has both "a" and "h" dependency)
    pos = ds.data["PartType0"]["Coordinates"][0, 0].compute()
    pos_nounits = ds_nounits.data["PartType0"]["Coordinates"][0, 0].compute()
    pos_cgs = pos.to_base_units().magnitude
    pos_nounits_cgs = a * ds.ureg("kpc").to("cm").magnitude * pos_nounits / h
    assert pos_cgs == pytest.approx(pos_nounits_cgs)


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_discover_groupcatalog(testdatapath):
    ds = load(testdatapath)
    assert ds.catalog is not None
    print(ds.catalog)


@require_testdata_path("series", only=["TNG50-4"])
def test_discover_groupcatalog2(testdatapath):
    # same but with actual paths as in TNG
    p = pathlib.Path(testdatapath)
    path = p / "output" / "snapdir_099" / "snap_099.0.hdf5"
    ds = load(path)
    assert ds.catalog is not None


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_aliases(testdatapath):
    ds = load(testdatapath)
    gas = ds.data["PartType0"]
    with pytest.raises(KeyError):
        print(gas["TestAlias"])
    gas.add_alias("TestAlias", "Coordinates")
    assert gas["TestAlias"] is not None
    # lets check the SpatialMixin alias as well
    assert ds.data["Group"]["Coordinates"] is not None


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_aliases_bydefault(testdatapath):
    ds = load(testdatapath)
    d = ds.data
    assert d["PartType0"] == d["gas"]
    assert d["PartType1"] == d["dm"]
    if "PartType2" in d:
        assert d["PartType2"] == d["lowres"]
    assert d["PartType3"] == d["tracers"]
    assert d["PartType4"] == d["stars"]
    assert d["PartType5"] == d["bh"]


@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_additionalfields(testdatapath):
    ds = load(testdatapath)
    gas = ds.data["PartType0"]
    assert "Temperature" in gas
    assert gas["Temperature"] is not None
    assert gas["Temperature"].compute() is not None
    # TODO: Test how newly defined fields play with units


@require_testdata_path("interface", only=["TNG50-4_snapshot_z127"])
def test_emptycatalog(testdatapath):
    ds = load(testdatapath)
    print(ds)
    pass
