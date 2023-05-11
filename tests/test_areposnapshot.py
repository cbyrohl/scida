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
