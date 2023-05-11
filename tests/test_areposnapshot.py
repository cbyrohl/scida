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
