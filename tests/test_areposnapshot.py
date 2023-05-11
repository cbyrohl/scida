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
