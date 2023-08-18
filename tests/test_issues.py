# tests for gh issues. to be cleaned up and moved to the right place eventually
import pathlib

import pytest

from scida import ArepoSimulation, load
from scida.convenience import find_path
from tests.helpers import write_gadget_testfile
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_issue_59(testdatapath):
    # current testing of interface did not include units=True yet.
    obj = load(testdatapath, units=False)
    _ = obj.return_data(haloID=42)
    # for units=True, this would be a pint.Quantity, so we need to check for that
    # in the Selector
    obj = load(testdatapath, units=True)
    _ = obj.return_data(haloID=42)


@require_testdata_path("interface", only=["TNG100-3_snapshot_z0_minimal"])
def test_issue_63(testdatapath, tmp_path):
    # underlying problem of issue 63 was that TNG100-2/TNG100-3 was not identified
    # subsequently, the group catalog had no units
    # identification of TNG100-2/TNG100-3 was broken but already fixed on main by now
    # however, we want to
    # 1. change "dimensionless" to "unknown" for fields with unknown units.
    # 2. IDs should not be pint quantities

    # check that units work for TNG100-3
    ds = load(testdatapath, units=True)
    gas = ds.data["PartType0"]
    coords = gas["Coordinates"]
    assert coords.to_base_units().units == ds.ureg("centimeter")
    grp = ds.data["Group"]
    coords = grp["GroupPos"]
    assert coords.to_base_units().units == ds.ureg("centimeter")

    # assert that IDs are not pint quantities
    assert not hasattr(ds.data["PartType0"]["ParticleIDs"], "units")

    # assert "unknown" units when no units known
    p = pathlib.Path(tmp_path) / "test.hdf5"
    write_gadget_testfile(p)
    ds = load(p, units=True)
    assert ds.data["PartType0"]["Coordinates"].units == "unknown"


@require_testdata_path("series", only=["TNGvariation_simulation"])
def test_issue_88(testdatapath, tmp_path):
    # pass "output" folder instead of base folder of simulation
    srs = load(testdatapath, units=True)
    assert isinstance(srs, ArepoSimulation)

    testdatapath += "/output"
    srs = load(testdatapath, units=True)
    assert isinstance(srs, ArepoSimulation)

    # make sure ~ is resolved as home folder
    path = "~/test"
    assert find_path(path).startswith("/")


def test_issue_78():
    # some of the default "datafolders" in config.yaml do not exist on test user
    # check that is does not matter
    path = "SomeDataset"
    with pytest.raises(ValueError) as e:
        # this dataset does not exist, so should still raise proper exception
        find_path(path)
    assert "unknown" in str(e.value)
