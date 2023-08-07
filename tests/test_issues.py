# tests for gh issues. to be cleaned up and moved to the right place eventually
from scida import load
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
def test_issue_63(testdatapath):
    ds = load(testdatapath, units=True)

    gas = ds.data["PartType0"]
    coords = gas["Coordinates"]
    assert coords.to_base_units().units == ds.ureg("centimeter")
    grp = ds.data["Group"]
    coords = grp["GroupPos"]
    assert coords.to_base_units().units == ds.ureg("centimeter")
