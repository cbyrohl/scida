import astropy.units as u
import pytest
from astropy.cosmology import FlatLambdaCDM

from scida import SwiftSnapshot, load
from tests.testdata_properties import require_testdata_path


def check_flamingosnap(obj: SwiftSnapshot, obj_wu: SwiftSnapshot):
    assert isinstance(obj, SwiftSnapshot)
    assert all(
        [k in obj.data for k in ["PartType0", "PartType4", "PartType5", "PartType6"]]
    )

    z = obj._metadata_raw["/Header"]["Redshift"]
    a = 1.0 / (1.0 + z)
    cosmology: FlatLambdaCDM = obj.cosmology
    assert pytest.approx(cosmology.h) == 0.681

    # unit length: 3.08568e+24 cm  (1Mpc)

    # test units
    pdata = obj.data["PartType0"]
    pdata_wu = obj_wu.data["PartType0"]
    # Coordinates
    uq = (pdata["Coordinates"][0, 0].compute()).to("cm").magnitude
    uq_ref = pdata_wu["Coordinates"][0, 0].compute() * a * (1.0 * u.Mpc).to("cm").value
    assert pytest.approx(uq) == uq_ref


@require_testdata_path("interface", only=["FLAMINGO_snapshot_reduced"])
def test_snapshot(testdatapath):
    """Test loading of a full snapshot"""
    obj: SwiftSnapshot = load(testdatapath, units=True)
    obj_wu: SwiftSnapshot = load(testdatapath, units=False)
    print(obj.info())
    check_flamingosnap(obj, obj_wu)
