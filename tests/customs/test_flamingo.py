import astropy.units as u
import pytest
from astropy.cosmology import FlatLambdaCDM

from scida import load
from scida.customs.swift.dataset import SwiftSnapshot
from scida.customs.swift.series import SwiftSimulation
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

    cgs_convfactors = {}
    cgs_convfactors["Coordinates"] = a * (1.0 * u.Mpc).to("cm").value  # from cMpc
    cgs_convfactors["Velocities"] = (1.0 * u.km / u.s).to("cm/s").value  # from km/s
    cgs_convfactors["Masses"] = (1e10 * u.Msun).to("g").value  # from 1e10 Msun
    cgs_convfactors["Temperatures"] = 1.0  # from K
    ureg = obj.ureg
    cgs_units = {}
    cgs_units["Coordinates"] = ureg.cm
    cgs_units["Velocities"] = ureg.cm / ureg.s
    cgs_units["Masses"] = ureg.g
    cgs_units["Temperatures"] = ureg.K

    for k, f in cgs_convfactors.items():
        print("Verifying unit conversion for", k)
        v = (pdata[k][0].compute()).to_base_units()
        uq = v.magnitude
        uq_ref = pdata_wu[k][0].compute() * f
        assert pytest.approx(uq) == uq_ref
        assert v.units == cgs_units[k]

    for ptype, d in obj.data.items():
        for k in d.keys():
            if not k.endswith("IDs"):
                continue
            assert not hasattr(d[k], "units")  # we do not want units for IDs


@require_testdata_path(
    "interface", only=["FLAMINGO_snapshot_reduced", "FLAMINGO_snapshot_minimal"]
)
def test_snapshot(testdatapath):
    """Test loading of a full snapshot"""
    obj: SwiftSnapshot = load(testdatapath, units=True)
    obj_wu: SwiftSnapshot = load(testdatapath, units=False)
    obj.info()
    check_flamingosnap(obj, obj_wu)


@require_testdata_path("series", only=["FLAMINGO_simulation_minimal"])
def test_series(testdatapath):
    obj: SwiftSimulation = load(testdatapath, units=True)
    obj.info()
    assert isinstance(obj, SwiftSimulation)
    z = 4.75
    ds = obj.get_dataset(redshift=z)
    assert isinstance(ds, SwiftSnapshot)
    assert ds.redshift == z
