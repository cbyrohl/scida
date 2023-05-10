import numpy as np
from pint import UnitRegistry

from astrodask.convenience import load
from astrodask.interfaces.mixins.units import update_unitregistry
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_cgs(testdatapath):
    ds = load(testdatapath, units="cgs")
    # test units of some fields
    ptype = "PartType0"
    gas = ds.data[ptype]
    coords = gas["Coordinates"]
    dens = gas["Density"]
    sfr = gas["StarFormationRate"]
    u = ds.ureg
    assert coords.units == u.cm
    assert dens.units == u.g / u.cm**3
    assert sfr.units == u.g / u.s


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_codeunits(testdatapath):
    ds = load(testdatapath, units=True)
    # test units of some fields
    ptype = "PartType0"
    gas = ds.data[ptype]
    coords = gas["Coordinates"]
    dens = gas["Density"]
    sfr = gas["StarFormationRate"]
    u = ds.ureg
    assert coords.units == u.code_length
    assert dens.units == u.code_mass / u.code_length**3
    code_time = u.code_velocity / u.code_length
    # SFR is an important test-case, as the units cannot be inferred from the dimensionality (not in code units)
    assert sfr.units != u.code_mass / code_time  # == code_mass/code_time is wrong!
    # we know that TNG uses u.Msun/yr for SFR
    assert sfr.units == u.Msun / u.yr


def test_update_unitregistry():
    # update fields with units
    ureg = UnitRegistry()
    update_unitregistry("units/illustris.yaml", ureg)
    assert np.isclose((1.0 * ureg.Msun).to(ureg.g).magnitude, 1.98847e33)
    assert np.isclose(
        (1.0 * ureg.code_length).to(ureg.kpc).magnitude, 1.0 / 0.6774
    )  # 1/h for a=1.0
