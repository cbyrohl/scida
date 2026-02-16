import h5py
import numpy as np
import pytest
from pint import UnitRegistry

from scida.config import get_config
from scida.convenience import load
from scida.interfaces.mixins.units import update_unitregistry


@pytest.mark.integration
def test_load_cgs(tngfile_dummy):
    p = tngfile_dummy.path
    ds = load(p, units="cgs")
    u = ds.ureg
    # test units of some fields
    ptype = "PartType0"
    gas = ds.data[ptype]
    coords = gas["Coordinates"]
    assert coords.units == u.cm
    dens = gas["Density"]
    assert dens.units == u.g / u.cm**3
    sfr = gas["StarFormationRate"]
    assert sfr.units == u.g / u.s


@pytest.mark.integration
def test_load_codeunits(tngfile_dummy):
    p = tngfile_dummy.path
    ds = load(p, units=True)
    u = ds.ureg
    #
    if "code_velocity" not in u:
        u.define("code_velocity = 1.0 * cm/s")
    if "code_length" not in u:
        u.define("code_length = 1.0 * cm")
    if "code_mass" not in u:
        u.define("code_mass = 1.0 * g")

    # test units of some fields
    ptype = "PartType0"
    gas = ds.data[ptype]
    coords = gas["Coordinates"]
    assert coords.units == u.code_length
    dens = gas["Density"]
    assert dens.units == u.code_mass / u.code_length**3
    code_time = u.code_velocity / u.code_length
    # SFR is an important test-case, as the units cannot be inferred from the dimensionality (not in code units)
    sfr = gas["StarFormationRate"]
    assert sfr.units != u.code_mass / code_time  # == code_mass/code_time is wrong!
    # we know that TNG uses u.Msun/yr for SFR
    assert sfr.units == u.Msun / u.yr

    # assert that IDs are not pint quantities
    assert not hasattr(gas["ParticleIDs"], "units")


@pytest.mark.unit
def test_update_unitregistry():
    # update fields with units
    ureg = UnitRegistry()
    update_unitregistry("units/general.yaml", ureg)
    ureg.define("h0 = 0.6774")
    ureg.define("code_length = 1.0 * kpc / h0")
    assert np.isclose((1.0 * ureg.Msun).to(ureg.g).magnitude, 1.98847e33)
    assert np.isclose(
        (1.0 * ureg.code_length).to(ureg.kpc).magnitude, 1.0 / 0.6774
    )  # 1/h for a=1.0


@pytest.mark.integration
def test_missingunits(monkeypatch, gadgetfile_dummy, caplog):
    p = gadgetfile_dummy.path
    with h5py.File(p, "r+") as f:
        f["PartType0"]["FieldWithoutUnits"] = np.ones(1)

    # raise
    monkeypatch.setenv("SCIDA_MISSING_UNITS", "raise")
    get_config(reload=True)
    with pytest.raises(ValueError) as exc_info:
        load(str(p), units=True)
    assert "Cannot determine units" in str(exc_info.value)

    # warn
    monkeypatch.setenv("SCIDA_MISSING_UNITS", "warn")
    get_config(reload=True)
    load(str(p), units=True)
    assert "Cannot determine units" in caplog.text

    # ignore
    monkeypatch.setenv("SCIDA_MISSING_UNITS", "ignore")
    get_config(reload=True)
    load(str(p), units=True)
    assert "Cannot determine units" in caplog.text


@pytest.mark.integration
def test_check_missingunits(monkeypatch, gadgetfile_dummy, caplog):
    p = gadgetfile_dummy.path
    with h5py.File(p, "r+") as f:
        f["PartType0"]["FieldWithoutUnits"] = np.ones(1)

    ds = load(str(p), units=True)
    assert ds.missing_units(), "Expected missing units"

    # print the log entries

    print(ds)


@pytest.mark.unit
def test_misc():
    ureg = UnitRegistry()
    print(ureg("dimensionless"))
    ureg.define("unknown = 1.0")
    a = 1.0 * ureg("unknown")
    print(a.dimensionality)
    print(ureg("unknown"))
