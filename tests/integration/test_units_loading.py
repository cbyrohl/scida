import h5py
import numpy as np
import pytest

from scida.config import get_config
from scida.convenience import load


@pytest.mark.integration
def test_load_cgs(tngfile_dummy):
    p = tngfile_dummy.path
    ds = load(p, units="cgs")
    u = ds.ureg
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
    if "code_velocity" not in u:
        u.define("code_velocity = 1.0 * cm/s")
    if "code_length" not in u:
        u.define("code_length = 1.0 * cm")
    if "code_mass" not in u:
        u.define("code_mass = 1.0 * g")
    ptype = "PartType0"
    gas = ds.data[ptype]
    coords = gas["Coordinates"]
    assert coords.units == u.code_length
    dens = gas["Density"]
    assert dens.units == u.code_mass / u.code_length**3
    code_time = u.code_velocity / u.code_length
    sfr = gas["StarFormationRate"]
    assert sfr.units != u.code_mass / code_time
    assert sfr.units == u.Msun / u.yr
    assert not hasattr(gas["ParticleIDs"], "units")


@pytest.mark.integration
def test_missingunits(monkeypatch, gadgetfile_dummy, caplog):
    p = gadgetfile_dummy.path
    with h5py.File(p, "r+") as f:
        f["PartType0"]["FieldWithoutUnits"] = np.ones(1)
    monkeypatch.setenv("SCIDA_MISSING_UNITS", "raise")
    get_config(reload=True)
    with pytest.raises(ValueError) as exc_info:
        load(str(p), units=True)
    assert "Cannot determine units" in str(exc_info.value)
    monkeypatch.setenv("SCIDA_MISSING_UNITS", "warn")
    get_config(reload=True)
    load(str(p), units=True)
    assert "Cannot determine units" in caplog.text
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
    print(ds)
