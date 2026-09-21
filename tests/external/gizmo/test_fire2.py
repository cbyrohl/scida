import astropy.units as u
import numpy as np
import pytest

from scida.convenience import load
from scida.customs.gizmo.dataset import GizmoSnapshot
from scida.customs.gizmo.series import GizmoSimulation
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("series", only=["FIRE2_series_minimal"])
def test_series(testdatapath):
    obj = load(testdatapath)
    assert isinstance(obj, GizmoSimulation)

    ds = obj[0]
    print(ds.info())
    check_firesnap(ds)

    rh = ds.data["rockstar_halo"]

    print(rh.info())
    for k, v in rh.items():
        print(k, v)

    # Rockstar source says:
    # > chars += fprintf(output, "#Units: Masses in Msun / h\n"
    # >   "#Units: Positions in Mpc / h (comoving)\n"
    # >   "#Units: Velocities in km / s (physical, peculiar)\n"
    # >   "#Units: Halo Distances, Lengths, and Radii in kpc / h (comoving)\n"
    # >   "#Units: Angular Momenta in (Msun/h) * (Mpc/h) * km/s (physical)\n"
    # >   "#Units: Spins are dimensionless\n");
    # > if (np)
    # >   chars += fprintf(output, "#Units: Total energy in (Msun/h)*(km/s)^2"
    # >   	     " (physical)\n""


def check_firesnap(obj: GizmoSnapshot):
    assert isinstance(obj, GizmoSnapshot)
    assert all(
        [k in obj.data for k in ["PartType0", "PartType1", "PartType2", "PartType4"]]
    )

    z = obj._metadata_raw["/Header"]["Redshift"]
    h = obj._metadata_raw["/Header"]["HubbleParam"]
    a = 1.0 / (1.0 + z)

    # test units
    # Coordinates
    uq = (1.0 * obj.data["PartType0"]["Coordinates"].units).to("cm").magnitude
    uq_ref = a * (1.0 * u.kpc).to("cm").value / h
    assert pytest.approx(uq) == uq_ref
    # Velocities
    uq = (1.0 * obj.data["PartType0"]["Velocities"].units).to("cm/s").magnitude
    uq_ref = (np.sqrt(a) * u.km / u.s).to("cm/s").value
    assert pytest.approx(uq) == uq_ref
    # Masses
    uq = (1.0 * obj.data["PartType0"]["Masses"].units).to("g").magnitude
    uq_ref = (1e10 * u.Msun / h).to("g").value
    assert pytest.approx(uq, 1e-4) == uq_ref
    # Density
    uq = (1.0 * obj.data["PartType0"]["Density"].units).to("g/cm**3").magnitude
    uq_ref = (1e10 * u.Msun * h**2 * a**-3 / u.kpc**3).to("g/cm**3").value
    assert pytest.approx(uq) == uq_ref
    # Potential
    uq = (1.0 * obj.data["PartType0"]["Potential"].units).to("cm**2/s**2").magnitude
    uq_ref = (1.0 * u.km**2 / u.s**2).to("cm**2/s**2").value
    assert pytest.approx(uq) == uq_ref
    # InternalEnergy
    uq = (
        (1.0 * obj.data["PartType0"]["InternalEnergy"].units).to("cm**2/s**2").magnitude
    )
    uq_ref = (1.0 * u.km**2 / u.s**2).to("cm**2/s**2").value
    assert pytest.approx(uq) == uq_ref
    # SmoothingLength
    uq = (1.0 * obj.data["PartType0"]["SmoothingLength"].units).to("cm").magnitude
    uq_ref = a * (1.0 * u.kpc).to("cm").value / h
    assert pytest.approx(uq) == uq_ref
    # StarFormationRate
    uq = (
        (1.0 * obj.data["PartType0"]["StarFormationRate"].units).to("Msun/yr").magnitude
    )
    uq_ref = (1.0 * u.Msun / u.yr).to("Msun/yr").value
    assert pytest.approx(uq) == uq_ref
    # TODO: Test BH units; need massive halo snapshot for this.


@pytest.mark.external
@require_testdata_path("interface", only=["FIRE2_snapshot_z1_minimal"])
def test_snapshot(testdatapath):
    """Test loading of a full snapshot"""
    obj: GizmoSnapshot = load(testdatapath)
    print(obj.info())
    check_firesnap(obj)
