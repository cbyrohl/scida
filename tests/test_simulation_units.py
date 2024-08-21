import numpy as np
from astropy.cosmology import FlatLambdaCDM

from scida.convenience import load
from tests.testdata_properties import require_testdata_path

# ureg = UnitRegistry()
# ureg.define("Msun = 1.98847e33 * g")
# ureg.define("h = 0.6774")


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_simulation_tng50(testdatapath):
    snp = load(testdatapath, units="cgs")

    # Compare dark matter mass
    cosmology: FlatLambdaCDM = snp.cosmology
    ureg = snp.unitregistry
    vol = np.prod(snp.boxsize) / cosmology.h**3 * ureg.kpc**3
    dens = (cosmology.Om0 - cosmology.Ob0) * cosmology.critical_density0
    dens = dens.value * ureg(dens.unit.to_string("ogip"))
    boxmass = vol * dens
    nparts = snp.header["NumPart_Total"][1]
    massperparticle = snp.header["MassTable"][1]
    partmass = nparts * massperparticle * 1e10 * ureg("Msun") / ureg("h")
    assert np.isclose(boxmass.to("Msun"), partmass.to("Msun"), rtol=1e-4)

    maxpos = snp.data["PartType0"]["Coordinates"].max()
    # TODO: Failing because "h" is still thought of as hours:
    # the registry appears to be reset when using dask array...
    # ...and then evaluating to numpy on compute? => file bug report
    assert np.isclose(maxpos.to("Mpc").compute().magnitude, 35 / snp.cosmology.h)
