import pathlib
import time

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from scida import load
from scida.convenience import load as convenience_load
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot
from scida.interface import BaseDataset
from scida.io import ChunkedHDF5Loader
from tests.testdata_properties import require_testdata, require_testdata_path


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_load(testdatapath):
    snp = GadgetStyleSnapshot(testdatapath)
    assert snp.file is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_loadmetadata(testdatapath):
    loader = ChunkedHDF5Loader(testdatapath)
    metadata1 = loader.load_metadata()
    assert len(metadata1.keys()) > 0
    GadgetStyleSnapshot(testdatapath)
    metadata2 = loader.load_metadata()
    assert len(metadata2.keys()) > 0
    assert metadata2.get("/_chunks", None) is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_snapshot_load_usecache(testdatapath):
    tstart = time.process_time()
    snp = GadgetStyleSnapshot(testdatapath)
    dt0 = time.process_time() - tstart
    tstart = time.process_time()
    snp = GadgetStyleSnapshot(testdatapath)
    dt1 = time.process_time() - tstart
    assert 4 * dt1 < dt0
    assert snp.file is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_group"])
def test_load_nonvirtual(testdatapath):
    snp = GadgetStyleSnapshot(testdatapath, virtualcache=False)
    assert snp.file is not None


@pytest.mark.external
@require_testdata("interface", only=["TNG50-4_group"])
def test_snapshot_save(testdata_interface):
    snp = testdata_interface
    snp.save("test.zarr")


@pytest.mark.external
@require_testdata("interface", only=["TNG50-4_group"])
def test_snapshot_loadsaved(testdata_interface):
    snp = testdata_interface
    pos = snp.data["Group"]["GroupPos"][0].compute()
    snp.save("test.zarr")
    ds = convenience_load("test.zarr", units=False)
    pos2 = ds.data["Group"]["GroupPos"][0].compute()
    assert ds.boxsize[0] == snp.header["BoxSize"]
    assert np.all(np.equal(pos, pos2))


@pytest.mark.external
@require_testdata("areposnapshot", only=["TNG50-1_snapshot_z0_minimal"])
def test_areposnapshot_load(testdata_areposnapshot):
    snp = testdata_areposnapshot
    expected_types = {"PartType" + str(i) for i in range(6)}
    overlap = set(snp.data.keys()) & expected_types
    assert len(overlap) >= 5
    print("chunks", list(snp.file["_chunks"].attrs.keys()))


@pytest.mark.external
@require_testdata("areposnapshot", only=["TNG50-1_snapshot_z0_minimal"])
def test_interface_fieldaccess(testdata_areposnapshot):
    assert testdata_areposnapshot["PartType0"]["Coordinates"] is not None


@pytest.mark.external
@require_testdata("illustrisgroup")
def test_illustrisgroup_load(testdata_illustrisgroup):
    grp = testdata_illustrisgroup
    assert "Group" in grp.data.keys()
    assert "Subhalo" in grp.data.keys()


@pytest.mark.external
@require_testdata("illustrissnapshot_withcatalog")
def test_areposnapshot_load_withcatalog(testdata_illustrissnapshot_withcatalog):
    snp = testdata_illustrissnapshot_withcatalog
    if snp.redshift >= 20.0:
        return
    parttypes = {
        "PartType0", "PartType1", "PartType3", "PartType4", "PartType5",
        "Group", "Subhalo",
    }
    overlap = set(snp.data.keys()) & parttypes
    assert len(overlap) == 7
    print(snp.data["Group"].keys())


@pytest.mark.external
@require_testdata("areposnapshot_withcatalog")
def test_areposnapshot_load_withcatalogandunits(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
    assert snp.file is not None


@pytest.mark.external
@require_testdata(
    "illustrissnapshot_withcatalog", exclude=["minimal", "z127"], exclude_substring=True
)
def test_areposnapshot_map_hquantity(testdata_illustrissnapshot_withcatalog):
    snp = testdata_illustrissnapshot_withcatalog
    snp.add_groupquantity_to_particles("GroupSFR")
    partarr = snp.data["PartType0"]["GroupSFR"]
    haloarr = snp.data["Group"]["GroupSFR"]
    assert np.isclose(partarr[0].compute(), haloarr[0].compute())


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_redshift_mismatch(testdatapath):
    grp_path = testdatapath.replace("snapshot", "group")
    load(testdatapath, catalog=grp_path)
    grp_path = grp_path.replace("z3", "z0")
    with pytest.raises(Exception) as exc_info:
        load(testdatapath, catalog=grp_path)
    assert str(exc_info.value).startswith("Redshift mismatch")


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_scalefactor_in_unitreg(testdatapath):
    ds = load(testdatapath, units=True)
    a = ds.ureg("a").to_base_units().magnitude
    h = ds.ureg("h").to_base_units().magnitude
    assert a == pytest.approx(1.0 / (1.0 + ds.redshift))
    assert h == pytest.approx(0.6774)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_scalefactor_in_units(testdatapath):
    ds_nounits = load(testdatapath, units=False)
    ds = load(testdatapath, units=True)
    a = ds.ureg("a").to_base_units().magnitude
    h = ds.ureg("h").to_base_units().magnitude
    vel = ds.data["PartType0"]["Velocities"][0, 0].compute()
    vel_nounits = ds_nounits.data["PartType0"]["Velocities"][0, 0].compute()
    vel_cgs = vel.to("cm/s").magnitude
    vel_nounits_cgs = 1e5 * np.sqrt(a) * vel_nounits
    assert vel_cgs == pytest.approx(vel_nounits_cgs)
    pos = ds.data["PartType0"]["Coordinates"][0, 0].compute()
    pos_nounits = ds_nounits.data["PartType0"]["Coordinates"][0, 0].compute()
    pos_cgs = pos.to_base_units().magnitude
    pos_nounits_cgs = a * ds.ureg("kpc").to("cm").magnitude * pos_nounits / h
    assert pos_cgs == pytest.approx(pos_nounits_cgs)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_discover_groupcatalog(testdatapath):
    ds = load(testdatapath)
    assert ds.catalog is not None
    print(ds.catalog)


@pytest.mark.external
@require_testdata_path("series", only=["TNG50-4"])
def test_discover_groupcatalog2(testdatapath):
    p = pathlib.Path(testdatapath)
    path = p / "output" / "snapdir_099" / "snap_099.0.hdf5"
    ds = load(path)
    assert ds.catalog is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_aliases(testdatapath):
    ds = load(testdatapath)
    gas = ds.data["PartType0"]
    with pytest.raises(KeyError):
        print(gas["TestAlias"])
    gas.add_alias("TestAlias", "Coordinates")
    assert gas["TestAlias"] is not None
    assert ds.data["Group"]["Coordinates"] is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_aliases_bydefault(testdatapath):
    ds = load(testdatapath)
    d = ds.data
    assert d["PartType0"] == d["gas"]
    assert d["PartType1"] == d["dm"]
    if "PartType2" in d:
        assert d["PartType2"] == d["lowres"]
    assert d["PartType3"] == d["tracers"]
    assert d["PartType4"] == d["stars"]
    assert d["PartType5"] == d["bh"]


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-1_snapshot_z3_minimal"])
def test_additionalfields(testdatapath):
    ds = load(testdatapath)
    gas = ds.data["PartType0"]
    assert "Temperature" in gas
    assert gas["Temperature"] is not None
    assert gas["Temperature"].compute() is not None


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot_z127"])
def test_emptycatalog(testdatapath):
    ds = load(testdatapath)
    print(ds)
    pass


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_simulation_tng50(testdatapath):
    """Verify TNG50 unit conversions against cosmological expectations."""
    snp = load(testdatapath, units="cgs")
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
    assert np.isclose(maxpos.to("Mpc").compute().magnitude, 35 / snp.cosmology.h)
