import time

import numpy as np

from scida.convenience import load
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot
from scida.io import ChunkedHDF5Loader
from tests.testdata_properties import require_testdata, require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_load(testdatapath):
    snp = GadgetStyleSnapshot(testdatapath)
    assert snp.file is not None


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_loadmetadata(testdatapath):
    loader = ChunkedHDF5Loader(testdatapath)
    metadata1 = loader.load_metadata()
    assert len(metadata1.keys()) > 0
    GadgetStyleSnapshot(testdatapath)  # load once, so this gets cached
    metadata2 = loader.load_metadata()
    assert len(metadata2.keys()) > 0
    assert metadata2.get("/_chunks", None) is not None


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


@require_testdata_path("interface", only=["TNG50-4_group"])
def test_load_nonvirtual(testdatapath):
    snp = GadgetStyleSnapshot(testdatapath, virtualcache=False)
    assert snp.file is not None


# TODO: setup and teardown savepath correctly
@require_testdata("interface", only=["TNG50-4_group"])
def test_snapshot_save(testdata_interface):
    snp = testdata_interface
    snp.save("test.zarr")


@require_testdata("interface", only=["TNG50-4_group"])
def test_snapshot_loadsaved(testdata_interface):
    snp = testdata_interface
    pos = snp.data["Group"]["GroupPos"][0].compute()
    snp.save("test.zarr")
    ds = load("test.zarr", units=False)
    pos2 = ds.data["Group"]["GroupPos"][0].compute()
    assert ds.boxsize[0] == snp.header["BoxSize"]
    assert np.all(np.equal(pos, pos2))


# @pytest.mark.skip()
# @pytest.mark.skipif(not (flag_test_long), reason="Not requesting time-taking tasks")
# def test_snapshot_load_zarr(snp):
#    snp.save("test.zarr")
#    snp_zarr = BaseSnapshot("test.zarr")
#    for k, o in snp.data.items():
#        for l, u in o.items():
#            assert u.shape == snp_zarr.data[k][l].shape


@require_testdata("areposnapshot", only=["TNG50-1_snapshot_z0_minimal"])
def test_areposnapshot_load(testdata_areposnapshot):
    snp = testdata_areposnapshot
    expected_types = {"PartType" + str(i) for i in range(6)}
    overlap = set(snp.data.keys()) & expected_types
    assert len(overlap) >= 5
    print("chunks", list(snp.file["_chunks"].attrs.keys()))


@require_testdata("areposnapshot", only=["TNG50-1_snapshot_z0_minimal"])
def test_interface_fieldaccess(testdata_areposnapshot):
    # check that we can directly access fields from the interface
    # should be same as ds.data["PartType0"]["Coordinates"]
    assert testdata_areposnapshot["PartType0"]["Coordinates"] is not None


@require_testdata("illustrisgroup")
def test_illustrisgroup_load(testdata_illustrisgroup):
    grp = testdata_illustrisgroup
    assert "Group" in grp.data.keys()
    assert "Subhalo" in grp.data.keys()


@require_testdata("areposnapshot_withcatalog", only=["TNG50-4_snapshot"])
def test_interface_groupedoperations_nonscalar(testdata_areposnapshot_withcatalog):
    # Test non-scalar output
    snp = testdata_areposnapshot_withcatalog
    g = snp.grouped()

    def customfunc3(dens, vol, fieldnames=["Density", "Masses"], shape=(2,)):
        return np.array([(dens * vol).max(), vol.max()])

    s = g.apply(customfunc3)
    res = s.evaluate()
    print(res)


@require_testdata("illustrissnapshot_withcatalog")
def test_areposnapshot_load_withcatalog(testdata_illustrissnapshot_withcatalog):
    snp = testdata_illustrissnapshot_withcatalog
    if snp.redshift >= 20.0:  # hacky... (no catalog at high redshifts)
        return
    parttypes = {
        "PartType0",
        "PartType1",
        "PartType3",
        "PartType4",
        "PartType5",
        "Group",
        "Subhalo",
    }
    overlap = set(snp.data.keys()) & parttypes
    assert len(overlap) == 7

    # Check whether the group catalog makes sense
    print(snp.data["Group"].keys())


@require_testdata("areposnapshot_withcatalog")
def test_areposnapshot_load_withcatalogandunits(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
    assert snp.file is not None


@require_testdata(
    "illustrissnapshot_withcatalog", exclude=["minimal", "z127"], exclude_substring=True
)
def test_areposnapshot_map_hquantity(testdata_illustrissnapshot_withcatalog):
    snp = testdata_illustrissnapshot_withcatalog
    snp.add_groupquantity_to_particles("GroupSFR")
    partarr = snp.data["PartType0"]["GroupSFR"]
    haloarr = snp.data["Group"]["GroupSFR"]
    assert np.isclose(partarr[0].compute(), haloarr[0].compute())
