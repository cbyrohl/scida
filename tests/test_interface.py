import time

import dask.array as da
import numpy as np
import pint

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


@require_testdata(
    "illustrissnapshot_withcatalog", exclude=["minimal", "z127"], exclude_substring=True
)
def test_areposnapshot_halooperation(testdata_illustrissnapshot_withcatalog):
    snap = testdata_illustrissnapshot_withcatalog

    def calculate_count(GroupID, parttype="PartType0"):
        """Halo Count per halo. Has to be 1 exactly."""
        return np.unique(GroupID).shape[0]

    def calculate_partcount(GroupID, parttype="PartType0"):
        """Particle Count per halo."""
        return GroupID.shape[0]

    def calculate_haloid(GroupID, parttype="PartType0"):
        """returns Halo ID"""
        return GroupID[-1]

    counttask = snap.map_halo_operation(calculate_count, compute=False, Nmin=20)
    partcounttask = snap.map_halo_operation(
        calculate_partcount, compute=False, chunksize=int(3e6)
    )
    hidtask = snap.map_halo_operation(
        calculate_haloid, compute=False, chunksize=int(3e6)
    )
    count = counttask.compute()
    partcount = partcounttask.compute()
    hid = hidtask.compute()

    count0 = np.where(partcount == 0)[0]
    diff0 = np.sort(np.concatenate((count0, count0 - 1)))
    # determine halos that hold no particles.
    assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(diff0.tolist())

    if not (np.diff(hid).max() == np.diff(hid).min() == 1):
        assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(
            diff0.tolist()
        )  # gotta check; potentially empty halos
        assert np.all(counttask.compute() <= 1)
    else:
        assert np.all(counttask.compute() == 1)
    print(count.shape, counttask.compute().shape)
    assert (
        count.shape == counttask.compute().shape
    ), "Expected shape different from result's shape."
    assert count.shape[0] == snap.data["Group"]["GroupPos"].shape[0]
    assert np.all(partcount == snap.data["Group"]["GroupLenType"][:, 0].compute())


@require_testdata("areposnapshot_withcatalog", nmax=1)
def test_interface_groupedoperations(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
    # check bound mass sums as a start
    g = snp.grouped("Masses")
    boundmass = np.sum(g.sum().evaluate())
    boundmass2 = da.sum(
        snp.data["PartType0"]["Masses"][: np.sum(snp.get_grouplengths())]
    ).compute()
    if isinstance(boundmass2, pint.Quantity):
        boundmass2 = boundmass2.magnitude
    assert np.isclose(boundmass, boundmass2)
    # Test chaining
    assert np.sum(g.half().sum().evaluate()) < np.sum(g.sum().evaluate())

    # Test custom function apply
    assert np.allclose(
        g.apply(lambda x: x[::2]).sum().evaluate(), g.half().sum().evaluate()
    )

    # Test unspecified fieldnames when grouping
    g2 = snp.grouped()

    def customfunc(arr, fieldnames="Masses"):
        return arr[::2]

    s = g2.apply(customfunc).sum()
    assert np.allclose(s.evaluate(), g.half().sum().evaluate())

    # Test custom dask array input
    arr = snp.data["PartType0"]["Density"] * snp.data["PartType0"]["Masses"]
    boundvol2 = snp.grouped(arr).sum().evaluate().sum()
    assert 0.0 < boundvol2 < 1.0

    # Test multifield
    def customfunc(dens, vol, fieldnames=["Density", "Masses"]):
        return dens * vol

    s = g2.apply(customfunc).sum()
    boundvol = s.evaluate().sum()
    assert np.isclose(boundvol, boundvol2)


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
