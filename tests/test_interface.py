import dask.array as da
import numpy as np
import pytest

from astrodask.interface import BaseSnapshot
from tests.testdata_properties import require_testdata, require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_load(cachedir, testdatapath):
    snp = BaseSnapshot(testdatapath, virtualcache=True)
    assert snp.file is not None


# TODO: revisit how to do this smartly and assert that cache speeds things up
# @pytest.mark.skip()
# def test_snapshot_load_usecache(snp, monkeypatch, tmp_path):
#    d = tmp_path / "cachedir"
#    d.mkdir()
#    print(str(d))
#    monkeypatch.setenv("ASTRODASK_CACHEDIR", str(d))
#    print("CONF", dict(_config))
#    assert snp.file is not None


@require_testdata_path("interface", only=["TNG50-4_group"])
def test_load_nonvirtual(testdatapath):
    snp = BaseSnapshot(testdatapath, virtualcache=False)
    assert snp.file is not None


# TODO: revisit additional flags
# TODO: setup and teardown savepath correctly
# @pytest.mark.skipif(not (flag_test_long), reason="Not requesting time-taking tasks")
@pytest.mark.skip()
@require_testdata("interface", only=["TNG50-4_snapshot"])
def test_snapshot_save(testdata_interface):
    snp = testdata_interface
    snp.save("test.zarr")


# @pytest.mark.skip()
# @pytest.mark.skipif(not (flag_test_long), reason="Not requesting time-taking tasks")
# def test_snapshot_load_zarr(snp):
#    snp.save("test.zarr")
#    snp_zarr = BaseSnapshot("test.zarr")
#    for k, o in snp.data.items():
#        for l, u in o.items():
#            assert u.shape == snp_zarr.data[k][l].shape


@require_testdata("areposnapshot")
def test_areposnapshot_load(testdata_areposnapshot):
    snp = testdata_areposnapshot
    assert len(snp.data.keys()) == 5
    print("chunks", list(snp.file["_chunks"].attrs.keys()))


@require_testdata("illustrisgroup")
def test_illustrisgroup_load(testdata_illustrisgroup):
    grp = testdata_illustrisgroup
    assert "Group" in grp.data.keys()
    assert "Subhalo" in grp.data.keys()


@require_testdata("areposnapshot_withcatalog")
def test_areposnapshot_halooperation(testdata_areposnapshot_withcatalog):
    snap = testdata_areposnapshot_withcatalog

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


@require_testdata("areposnapshot_withcatalog")
def test_interface_groupedoperations(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
    # check bound mass sums as a start
    g = snp.grouped("Masses")
    boundmass = np.sum(g.sum().evaluate())
    boundmass2 = da.sum(
        snp.data["PartType0"]["Masses"][: np.sum(snp.get_grouplengths())]
    ).compute()
    assert np.isclose(boundmass, boundmass2)
    # Test chaining
    assert np.sum(g.half().sum().evaluate()) < np.sum(g.sum().evaluate())


@require_testdata("areposnapshot_withcatalog")
def test_areposnapshot_load_withcatalog(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
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


@require_testdata("areposnapshot_withcatalog")
def test_areposnapshot_load_withcatalogandunits(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
    assert snp.file is not None


@require_testdata("areposnapshot_withcatalog")
def test_areposnapshot_map_hquantity(testdata_areposnapshot_withcatalog):
    snp = testdata_areposnapshot_withcatalog
    snp.add_groupquantity_to_particles("GroupSFR")
    partarr = snp.data["PartType0"]["GroupSFR"]
    haloarr = snp.data["Group"]["GroupSFR"]
    assert partarr[0].compute() == haloarr[0].compute()
