import numpy as np
import pytest


from ..interface import BaseSnapshot
from ..interfaces.arepo import ArepoSnapshot, ArepoSnapshotWithUnits
from . import path,gpath

flag_test_long = False # Set to true to run time-taking tests.


def test_snapshot_load():
    snp = BaseSnapshot(path)


def test_groups_load():
    snp = BaseSnapshot(gpath)

def test_groups_load_nonvirtual():
    snp = BaseSnapshot(gpath, virtualcache=False)


@pytest.mark.skipif(not(flag_test_long), reason="Not requesting time-taking tasks")
def test_snapshot_save():
    snp = BaseSnapshot(path)
    snp.save("test.zarr")


@pytest.mark.skipif(not(flag_test_long), reason="Not requesting time-taking tasks")
def test_snapshot_load_zarr():
    snp = BaseSnapshot(path)
    snp.save("test.zarr")
    snp_zarr = BaseSnapshot("test.zarr")
    for k,o in snp.data.items():
        for l,u in o.items():
            assert u.shape==snp_zarr.data[k][l].shape


def test_areposnapshot_load():
    snp = ArepoSnapshot(path)
    assert len(snp.data.keys()) == 5

def test_areposnapshot_halooperation():
    snap = ArepoSnapshot(path, catalog=path.replace("snapdir", "groups"))

    def calculate_count(GroupID, parttype="PartType0"):
        """Halo Count per halo. Has to be 1 exactly."""
        return np.unique(GroupID).shape[0]

    def calculate_partcount(GroupID, parttype="PartType0"):
        """Particle Count per halo."""
        return GroupID.shape[0]

    def calculate_haloid(GroupID, parttype="PartType0"):
        """returns Halo ID"""
        return GroupID[-1]

    counttask = snap.map_halo_operation(calculate_count, compute=False, chunksize=int(3e6))
    partcounttask = snap.map_halo_operation(calculate_partcount, compute=False, chunksize=int(3e6))
    hidtask = snap.map_halo_operation(calculate_haloid, compute=False, chunksize=int(3e6))
    count = counttask.compute()
    partcount = partcounttask.compute()
    hid = hidtask.compute()
    print(hid)

    count0 = np.where(partcount == 0)[0]
    diff0 = np.sort(np.concatenate((count0, count0 - 1)))
    assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(diff0.tolist())
    if not (np.diff(hid).max() == np.diff(hid).min() == 1):
        assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(
            diff0.tolist())  # gotta check; potentially empty halos
        assert np.all(counttask.compute() <= 1)
    else:
        assert np.all(counttask.compute() == 1)
    assert count.shape == counttask.compute().shape
    assert count.shape[0] == snap.data["Group"]["GroupPos"].shape[0]
    assert np.all(partcount == snap.data["Group"]["GroupLenType"][:, 0].compute())


def test_areposnapshot_load_withcatalog():
    snp = ArepoSnapshot(path, catalog=gpath)
    assert len(snp.data.keys()) == 7


def test_areposnapshot_load_withcatalogandunits():
    snp = ArepoSnapshotWithUnits(path, catalog=gpath)
