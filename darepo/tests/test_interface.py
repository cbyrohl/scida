import numpy as np


from ..interface import BaseSnapshot
from ..interfaces.arepo import ArepoSnapshot, ArepoSnapshotWithUnits


# We test on a snapshot from TNG50-4
path = "/data/cbyrohl/TNGdata/TNG50-4/output/snapdir_042"
gpath = "/data/cbyrohl/TNGdata/TNG50-4/output/groups_042"


def test_snapshot_load():
    snp = BaseSnapshot(path)


def test_groups_load():
    snp = BaseSnapshot(gpath)


def test_snapshot_save():
    snp = BaseSnapshot(path)
    snp.save("test.zarr")


def test_areposnapshot_load():
    snp = ArepoSnapshot(path)
    assert len(snp.data.keys()) == 5


def test_areposnapshot_load_withcatalog():
    snp = ArepoSnapshot(path, catalog=gpath)
    assert len(snp.data.keys()) == 7


def test_areposnapshot_load_withcatalogandunits():
    snp = ArepoSnapshotWithUnits(path, catalog=gpath)
