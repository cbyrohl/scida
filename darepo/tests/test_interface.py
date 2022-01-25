import numpy as np
import pytest


from ..interface import BaseSnapshot
from ..interfaces.arepo import ArepoSnapshot, ArepoSnapshotWithUnits
from . import path,gpath




def test_snapshot_load():
    snp = BaseSnapshot(path)


def test_groups_load():
    snp = BaseSnapshot(gpath)


#@pytest.mark.dependency()
def test_snapshot_save():
    snp = BaseSnapshot(path)
    snp.save("test.zarr")


#@pytest.mark.dependency(depends=["test_snapshot_save"])
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


def test_areposnapshot_load_withcatalog():
    snp = ArepoSnapshot(path, catalog=gpath)
    assert len(snp.data.keys()) == 7


def test_areposnapshot_load_withcatalogandunits():
    snp = ArepoSnapshotWithUnits(path, catalog=gpath)
