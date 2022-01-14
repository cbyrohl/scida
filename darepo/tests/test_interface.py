import numpy as np


#from darepo.interface import BaseSnapshot
from ..interface import BaseSnapshot,ArepoSnapshot


# We test on a snapshot from TNG50-4
path = "/data/cbyrohl/TNGdata/TNG50-4/output/snapdir_042"

def test_snapshot_load():
    snp = BaseSnapshot(path)

def test_snapshot_save():
    snp = BaseSnapshot(path)
    snp.save("test.zarr")

def test_areposnapshot_load():
    snp = ArepoSnapshot(path)
