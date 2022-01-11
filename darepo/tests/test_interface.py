import pytest


#from darepo.interface import BaseSnapshot
from ..interface import BaseSnapshot


# We test on a snapshot from TNG50-4
path = "/data/cbyrohl/TNGdata/TNG50-4/output/snapdir_042"

def test_snapshot_load():
    snp = BaseSnapshot(path)