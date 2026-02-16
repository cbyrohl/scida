import logging

import dask.array as da
import numpy as np
import pytest

from scida import load
from scida.customs.arepo.dataset import part_type_num
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_selector(testdatapath):
    parttype = "PartType0"
    hid = 41
    obj = load(testdatapath, units=False)
    d = obj.return_data(haloID=hid)
    hid41_len = d[parttype]["GroupID"].shape[0]
    assert da.all(d[parttype]["GroupID"] == hid).compute()

    shid = 42 * 42
    d = obj.return_data(subhaloID=shid)
    p0_shid = d[parttype]["SubhaloID"]
    assert p0_shid.shape[0] == obj.get_subhalolengths(parttype)[shid]
    assert da.all(p0_shid == shid).compute()

    local_shid = 1
    hid = 41
    nshs = int(obj.data["Group"]["GroupNsubs"][hid].compute())
    assert nshs > 0
    d = obj.return_data(haloID=hid, localSubhaloID=local_shid)
    shid = obj.data["Group"]["GroupFirstSub"][hid] + local_shid
    p0_shid = d[parttype]["SubhaloID"]
    assert p0_shid.shape[0] == obj.get_subhalolengths(parttype)[shid]
    assert p0_shid.shape[0] < hid41_len
    local_shid = nshs
    pytest.raises(ValueError, obj.return_data, haloID=hid, localSubhaloID=local_shid)

    d = obj.return_data(unbound=True)
    unbound_id = obj.misc["unboundID"]
    assert da.all(d[parttype]["GroupID"] == unbound_id).compute()


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_subhalolengths(testdatapath):
    pass


def halooperations(path, catalogpath=None):
    snap = load(path, catalog=catalogpath)

    def calculate_count(GroupID, parttype="PartType0"):
        return np.unique(GroupID).shape[0]

    def calculate_partcount(GroupID, parttype="PartType0"):
        return GroupID.shape[0]

    def calculate_haloid(GroupID, parttype="PartType0"):
        if len(GroupID) > 0:
            return GroupID[-1]
        else:
            return -21

    counttask = snap.map_group_operation(calculate_count, compute=False, nchunks_min=20)
    partcounttask = snap.map_group_operation(calculate_partcount, compute=False)
    hidtask = snap.map_group_operation(calculate_haloid, compute=False)
    count = counttask.compute()
    partcount = partcounttask.compute()
    hid = hidtask.compute()

    count0 = np.where(partcount == 0)[0]
    diff0 = np.sort(np.concatenate((count0, count0 - 1)))
    assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(diff0.tolist())

    if not (np.diff(hid).max() == np.diff(hid).min() == 1):
        assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(diff0.tolist())
        assert np.all(counttask.compute() <= 1)
    else:
        assert np.all(counttask.compute() == 1)
    assert count.shape == counttask.compute().shape
    assert count.shape[0] == snap.data["Group"]["GroupPos"].shape[0]
    assert np.all(partcount == snap.data["Group"]["GroupLenType"][:, 0].compute())

    nmax = 10
    partcounttask = snap.map_group_operation(
        calculate_partcount, compute=False, nmax=nmax
    )
    partcount2 = partcounttask.compute()
    assert partcount2.shape[0] == nmax
    assert np.all(partcount2 == partcount[:nmax])

    idxlist = [3, 5, 7, 25200]
    partcounttask = snap.map_group_operation(
        calculate_partcount, compute=False, idxlist=idxlist
    )
    partcount2 = partcounttask.compute()
    assert partcount2.shape[0] == len(idxlist)
    assert np.all(partcount2 == partcount[idxlist])


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_areposnapshot_selector_halos_realdata(testdatapath):
    halooperations(testdatapath)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_areposnapshot_selector_subhalos_realdata(testdatapath):
    snap = load(testdatapath)
    parttype = "PartType1"

    def calculate_pindex_min(uid, parttype=parttype):
        try:
            return uid.min()
        except:  # noqa
            return -21

    def calculate_subhalocount(SubhaloID, parttype=parttype):
        return np.unique(SubhaloID).shape[0]

    def calculate_halocount(GroupID, parttype=parttype, dtype=np.int64):
        return np.unique(GroupID).shape[0]

    def calculate_partcount(SubhaloID, parttype=parttype, dtype=np.int64):
        return SubhaloID.shape[0]

    def calculate_subhaloid(SubhaloID, parttype=parttype, fill_value=-21, dtype=np.int64):
        return SubhaloID[0]

    def calculate_haloid(GroupID, parttype=parttype, fill_value=-21, dtype=np.int64):
        return GroupID[0]

    pindextask = snap.map_group_operation(
        calculate_pindex_min, compute=False, nchunks_min=20, objtype="subhalo"
    )
    shcounttask = snap.map_group_operation(
        calculate_subhalocount, compute=False, nchunks_min=20, objtype="subhalo"
    )
    hcounttask = snap.map_group_operation(
        calculate_halocount, compute=False, nchunks_min=20, objtype="subhalo"
    )
    partcounttask = snap.map_group_operation(
        calculate_partcount, compute=False, objtype="subhalo"
    )
    hidtask = snap.map_group_operation(
        calculate_haloid, compute=False, objtype="subhalo"
    )
    sidtask = snap.map_group_operation(
        calculate_subhaloid, compute=False, objtype="subhalo"
    )
    pindex_min = pindextask.compute()
    hcount = hcounttask.compute()
    shcount = shcounttask.compute()
    partcount = partcounttask.compute()
    hid = hidtask.compute()
    sid = sidtask.compute()

    shgrnr = snap.data["Subhalo"]["SubhaloGrNr"].compute()
    assert hid.shape[0] == shgrnr.shape[0]

    sh_pcount = snap.data["Subhalo"]["SubhaloLenType"][
        :, part_type_num(parttype)
    ].compute()
    mask = sh_pcount > 0

    assert np.all(hcount[mask] == 1)
    assert np.all(shcount[mask] == 1)
    assert np.all(hid[mask] == shgrnr[mask])
    assert np.all(sid[mask] == np.arange(sid.shape[0])[mask])

    shoffsets = snap.get_subhalooffsets(parttype)
    assert np.all(pindex_min[mask] == shoffsets[mask])

    shlengths = snap.get_subhalolengths(parttype)
    assert np.all(partcount[mask] == shlengths[mask])
