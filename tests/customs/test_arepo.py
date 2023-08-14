import logging

import dask.array as da
import numpy as np
import pint

from scida import load
from scida.customs.arepo.dataset import part_type_num
from tests.testdata_properties import require_testdata, require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_selector(testdatapath):
    parttype = "PartType0"
    # test selecting by halo id
    hid = 41
    obj = load(testdatapath, units=False)
    d = obj.return_data(haloID=hid)
    assert da.all(d[parttype]["GroupID"] == hid).compute()

    # test selecting by subhalo id
    shid = 42 * 42
    d = obj.return_data(subhaloID=shid)
    p0_shid = d[parttype]["SubhaloID"]
    assert p0_shid.shape[0] == obj.get_subhalolengths(parttype)[shid]
    assert da.all(p0_shid == shid).compute()

    # test selecting by unbound gas
    d = obj.return_data(unbound=True)
    # unbound gas has groupid = max_int64 = 9223372036854775807
    unbound_id = obj.misc["unboundID"]
    assert da.all(d[parttype]["GroupID"] == unbound_id).compute()


# test catalog calculation off (sub)halo lengths and offsets
# for particle data fields and selections
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_subhalolengths(testdatapath):
    pass
    # obj = load(testdatapath, units=False)
    # glen = obj.get_grouplengths()
    # shlen = obj.get_subhalolengths()
    # offsets = obj.get_subhalooffsets()
    # TBD


def halooperations(path, catalogpath=None):
    snap = load(path, catalog=catalogpath)

    def calculate_count(GroupID, parttype="PartType0"):
        """Number of unique halo associations found in each halo. Has to be 1 exactly."""
        return np.unique(GroupID).shape[0]

    def calculate_partcount(GroupID, parttype="PartType0"):
        """Particle Count per halo."""
        return GroupID.shape[0]

    def calculate_haloid(GroupID, parttype="PartType0"):
        """returns Halo ID"""
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
    # determine halos that hold no particles.
    assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(diff0.tolist())

    if not (np.diff(hid).max() == np.diff(hid).min() == 1):
        assert set(np.where(np.diff(hid) != 1)[0].tolist()) == set(
            diff0.tolist()
        )  # gotta check; potentially empty halos
        assert np.all(counttask.compute() <= 1)
    else:
        assert np.all(counttask.compute() == 1)
    assert (
        count.shape == counttask.compute().shape
    ), "Expected shape different from result's shape."
    assert count.shape[0] == snap.data["Group"]["GroupPos"].shape[0]
    assert np.all(partcount == snap.data["Group"]["GroupLenType"][:, 0].compute())

    # test nmax
    nmax = 10
    partcounttask = snap.map_group_operation(
        calculate_partcount, compute=False, nmax=nmax
    )
    partcount2 = partcounttask.compute()
    assert partcount2.shape[0] == nmax
    assert np.all(partcount2 == partcount[:nmax])

    # test idxlist
    idxlist = [3, 5, 7, 25200]
    partcounttask = snap.map_group_operation(
        calculate_partcount, compute=False, idxlist=idxlist
    )
    partcount2 = partcounttask.compute()
    assert partcount2.shape[0] == len(idxlist)
    assert np.all(partcount2 == partcount[idxlist])


# Test the map_halo_operation/grouped functionality
# todo: need to have catalog have more than 1 halo for this.
# def test_areposnapshot_selector_halos(tngfile_dummy, gadgetcatalogfile_dummy):
#     path = tngfile_dummy.path
#     catalogpath = gadgetcatalogfile_dummy.path
#     halooperations(path, catalogpath)


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_areposnapshot_selector_halos_realdata(testdatapath):
    halooperations(testdatapath)


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_areposnapshot_selector_subhalos_realdata(testdatapath):
    snap = load(testdatapath)
    # "easy starting point as subhalos are guaranteed to have dm particles"
    # apparently above statement is not true. there are subhalos without dm particles in TNG.
    parttype = "PartType1"

    def calculate_pindex_min(uid, parttype=parttype):
        """Minimum particle index to consider."""
        try:
            return uid.min()
        except:  # noqa
            return -21

    def calculate_subhalocount(SubhaloID, parttype=parttype):
        """Number of unique subhalo associations found in each subhalo. Has to be 1 exactly."""
        return np.unique(SubhaloID).shape[0]

    def calculate_halocount(GroupID, parttype=parttype, dtype=np.int64):
        """Number of unique halo associations found in each subhalo. Has to be 1 exactly."""
        return np.unique(GroupID).shape[0]

    def calculate_partcount(SubhaloID, parttype=parttype, dtype=np.int64):
        """Particle Count per halo."""
        return SubhaloID.shape[0]

    def calculate_subhaloid(
        SubhaloID, parttype=parttype, fill_value=-21, dtype=np.int64
    ):
        """returns Subhalo ID"""
        return SubhaloID[0]

    def calculate_haloid(GroupID, parttype=parttype, fill_value=-21, dtype=np.int64):
        """returns Halo ID"""
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
    # the hid should SubhaloGrNr
    # the sid should just be the calling subhalo index itself

    shgrnr = snap.data["Subhalo"]["SubhaloGrNr"].compute()
    assert hid.shape[0] == shgrnr.shape[0]

    sh_pcount = snap.data["Subhalo"]["SubhaloLenType"][
        :, part_type_num(parttype)
    ].compute()
    mask = sh_pcount > 0

    # each subhalo belongs only to one halo
    assert np.all(hcount[mask] == 1)
    # all subhalo particles of given subhalo to one halo (which is itself...)
    assert np.all(shcount[mask] == 1)
    # all particles of given subhalo only belong to the parent halo
    assert np.all(hid[mask] == shgrnr[mask])
    # all particles of given subhalo only belong to a specified subhalo
    assert np.all(sid[mask] == np.arange(sid.shape[0])[mask])

    # check for correct particle offsets
    shoffsets = snap.get_subhalooffsets(parttype)
    assert np.all(pindex_min[mask] == shoffsets[mask])

    # check for correct particle count
    shlengths = snap.get_subhalolengths(parttype)
    assert np.all(partcount[mask] == shlengths[mask])


@require_testdata("areposnapshot_withcatalog", only=["TNG50-4_snapshot"])
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

    def customfunc1(arr, fieldnames="Masses"):
        return arr[::2]

    s = g2.apply(customfunc1).sum()
    assert np.allclose(s.evaluate(), g.half().sum().evaluate())

    # Test custom dask array input
    arr = snp.data["PartType0"]["Density"] * snp.data["PartType0"]["Masses"]
    boundvol2 = snp.grouped(arr).sum().evaluate().sum()
    assert 0.0 < boundvol2 < 1.0

    # Test multifield
    def customfunc2(dens, vol, fieldnames=["Density", "Masses"]):
        return dens * vol

    s = g2.apply(customfunc2).sum()
    boundvol = s.evaluate().sum()
    assert np.isclose(boundvol, boundvol2)

    # Test nmax attribute
    m = snp.grouped("Masses").sum().evaluate()
    m10 = snp.grouped("Masses").sum().evaluate(nmax=10)
    assert m10.shape[0] == 10
    assert np.allclose(m[:10], m10)

    # Test idxlist attribute
    idxlist = [3, 5, 7, 25200]
    m4 = snp.grouped("Masses").sum().evaluate(idxlist=idxlist)
    assert m4.shape[0] == len(idxlist)
    assert np.allclose(m[idxlist], m4)

    # Test subhalos
    nsubs = snp.data["Subhalo"]["SubhaloMass"].shape[0]
    m = snp.grouped("Masses", objtype="subhalos").sum().evaluate()
    assert m.shape[0] == nsubs


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_groupedoperations_nonscalar(testdatapath, caplog):
    """Test grouped operations with non-scalar function outputs."""
    snp = load(testdatapath)

    # 1. specify non-scalar operation output via shape parameter
    ngrp = snp.data["Group"]["GroupMass"].shape[0]
    g = snp.grouped()
    shape = (2,)

    def customfunc(mass, fieldnames=["Masses"], shape=shape):
        return np.array([np.min(mass), np.max(mass)])

    s = g.apply(customfunc)
    res = s.evaluate()
    assert res.shape[0] == ngrp
    assert res.shape[1] == shape[0]
    assert np.all(res[:, 0] <= res[:, 1])

    # 2. check behavior when forgetting additional shape specification
    # for non-scalar operation output
    # 2.1 simple case where inference should work
    def customfunc2(mass, fieldnames=["Masses"]):
        return np.array([np.min(mass), np.max(mass)])

    s = g.apply(customfunc2)
    res = s.evaluate()
    assert res.shape[1] == shape[0]

    # 2.2 case where inference should fail
    def customfunc3(mass, fieldnames=["Masses"]):
        return [mass[2], mass[3]]

    s = g.apply(customfunc3)
    caplog.set_level(logging.WARNING)
    try:
        res = s.evaluate()
    except IndexError:
        pass  # we expect an index error further down the evaluate call.
    assert "Exception during shape inference" in caplog.text
