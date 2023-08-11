import dask.array as da
import numpy as np
import pint

from scida import load
from tests.testdata_properties import require_testdata, require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_selector(testdatapath):
    hid = 41
    obj = load(testdatapath, units=False)
    d = obj.return_data(haloID=hid)
    assert da.all(d["PartType0"]["GroupID"] == hid).compute()

    d = obj.return_data(unbound=True)
    # unbound gas has groupid = max_int64 = 9223372036854775807
    assert da.all(d["PartType0"]["GroupID"] == 9223372036854775807).compute()


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
        return GroupID[-1]

    counttask = snap.map_halo_operation(calculate_count, compute=False, min_grpcount=20)
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
    assert (
        count.shape == counttask.compute().shape
    ), "Expected shape different from result's shape."
    assert count.shape[0] == snap.data["Group"]["GroupPos"].shape[0]
    assert np.all(partcount == snap.data["Group"]["GroupLenType"][:, 0].compute())

    # test nmax
    nmax = 10
    partcounttask = snap.map_halo_operation(
        calculate_partcount, compute=False, chunksize=int(3e6), nmax=nmax
    )
    partcount2 = partcounttask.compute()
    assert partcount2.shape[0] == nmax
    assert np.all(partcount2 == partcount[:nmax])

    # test idxlist
    idxlist = [3, 5, 7, 25200]
    partcounttask = snap.map_halo_operation(
        calculate_partcount, compute=False, chunksize=int(3e6), idxlist=idxlist
    )
    partcount2 = partcounttask.compute()
    assert partcount2.shape[0] == len(idxlist)
    assert np.all(partcount2 == partcount[idxlist])


# Test the map_halo_operation/grouped functionality
# todo: need to have catalog have more than 1 halo for this.
# def test_areposnapshot_halooperation(tngfile_dummy, gadgetcatalogfile_dummy):
#     path = tngfile_dummy.path
#     catalogpath = gadgetcatalogfile_dummy.path
#     halooperations(path, catalogpath)


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_areposnapshot_halooperation_realdata(testdatapath):
    halooperations(testdatapath)


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
    assert np.allclose(m[idxlist], m10)
