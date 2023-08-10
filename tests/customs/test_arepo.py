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
