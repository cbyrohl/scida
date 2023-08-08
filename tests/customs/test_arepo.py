import dask.array as da

from scida import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_selector(testdatapath):
    hid = 41
    obj = load(testdatapath, units=False)
    d = obj.return_data(haloID=hid)
    assert da.all(d["PartType0"]["GroupID"] == hid).compute()

    d = obj.return_data(unbound=True)
    # unbound gas has groupid = max_int64 = 9223372036854775807
    assert da.all(d["PartType0"]["GroupID"] == 9223372036854775807).compute()
