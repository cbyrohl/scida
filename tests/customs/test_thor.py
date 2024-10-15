from scida.convenience import load
from scida.interface import Dataset
from tests.testdata_properties import require_testdata_path


# for now we do individual photon types (original, peel, input) separately
@require_testdata_path("interface", only=["thor_mcrt/peel"])
def test_mcrt(testdatapath):
    ds = load(testdatapath)
    assert isinstance(ds, Dataset)
    assert len(ds.data.keys()) > 0
    print(ds.data.keys())
