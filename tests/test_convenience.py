from astrodask.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface")
def test_load(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
    assert obj.data is not None
    print(obj.metadata)


# TODO: Need to write test for TNG50-4_snapshot+TNG50-4_group using require_testdata(..., only=...)
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_load_areposnap(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
    assert obj.data is not None
    assert all([k in obj.data for k in ["PartType%i" % i for i in [0, 1, 3, 4, 5]]])
    assert "Coordinates" in obj.data["PartType0"]
    assert obj.header["BoxSize"] == 35000.0
