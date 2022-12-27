from astrodask.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface")
def test_interface_load(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
