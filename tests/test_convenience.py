from tests.testdata_properties import require_testdata_path

from astrodask.convenience import load


@require_testdata_path("interface")
def test_interface_load(testdatapath_interface):
    obj = load(testdatapath_interface)
    assert obj.file is not None
