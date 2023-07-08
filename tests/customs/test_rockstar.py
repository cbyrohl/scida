from scida.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("rockstar")
def test_load_check_seriestype(testdatapath):
    # this test intentionally ignores all additional hints from config files
    # catch_exception = True allows us better control for debugging
    obj = load(testdatapath)
    print(obj.info())
    print(obj.data.keys())
    pass
