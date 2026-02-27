import pytest

from scida.convenience import load
from scida.customs.rockstar.dataset import RockstarCatalog
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("rockstar")
def test_load(testdatapath):
    # this test intentionally ignores all additional hints from config files
    # catch_exception = True allows us better control for debugging
    obj = load(testdatapath)
    print(obj.info())
    print(obj.data.keys())
    assert isinstance(obj, RockstarCatalog)
