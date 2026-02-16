import pytest

from scida import ArepoSnapshot
from scida.convenience import load
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("interface", only=["AURIGA66_snapshot"])
def test_series(testdatapath):
    obj = load(testdatapath)
    assert isinstance(obj, ArepoSnapshot)
