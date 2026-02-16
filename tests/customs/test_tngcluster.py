import pytest

from scida import TNGClusterSnapshot
from scida.convenience import load
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("interface", only=["TNGCluster_snapshot_z0_minimal"])
def test_snapshot(testdatapath):
    """Test loading of a full snapshot"""
    obj = load(testdatapath)
    assert isinstance(obj, TNGClusterSnapshot)
