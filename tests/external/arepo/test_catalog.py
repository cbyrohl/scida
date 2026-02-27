import pytest

from scida import load
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_groupids_for_particles(testdatapath):
    """Test group/subhalo ID assignment for particles."""
    obj = load(testdatapath, units=True)
    pdata = obj.data["PartType0"]
    assert pdata["GroupID"][0].compute() == 0
    assert pdata["LocalSubhaloID"][0].compute() == 0
    assert pdata["SubhaloID"][0].compute() == 0
    assert pdata["GroupID"][-1].compute() == obj.misc["unboundID"]
    assert pdata["LocalSubhaloID"][-1].compute() == obj.misc["unboundID"]
    assert pdata["SubhaloID"][-1].compute() == obj.misc["unboundID"]
