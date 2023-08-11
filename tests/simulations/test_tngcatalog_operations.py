from testdata_properties import require_testdata_path

from scida import load


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_groupids_for_particles(testdatapath):
    obj = load(testdatapath, units=True)
    pdata = obj.data["PartType0"]
    # still missing some tests
    # first particles should belong to first halo and subhalo in TNG
    assert pdata["GroupID"][0].compute() == 0
    assert pdata["LocalSubhaloID"][0].compute() == 0
    assert pdata["SubhaloID"][0].compute() == 0
    # last particles should be unbound
    assert pdata["GroupID"][-1].compute() == obj.misc["unboundID"]
    assert pdata["LocalSubhaloID"][-1].compute() == obj.misc["unboundID"]
    assert pdata["SubhaloID"][-1].compute() == obj.misc["unboundID"]
