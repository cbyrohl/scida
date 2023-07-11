from testdata_properties import require_testdata_path

from scida import GadgetStyleSnapshot


@require_testdata_path("interface", only=["TNG50-4_group"])
def test_grp(testdatapath):
    grp = GadgetStyleSnapshot(testdatapath, virtualcache=False)
    assert len(grp.header["Ngroups_ThisFile"]) == 11
    pass
