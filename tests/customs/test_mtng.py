from scida import MTNGArepoSnapshot
from scida.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["MTNGarepo_270hydro_snapshot_z2"])
def test_fullsnapshot(testdatapath):
    """Test loading of a full snapshot"""
    obj = load(testdatapath)
    # print(obj.info())
    containernames = [
        "PartType0",
        "PartType1",
        "PartType4",
        "PartType5",
        "Group",
        "Subhalo",
    ]
    containernames += ["PartType1_mostbound"]
    assert all([k in obj.data for k in containernames])


@require_testdata_path("interface", only=["MTNGarepo_270hydro_snapshot_z10"])
def test_partialsnapshot(testdatapath):
    """Test loading of a partial snapshot"""
    obj = load(testdatapath)
    assert isinstance(obj, MTNGArepoSnapshot)
    containernames = ["Group", "Subhalo", "PartType1_mostbound"]
    containernames_excl = [
        "PartType0",
        "PartType1",
        "PartType3",
        "PartType4",
        "PartType5",
    ]
    assert all([k in obj.data for k in containernames])
    assert not any([k in obj.data for k in containernames_excl])
