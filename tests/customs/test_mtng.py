from scida.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["MTNGarepo_270hydro_snapshot_z2"])
def test_snapshot(testdatapath):
    obj = load(testdatapath)
    # print(obj.info())
    containernames = [
        "PartType0",
        "PartType1",
        "PartType3",
        "PartType4",
        "PartType5",
        "Group",
        "Subhalo",
    ]
    containernames += ["PartType1_mostbound"]
    assert all([k in obj.data for k in containernames])
