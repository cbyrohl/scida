from astrodask.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot", "TNG50-4_group"])
def test_coordsalias(testdatapath):
    obj = load(testdatapath)
    assert obj.file is not None
    assert obj.data is not None
    for ptype in ["PartType%i" % i for i in range(6)] + ["Group", "Subhalo"]:
        if ptype not in obj.data or ptype == "PartType3":
            continue
        coords = obj.get_coords(parttype=ptype)
        assert coords is not None
