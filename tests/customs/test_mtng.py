import pytest

from scida import ArepoSimulation, MTNGArepoSnapshot
from scida.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("series", only=["MTNGarepo_1080hydro_simulation"])
def test_fullsimulation(testdatapath):
    series = load(
        testdatapath, catalog=False
    )  # catalog not working for minimal tesdata
    assert isinstance(series, ArepoSimulation)
    ds = series.get_dataset(redshift=0.0)
    ds.evaluate_lazy()
    assert isinstance(ds, MTNGArepoSnapshot)
    print(ds)
    assert pytest.approx(ds.redshift) == 0.0
    assert ds.cosmology is not None
    print(ds.cosmology)


@require_testdata_path("interface", only=["MTNGarepo_270hydro_snapshot_z2"])
def test_fullsnapshot(testdatapath):
    """Test loading of a full snapshot"""
    obj = load(testdatapath)
    assert isinstance(obj, MTNGArepoSnapshot)
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
