import pytest

from astrodask.config import get_config
from astrodask.interface import BaseSnapshot
from astrodask.interfaces.arepo import ArepoSnapshot  # ArepoSnapshotWithUnits
from astrodask.interfaces.illustris import IllustrisSnapshot
from astrodask.series import DatasetSeries

flag_test_long = False  # Set to true to run time-taking tests.

scope_snapshot = "function"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "big: mark test as big")


@pytest.fixture(scope="function")
def testdata_interface(request) -> BaseSnapshot:
    path = request.param
    yield BaseSnapshot(path)


@pytest.fixture(scope="function")
def testdata_series(request) -> DatasetSeries:
    path = request.param
    return DatasetSeries(path)


@pytest.fixture(scope="function")
def testdata_areposnapshot(request) -> ArepoSnapshot:
    path = request.param
    return ArepoSnapshot(path)


@pytest.fixture(scope="function")
def testdata_areposnapshot_withcatalog(request) -> ArepoSnapshot:
    tng50_snappath, tng50_grouppath = request.param[0], request.param[1]
    try:
        snp = ArepoSnapshot(tng50_snappath, catalog=tng50_grouppath)
    except ValueError:
        snp = ArepoSnapshot(
            tng50_snappath,
            catalog=tng50_grouppath,
            catalog_kwargs=dict(fileprefix="group"),
        )
    return snp


# @pytest.fixture(scope="function")
# def testdata_areposnapshot_withcatalog_andunits(request) -> ArepoSnapshotWithUnits:
#    tng50_snappath, tng50_grouppath = request.param[0], request.param[1]
#    return ArepoSnapshotWithUnits(tng50_snappath, catalog=tng50_grouppath)


@pytest.fixture(scope="function")
def testdata_illustrissnapshot(request) -> IllustrisSnapshot:
    tng_snappath = request.param
    return IllustrisSnapshot(tng_snappath)


@pytest.fixture(scope="function")
def testdata_illustrisgroup(request) -> IllustrisSnapshot:
    tng_snappath = request.param
    return IllustrisSnapshot(tng_snappath, fileprefix="group")


@pytest.fixture(scope="function")
def testdata_illustrissnapshot_withcatalog(request) -> IllustrisSnapshot:
    tng_snappath, tng_grouppath = request.param[0], request.param[1]
    return IllustrisSnapshot(tng_snappath, catalog=tng_grouppath)


@pytest.fixture(scope="function", autouse=True)
def cachedir(monkeypatch, tmp_path_factory):
    path = tmp_path_factory.mktemp("cache")
    monkeypatch.setenv("ASTRODASK_CACHE_PATH", str(path))
    return path


@pytest.fixture(scope="function", autouse=True)
def cleancache(cachedir):
    """Always start with empty cache."""
    get_config(reload=True)
    return cachedir
