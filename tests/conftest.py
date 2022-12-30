import os

import pytest

from astrodask.interface import BaseSnapshot
from astrodask.interfaces.arepo import ArepoSnapshot  # ArepoSnapshotWithUnits
from astrodask.interfaces.illustris import IllustrisSnapshot
from astrodask.series import DatasetSeries

flag_test_long = False  # Set to true to run time-taking tests.

scope_snapshot = "function"


def pytest_configure(config):
    os.environ["cachedir"] = "TEST"
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "big: mark test as big")


# def download_and_extract(url, path, progress=False):
#    with requests.get(url, stream=True) as r:
#        ltot = int(r.headers.get("content-length"))
#        with open(path, "wb") as f:
#            dl = 0
#            for data in r.iter_content(chunk_size=2**24):
#                dl += len(data)
#                f.write(data)
#                d = int(32 * dl / ltot)
#                if progress:
#                    sys.stdout.write("\r Progress: [%s%s]" % ("=" * d, " " * (32 - d)))
#                    sys.stdout.flush()
#    tar = tarfile.open(path, "r:gz")
#    tar.extractall(path.parents[0])
#    foldername = tar.getmembers()[
#        0
#    ].name  # the parent folder of the extracted TNG tar.gz
#    os.remove(path)  # remove downloaded tar.gz
#    tar.close()
#    return os.path.join(path.parents[0], foldername)


# @pytest.fixture(scope="session", autouse=True)
# def tng50_snappath(tmp_path_factory):
#    url = "https://heibox.uni-heidelberg.de/f/dc65a8c75220477eb62d/?dl=1"
#    filename = "snapdir.tar.gz"
#    if os.path.exists(path) and not force_download:
#        datapath = path
#    else:
#        fn = tmp_path_factory.mktemp("data") / filename
#        datapath = download_and_extract(url, fn)
#    return datapath


# @pytest.fixture(scope="session", autouse=True)
# def tng50_grouppath(tmp_path_factory):
#    url = "https://heibox.uni-heidelberg.de/f/ff27fb6975fb4dc391ef/?dl=1"
#    filename = "groups.tar.gz"
#    # Check if locally available, otherwise download.
#    if os.path.exists(gpath) and not force_download:
#        datapath = gpath
#    else:
#        fn = tmp_path_factory.mktemp("data") / filename
#        datapath = download_and_extract(url, fn)
#    return datapath


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
    return ArepoSnapshot(tng50_snappath, catalog=tng50_grouppath)


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
    monkeypatch.setenv("ASTRODASK_CACHEDIR", str(path))
