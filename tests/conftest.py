import pytest
import requests
import sys
import tarfile
import os

import astrodask.interface
from astrodask.interface import BaseSnapshot
from astrodask.interfaces.arepo import ArepoSnapshot
from tests import path, gpath


flag_test_long = False  # Set to true to run time-taking tests.
force_download = False  # forcing download of TNG50 snapshots over local discovery

scope_snapshot = "function"


def pytest_configure(config):
    os.environ["cachedir"] = "TEST"


def download_and_extract(url, path, progress=False):
    with requests.get(url, stream=True) as r:
        ltot = int(r.headers.get("content-length"))
        with open(path, "wb") as f:
            dl = 0
            for data in r.iter_content(chunk_size=2**24):
                dl += len(data)
                f.write(data)
                d = int(32 * dl / ltot)
                if progress:
                    sys.stdout.write("\r Progress: [%s%s]" % ("=" * d, " " * (32 - d)))
                    sys.stdout.flush()
    tar = tarfile.open(path, "r:gz")
    tar.extractall(path.parents[0])
    foldername = tar.getmembers()[
        0
    ].name  # the parent folder of the extracted TNG tar.gz
    os.remove(path)  # remove downloaded tar.gz
    tar.close()
    return os.path.join(path.parents[0], foldername)


@pytest.fixture(scope="session", autouse=True)
def tng50_snappath(tmp_path_factory):
    url = "https://heibox.uni-heidelberg.de/f/dc65a8c75220477eb62d/?dl=1"
    filename = "snapdir.tar.gz"
    if os.path.exists(path) and not force_download:
        datapath = path
    else:
        fn = tmp_path_factory.mktemp("data") / filename
        datapath = download_and_extract(url, fn)
    return datapath


@pytest.fixture(scope="session", autouse=True)
def tng50_grouppath(tmp_path_factory):
    url = "https://heibox.uni-heidelberg.de/f/ff27fb6975fb4dc391ef/?dl=1"
    filename = "groups.tar.gz"
    # Check if locally available, otherwise download.
    if os.path.exists(gpath) and not force_download:
        datapath = gpath
    else:
        fn = tmp_path_factory.mktemp("data") / filename
        datapath = download_and_extract(url, fn)
    return datapath


@pytest.fixture(scope="function")
def snp(tng50_snappath) -> BaseSnapshot:
    return BaseSnapshot(tng50_snappath)


@pytest.fixture(scope="function")
def grp(tng50_grouppath) -> BaseSnapshot:
    return BaseSnapshot(tng50_grouppath)


@pytest.fixture(scope="function")
def areposnp(tng50_snappath) -> ArepoSnapshot:
    from astrodask.interfaces.arepo import ArepoSnapshot

    return ArepoSnapshot(tng50_snappath)


@pytest.fixture(scope="function")
def areposnpfull(tng50_snappath, tng50_grouppath) -> ArepoSnapshot:
    return ArepoSnapshot(tng50_snappath, catalog=tng50_grouppath)
