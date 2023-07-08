import logging

import pytest

from scida import ArepoSnapshot, load
from scida.config import get_config
from scida.interfaces.gadgetstyle import GadgetStyleSnapshot
from scida.series import DatasetSeries

flag_test_long = False  # Set to true to run time-taking tests.

scope_snapshot = "function"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "big: mark test as big")


@pytest.fixture(scope="session", autouse=True)
def set_logging():
    logging.root.setLevel(logging.DEBUG)


@pytest.fixture(scope="function")
def testdata_interface(request) -> GadgetStyleSnapshot:
    path = request.param
    yield GadgetStyleSnapshot(path)


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
    snappath, grouppath = request.param[0], request.param[1]
    snp = load(snappath, catalog=grouppath)
    return snp


@pytest.fixture(scope="function")
def testdata_illustrissnapshot_withcatalog(request) -> ArepoSnapshot:
    snappath, grouppath = request.param[0], request.param[1]
    snp = load(snappath, catalog=grouppath)
    return snp


@pytest.fixture(scope="function", autouse=True)
def cachedir(monkeypatch, tmp_path_factory):
    path = tmp_path_factory.mktemp("cache")
    monkeypatch.setenv("SCIDA_CACHE_PATH", str(path))
    return path


@pytest.fixture(scope="function", autouse=True)
def cleancache(cachedir):
    """Always start with empty cache."""
    get_config(reload=True)
    return cachedir
