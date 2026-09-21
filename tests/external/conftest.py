import pytest

from scida import ArepoSnapshot, load
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot
from scida.series import DatasetSeries


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
