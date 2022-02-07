import pytest

from ..interface import BaseSnapshot
from ..interfaces.arepo import ArepoSnapshot
from . import path,gpath


flag_test_long = False  # Set to true to run time-taking tests.

scope_snapshot = "function"

@pytest.fixture(scope="function")
def snp() -> BaseSnapshot:
    return BaseSnapshot(path)

@pytest.fixture(scope="function")
def grp() -> BaseSnapshot:
    return BaseSnapshot(gpath)

@pytest.fixture(scope="function")
def areposnp() -> ArepoSnapshot:
    return ArepoSnapshot(path)

@pytest.fixture(scope="function")
def areposnpfull() -> ArepoSnapshot:
    return ArepoSnapshot(path, catalog=gpath)
