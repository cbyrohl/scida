import logging

import pytest

from scida.config import get_config


numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


@pytest.fixture(scope="function", autouse=True)
def cachedir(monkeypatch, tmp_path_factory):
    """Isolate cache for every test and reload config."""
    path = tmp_path_factory.mktemp("cache")
    monkeypatch.setenv("SCIDA_CACHE_PATH", str(path))
    get_config(reload=True)
    return path
