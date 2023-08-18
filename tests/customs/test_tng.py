import logging

from scida import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_simulation_tng50(testdatapath, caplog):
    caplog.set_level(logging.WARNING)
    load(testdatapath)
    assert "Unit mismatch" not in caplog.text
