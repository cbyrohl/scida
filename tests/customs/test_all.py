import logging

from scida import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface")
def test_allunitsdiscovered(testdatapath, caplog):
    load(testdatapath)
    caplog.set_level(logging.DEBUG)
    assert (
        "Cannot determine units from neither unit file nor metadata" not in caplog.text
    )
