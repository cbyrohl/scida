import logging

import pytest

from scida import load
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("interface")
def test_allunitsdiscovered(testdatapath, caplog):
    ds = load(testdatapath)
    caplog.set_level(logging.DEBUG)
    if "gaia" in testdatapath:
        return
    if "thor" in testdatapath:
        return
    assert (
        "Cannot determine units from neither unit file nor metadata" not in caplog.text
    )
    assert not ds.missing_units()
