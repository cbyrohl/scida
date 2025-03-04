import logging

from scida import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface")
def test_allunitsdiscovered(testdatapath, caplog):
    ds = load(testdatapath)
    caplog.set_level(logging.DEBUG)
    if "gaia" in testdatapath:
        # skip gaia for now; because invalid pint units (like mag) are not yet handled
        return
    if "thor" in testdatapath:
        # no metadata units in thor yet
        return
    assert "Cannot determine units from neither unit file nor metadata" not in caplog.text
    assert not ds.missing_units()
