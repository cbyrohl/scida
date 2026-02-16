from tests.helpers import DummyGadgetCatalogFile, DummyGadgetSnapshotFile, DummyTNGFile
import pytest


@pytest.fixture
def gadgetfile_dummy(tmp_path):
    dummy = DummyGadgetSnapshotFile()
    dummy.write(tmp_path / "dummy_gadgetfile.hdf5")
    return dummy


@pytest.fixture
def tngfile_dummy(tmp_path):
    dummy = DummyTNGFile()
    dummy.write(tmp_path / "dummy_tngfile.hdf5")
    return dummy


@pytest.fixture
def gadgetcatalogfile_dummy(tmp_path):
    dummy = DummyGadgetCatalogFile()
    dummy.write(tmp_path / "dummy_gadgetcatalogfile.hdf5")
    return dummy
