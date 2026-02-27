import pathlib

import pytest

from scida.io import load

from tests.helpers import write_gadget_testfile, write_hdf5flat_testfile


@pytest.mark.integration
def test_ioload_hdf5flat(tmp_path):
    p = pathlib.Path(tmp_path) / "test.hdf5"
    write_hdf5flat_testfile(p)
    res = load(p)
    fcc, metadata, hf, tmpfile = res
    assert "field1" in fcc
    assert "field2" in fcc


@pytest.mark.integration
def test_ioload_gadget(tmp_path):
    p = pathlib.Path(tmp_path) / "test.hdf5"
    write_gadget_testfile(p)
    res = load(p)
    fcc, metadata, hf, tmpfile = res
    assert "BoxSize" in metadata["/Header"]
    assert "PartType0" in fcc
