import pathlib

from scida.io import load

from .helpers import write_gadget_testfile, write_hdf5flat_testfile


def test_ioload_hdf5flat(tmp_path):
    # the flat test file is a single hdf5 file with fields in the root group
    p = pathlib.Path(tmp_path) / "test.hdf5"
    write_hdf5flat_testfile(p)

    res = load(p)
    fcc, metadata, hf, tmpfile = res

    assert "field1" in fcc
    assert "field2" in fcc


def test_ioload_gadget(tmp_path):
    # the gadget test file is a single hdf5 file with fields nested into groups
    p = pathlib.Path(tmp_path) / "test.hdf5"
    write_gadget_testfile(p)

    res = load(p)
    fcc, metadata, hf, tmpfile = res

    assert "BoxSize" in metadata["/"]
    assert "PartType0" in fcc
