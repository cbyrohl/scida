import pathlib

import dask
import h5py
import numpy as np
import pytest
import zarr

import scida
from scida.config import get_config
from scida.io import load
from tests.helpers import (
    DummyGadgetSnapshotFile,
    write_gadget_testfile,
    write_hdf5flat_testfile,
)


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


@pytest.mark.integration
@pytest.mark.parametrize("extension", ["hdf5", "zarr"])
def test_load_distinguishes_field_paths_with_underscores(tmp_path, extension):
    path = tmp_path / f"input.{extension}"
    if extension == "hdf5":
        with h5py.File(path, "w") as source:
            source["a/b_c"] = [0, 1, 2]
            source["a_b/c"] = [10, 11, 12]
    else:
        source = zarr.open_group(str(path), mode="w")
        source["a/b_c"] = [0, 1, 2]
        source["a_b/c"] = [10, 11, 12]

    dataset = scida.load(path, units=False)
    first, second = dask.compute(dataset["a"]["b_c"], dataset["a_b"]["c"])
    np.testing.assert_array_equal(first, [0, 1, 2])
    np.testing.assert_array_equal(second, [10, 11, 12])

    output = tmp_path / "output.zarr"
    dataset.save(str(output))
    saved = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(saved["a/b_c"][:], [0, 1, 2])
    np.testing.assert_array_equal(saved["a_b/c"][:], [10, 11, 12])


@pytest.mark.integration
def test_load_distinguishes_fields_with_duplicate_stored_dask_names(tmp_path):
    path = tmp_path / "input.hdf5"
    with h5py.File(path, "w") as source:
        source["first"] = [0, 1, 2]
        source["second"] = [10, 11, 12]
        names = source.create_group("__dask_tokenize__")
        names.attrs["first"] = "stored-name"
        names.attrs["second"] = "stored-name"

    dataset = scida.load(path, units=False)
    first, second = dask.compute(dataset["first"], dataset["second"])
    np.testing.assert_array_equal(first, [0, 1, 2])
    np.testing.assert_array_equal(second, [10, 11, 12])


@pytest.mark.integration
def test_load_distinguishes_snapshot_prefixes_in_same_directory(tmp_path, monkeypatch):
    monkeypatch.setitem(get_config(), "nthreads", 1)
    snapshot = DummyGadgetSnapshotFile()
    particle_counts = [1, 0, 0, 0, 0, 0]
    snapshot.create_dummyheader(particle_counts)
    snapshot.create_dummyfieldcontainer(particle_counts)
    snapshot.header["NumFilesPerSnapshot"] = 2
    snapshot.header["NumPart_Total"] = [2, 0, 0, 0, 0, 0]

    for prefix, masses in (("snap_000", [0.0, 1.0]), ("snap_001", [10.0, 11.0])):
        for chunk, mass in enumerate(masses):
            snapshot.particles["PartType0"]["Masses"] = np.array([mass])
            snapshot.write(tmp_path / f"{prefix}.{chunk}.hdf5")

    first = scida.load(tmp_path, fileprefix="snap_000", units=False)
    second = scida.load(tmp_path, fileprefix="snap_001", units=False)
    first_masses, second_masses = dask.compute(
        first["PartType0"]["Masses"], second["PartType0"]["Masses"]
    )

    np.testing.assert_array_equal(first_masses, [0.0, 1.0])
    np.testing.assert_array_equal(second_masses, [10.0, 11.0])
