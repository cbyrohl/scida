"""Regression tests for specific GitHub issues."""
import os
import pathlib

import h5py
import numpy as np
import pytest

from scida import ArepoSimulation, ArepoSnapshot, load
from scida.convenience import find_path
from tests.helpers import write_gadget_testfile
from tests.testdata_properties import require_testdata_path


@pytest.mark.unit
def test_find_path_nonexistent():
    """Regression test for GitHub issue #78.

    Ensure that non-existent paths in default datafolders do not cause errors,
    and that looking up a non-existent dataset raises a proper ValueError.
    """
    path = "SomeDataset"
    with pytest.raises(ValueError) as e:
        find_path(path)
    assert "unknown" in str(e.value)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_selector_with_pint_units(testdatapath):
    """Regression test for GitHub issue #59.

    Selector must handle pint Quantities when units=True.
    """
    obj = load(testdatapath, units=False)
    _ = obj.return_data(haloID=42)
    obj = load(testdatapath, units=True)
    _ = obj.return_data(haloID=42)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG100-3_snapshot_z0_minimal"])
def test_tng_unit_identification(testdatapath, tmp_path):
    """Regression test for GitHub issue #63.

    TNG100-2/TNG100-3 identification was broken, causing missing units in catalogs.
    Also verifies: 1) unknown units labeled 'unknown', 2) IDs are not pint quantities.
    """
    ds = load(testdatapath, units=True)
    assert isinstance(ds, ArepoSnapshot)
    gas = ds.data["PartType0"]
    coords = gas["Coordinates"]
    assert coords.to_base_units().units == ds.ureg("centimeter")
    grp = ds.data["Group"]
    coords = grp["GroupPos"]
    assert coords.to_base_units().units == ds.ureg("centimeter")
    assert not hasattr(ds.data["PartType0"]["ParticleIDs"], "units")

    p = pathlib.Path(tmp_path) / "test.hdf5"
    write_gadget_testfile(p)
    with h5py.File(p, "r+") as hf:
        hf["PartType0"]["ExtraField"] = np.zeros((1,), dtype=float)
    ds = load(p, units=True)
    assert ds.data["PartType0"]["ExtraField"].units == "unknown"


@pytest.mark.external
@require_testdata_path("series", only=["TNGvariation_simulation"])
def test_series_output_subfolder(testdatapath, tmp_path):
    """Regression test for GitHub issue #88.

    Passing "output" subfolder instead of base folder should still work.
    Also verifies ~ is resolved as home folder.
    """
    srs = load(testdatapath, units=True)
    assert isinstance(srs, ArepoSimulation)
    testdatapath += "/output"
    srs = load(testdatapath, units=True)
    assert isinstance(srs, ArepoSimulation)
    path = "~/tmp_test"
    abspath = os.path.expanduser(path)
    os.makedirs(abspath, exist_ok=True)
    assert find_path(path).startswith("/")


@pytest.mark.external
@require_testdata_path("interface", only=["MCST_ST8_snapshot_z3"])
def test_info_cosmological_output(testdatapath, capsys):
    """Regression test for GitHub issue #185.

    Verify info() output includes cosmological information for cosmological snapshots.
    """
    obj = load(testdatapath, units=True)
    particles = obj.return_data(haloID=0)
    print(particles)
    obj.info()
    captured = capsys.readouterr()
    assert "=== Cosmological Simulation ===" in captured.out
    assert "z =" in captured.out


@pytest.mark.external
@require_testdata_path("series", only=["arepoturbbox_series"])
def test_noncosmo_simulation(testdatapath, capsys):
    """Regression test for GitHub issue #187.

    Non-cosmological simulations should not show redshift info.
    """
    sim = load(testdatapath)
    assert isinstance(sim, ArepoSimulation)
    ds = sim.get_dataset(0)
    assert isinstance(ds, ArepoSnapshot)
    veldiv = ds.data["PartType0"]["VelocityDivergence"][0].compute()
    assert veldiv == 0.0
    assert "Cosmology" not in str(type(ds))
    sim.info()
    captured = capsys.readouterr()
    assert "=== metadata ===" in captured.out
    assert "redshift: " not in captured.out
    ds.info()
    captured = capsys.readouterr()
    assert "=== Cosmological Simulation ===" not in captured.out
    assert "z =" not in captured.out


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_virtual_cache_catalog(testdatapath, capsys, cachedir):
    """Regression test for GitHub issue #196.

    Verify virtual caching for catalog fields works correctly.
    """
    ds = load(testdatapath, units=True)
    h5file_catalog = ds.catalog.file
    dataset = h5file_catalog["Group"]["GroupPos"]
    assert dataset.is_virtual
    dataset = h5file_catalog["Group"]["GroupLenType"]
    assert not dataset.is_virtual
