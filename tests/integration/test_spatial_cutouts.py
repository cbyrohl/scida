import dask.array as da
import numpy as np
import pint
import pytest

from scida import load
from scida.misc import rectangular_cutout_mask
from tests.helpers import DummyTNGFile


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["numpy", "dask"])
@pytest.mark.parametrize("withunits", [False, True])
def test_rectangular_cutout_across_periodic_boundary(backend, withunits):
    coordinates = np.array([[2, 5, 5], [9.5, 5, 5], [5, 5, 5], [1, 7, 5]])
    center = [1, 5, 5]
    width = [4, 4, 4]
    boxsize = 10
    if backend == "dask":
        coordinates = da.from_array(coordinates, chunks=(2, 3))
    if withunits:
        ureg = pint.UnitRegistry()
        coordinates = ureg.Quantity(coordinates, "kpc")
        center = ureg.Quantity([1000, 5000, 5000], "pc")
        width = ureg.Quantity([4000, 4000, 4000], "pc")
        boxsize = ureg.Quantity(10, "kpc")

    mask = rectangular_cutout_mask(
        center, width, coordinates, boxsize=boxsize, backend=backend, chunksize=2
    )

    if backend == "dask":
        assert isinstance(mask, da.Array)
        mask = mask.compute()
    np.testing.assert_array_equal(mask, [True, True, False, False])


@pytest.mark.integration
@pytest.mark.parametrize("units", [True, "cgs"])
def test_snapshot_cutout_uses_native_box_size(tmp_path, units):
    path = tmp_path / "snap_042.hdf5"
    source = DummyTNGFile()
    source.header["NumFilesPerSnapshot"] = 1
    source.particles["PartType0"]["Coordinates"][0] = [
        source.header["BoxSize"] - 1,
        0,
        0,
    ]
    source.write(path)
    snapshot = load(path, units=units)
    center = snapshot.ureg.Quantity([1, 0, 0], "code_length")
    width = snapshot.ureg.Quantity([6, 6, 6], "code_length")

    mask = snapshot.rectangular_cutout_mask(center, width)

    assert isinstance(mask, da.Array)
    np.testing.assert_array_equal(mask.compute(), [True])
