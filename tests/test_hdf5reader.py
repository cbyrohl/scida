import tempfile

import dask
import dask.array as da
import h5py
import numpy as np
import pytest


def test_hdf5_read():
    # create dummy file and dataset
    data = np.ones((10, 10))
    with tempfile.NamedTemporaryFile() as tmpfile:
        with h5py.File(tmpfile.name, "w") as f:
            f.create_dataset("test", data=data)

        # read dataset with dask
        dask.config.set(scheduler="processes")

        with h5py.File(tmpfile.name, "r") as f:
            dset = da.from_array(f["test"], chunks=(5, 5))
            # should fail with "TypeError: h5py objects cannot be pickled"
            pytest.raises(TypeError, dset.compute)
