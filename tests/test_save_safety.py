import os
import sys
import unittest
import unittest.mock as mock


# Robust mock for packages
def mock_package(name):
    m = mock.MagicMock()
    m.__path__ = []
    sys.modules[name] = m
    return m


mock_package("zarr")
mock_package("dask")
mock_package("dask.array")
mock_package("dask.dataframe")
mock_package("dask.distributed")
mock_package("scida.io")
mock_package("requests")
mock_package("matplotlib")
mock_package("matplotlib.pyplot")
mock_package("pint")
mock_package("h5py")
mock_package("numpy")
mock_package("numpy.typing")
mock_package("tqdm")
mock_package("astropy")
mock_package("astropy.io")
mock_package("numba")
mock_package("yaml")
mock_package("distributed")

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

# We want to avoid scida/__init__.py if possible, but scida.interface imports scida.io
# which is a subpackage of scida.

from scida.interface import BaseDataset


class TestVulnerabilityFix(unittest.TestCase):
    def test_save_safety_check(self):
        # Setup mock Dataset
        ds = BaseDataset.__new__(BaseDataset)
        ds.data = mock.MagicMock()
        ds.data.keys.return_value = []
        ds.__dict__ = {}

        fname = "test_target"

        # Mock os functions
        with mock.patch("os.path.exists") as mock_exists, mock.patch(
            "os.path.isdir"
        ) as mock_isdir, mock.patch("os.listdir") as mock_listdir:

            # Case 1: non-empty, non-zarr directory -> should raise Exception
            mock_exists.side_effect = lambda x: True if x == fname else False
            mock_isdir.return_value = True
            mock_listdir.return_value = ["some_file"]

            with self.assertRaisesRegex(Exception, "is not a zarr group"):
                ds.save(fname, overwrite=True)

            # Case 2: empty directory -> should proceed
            mock_listdir.return_value = []
            try:
                ds.save(fname, overwrite=True)
            except Exception as e:
                if "is not a zarr group" in str(e):
                    self.fail("Safety check failed on empty directory")

            # Case 3: zarr group (has .zgroup) -> should proceed
            mock_listdir.return_value = [".zgroup"]
            mock_exists.side_effect = lambda x: (
                True if x in [fname, os.path.join(fname, ".zgroup")] else False
            )
            try:
                ds.save(fname, overwrite=True)
            except Exception as e:
                if "is not a zarr group" in str(e):
                    self.fail("Safety check failed on valid Zarr group")

            # Case 4: zarr array (has .zarray) -> should proceed
            mock_listdir.return_value = [".zarray"]
            mock_exists.side_effect = lambda x: (
                True if x in [fname, os.path.join(fname, ".zarray")] else False
            )
            try:
                ds.save(fname, overwrite=True)
            except Exception as e:
                if "is not a zarr group" in str(e):
                    self.fail("Safety check failed on valid Zarr array")

            # Case 5: overwrite=False -> should proceed even if non-zarr
            mock_listdir.return_value = ["some_file"]
            mock_exists.side_effect = lambda x: True if x == fname else False
            try:
                ds.save(fname, overwrite=False)
            except Exception as e:
                if "is not a zarr group" in str(e):
                    self.fail("Safety check incorrectly triggered when overwrite=False")


if __name__ == "__main__":
    unittest.main()
