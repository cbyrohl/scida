"""Some utility functions"""

import zarr

from .interface import Dataset


def copy_to_zarr(fp_in, fp_out, compressor=None):
    """
    Reads and converts a scida Dataset to a zarr object on disk

    Parameters
    ----------
    fp_in: str
        object path to convert
    fp_out: str
        output path
    compressor:
        zarr compressor to use, see https://zarr.readthedocs.io/en/stable/tutorial.html#compressors

    Returns
    -------
    None
    """
    ds = Dataset(fp_in)
    compressor_dflt = zarr.storage.default_compressor
    zarr.storage.default_compressor = compressor
    ds.save(fp_out, cast_uints=True)
    zarr.storage.default_compressor = compressor_dflt
