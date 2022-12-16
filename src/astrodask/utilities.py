import zarr
from .interface import Dataset


def copy_to_zarr(fp_in,fp_out,compressor=None):
    ds = Dataset(fp_in)
    compressor_dflt = zarr.storage.default_compressor
    zarr.storage.default_compressor = compressor
    ds.save(fp_out,cast_uints=True)
    zarr.storage.default_compressor = compressor_dflt