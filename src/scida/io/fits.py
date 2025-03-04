"""
FITS file reader for scida
"""

import dask
import dask.array as da
import numpy as np
from dask import delayed

from scida.misc import parse_size


def fitsrecords_to_daskarrays(fitsrecords):
    """
    Convert a FITS record array to a dictionary of dask arrays.
    Parameters
    ----------
    fitsrecords: np.ndarray
        FITS record array

    Returns
    -------
    dict
        dictionary of dask arrays

    """
    load_arr = delayed(lambda slc, field: fitsrecords[slc][field])
    shape = fitsrecords.shape
    darrdict = {}
    csize = dask.config.get("array.chunk-size")

    csize = parse_size(csize)  # need int

    nbytes_dtype_max = 1
    for fieldname in fitsrecords.dtype.names:
        nbytes_dtype = fitsrecords.dtype[fieldname].itemsize
        nbytes_dtype_max = max(nbytes_dtype_max, nbytes_dtype)
    chunksize = csize // nbytes_dtype_max

    for fieldname in fitsrecords.dtype.names:
        chunks = []
        for index in range(0, shape[-1], chunksize):
            dtype = fitsrecords.dtype[fieldname]
            chunk_size = min(chunksize, shape[-1] - index)
            slc = slice(index, index + chunk_size)
            shp = (chunk_size,)
            if dtype.subdtype is not None:
                # for now, we expect this to be void type
                assert dtype.type is np.void
                break  # do not handle void type for now => skip field
                # shp = shp + dtype.subdtype[0].shape
                # dtype = dtype.subdtype[0].base
            chunk = da.from_delayed(load_arr(slc, fieldname), shape=shp, dtype=dtype)
            chunks.append(chunk)
        if len(chunks) > 0:
            darrdict[fieldname] = da.concatenate(chunks, axis=0)
    return darrdict
