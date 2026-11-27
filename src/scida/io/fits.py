"""
FITS file reader for scida
"""

from __future__ import annotations

import logging

import dask
import dask.array as da
from dask import delayed

from scida.misc import parse_size

log = logging.getLogger(__name__)


def fitsrecords_to_daskarrays(fitsrecords):
    """
    Convert a FITS record array to a dictionary of dask arrays.
    Parameters
    ----------
    fitsrecords: numpy.ndarray
        FITS record array

    Returns
    -------
    dict
        dictionary of dask arrays

    """
    load_arr = delayed(lambda slc, field: fitsrecords[slc][field])
    nrows = len(fitsrecords)
    # Inspect the decoded shape and dtype without reading table rows. FITS
    # storage dtypes can differ from their decoded values (e.g. logical columns).
    empty_records = fitsrecords[:0]
    darrdict = {}
    csize = dask.config.get("array.chunk-size")

    csize = parse_size(csize)  # need int

    nbytes_dtype_max = 1
    for fieldname in fitsrecords.dtype.names:
        nbytes_dtype = fitsrecords.dtype[fieldname].itemsize
        nbytes_dtype_max = max(nbytes_dtype_max, nbytes_dtype)
    chunksize = max(1, csize // nbytes_dtype_max)

    for fieldname in fitsrecords.dtype.names:
        column = empty_records[fieldname]
        if column.dtype.hasobject:
            log.warning(
                "Skipping unsupported variable-length FITS column '%s'.", fieldname
            )
            continue
        if nrows == 0:
            darrdict[fieldname] = da.from_array(column, chunks=column.shape)
            continue

        chunks = []
        for index in range(0, nrows, chunksize):
            chunk_size = min(chunksize, nrows - index)
            slc = slice(index, index + chunk_size)
            shp = (chunk_size,) + column.shape[1:]
            chunk = da.from_delayed(
                load_arr(slc, fieldname), shape=shp, dtype=column.dtype
            )
            chunks.append(chunk)
        darrdict[fieldname] = da.concatenate(chunks, axis=0)
    return darrdict
