import dask
from dask import delayed
import dask.array as da
import numpy as np

_sizeunits = {"B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12,
         "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}
_sizeunits = {k.lower(): v for k, v in _sizeunits.items()}


def parse_size(size):
    idx = 0
    for c in size:
        if c.isnumeric():
            continue
        idx += 1
    number = size[:idx]
    unit = size[idx:]
    return int(float(number) * _sizeunits[unit.lower().strip()])

def fitsrecords_to_daskarrays(fitsrecords):
    load_arr = delayed(lambda slc, field: fitsrecords[slc][field])
    shape = fitsrecords.shape
    darrdict = {}
    csize = dask.config.get('array.chunk-size')

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
            chunk = da.from_delayed(
                load_arr(slc, fieldname),
                shape=shp,
                dtype=dtype
            )
            chunks.append(chunk)
        if len(chunks) > 0:
            darrdict[fieldname] = da.concatenate(chunks, axis=0)
    return darrdict
