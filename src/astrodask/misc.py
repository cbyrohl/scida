import io
import os
from typing import Optional

import dask.array as da
import numpy as np

from astrodask.config import get_config
from astrodask.helpers_misc import hash_path


def return_hdf5cachepath(path_original) -> str:
    fp = return_cachefile_path(os.path.join(hash_path(path_original), "data.hdf5"))
    return fp


def path_hdf5cachefile_exists(path) -> bool:
    """Checks whether a cache file exists for given path."""
    fp = return_hdf5cachepath(path)
    if os.path.isfile(fp):
        return True
    return False


def return_cachefile_path(fname: str) -> Optional[str]:
    """If path cannot be generated, return False"""
    config = get_config()
    if "cache_path" not in config:
        return None
    cp = config["cache_path"]
    cp = os.path.expanduser(cp)
    if not os.path.exists(cp):
        os.mkdir(cp)
    fp = os.path.join(cp, fname)
    fp = os.path.expanduser(fp)
    bp = os.path.dirname(fp)
    if not os.path.exists(bp):
        os.mkdir(bp)
    return fp


def sprint(*args, **kwargs):
    """print to string"""
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def map_interface_args(paths, *args, **kwargs):
    """Map arguments for interface if they are not lists"""
    n = len(paths)
    for i, path in enumerate(paths):
        targs = []
        for arg in args:
            if not (isinstance(arg, list)) or len(arg) != n:
                targs.append(arg)
            else:
                targs.append(arg[i])
        tkwargs = {}
        for k, v in kwargs.items():
            if not (isinstance(v, list)) or len(v) != n:
                tkwargs[k] = v
            else:
                tkwargs[k] = v[i]
        yield path, targs, tkwargs


def str_is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def rectangular_cutout_mask(
    center, width, coords, pbc=True, boxsize=None, backend="dask", chunksize="auto"
):
    center = np.array(center)
    width = np.array(width)
    if backend == "dask":
        be = da
    else:
        be = np

    dists = coords - center
    dists = be.fabs(dists)
    if pbc:
        if boxsize is None:
            raise ValueError("Need to specify for boxsize for PBC.")
        dists = be.where(dists > 0.5 * boxsize, be.fabs(boxsize - dists), dists)

    kwargs = {}
    if backend == "dask":
        kwargs["chunks"] = chunksize
    mask = be.ones(coords.shape[0], dtype=np.bool, **kwargs)
    for i in range(3):
        mask &= dists[:, i] < (
            width[i] / 2.0
        )  # TODO: This interval is not closed on the left side.
    return mask
