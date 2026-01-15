"""
Miscellaneous helper functions.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
from collections import defaultdict
from collections.abc import MutableMapping

import dask.array as da
import numpy as np

from scida.config import get_config, get_simulationconfig
from scida.fields import FieldContainer
from scida.helpers_misc import hash_path

log = logging.getLogger(__name__)


def get_container_from_path(
    element: str, container: FieldContainer = None, create_missing: bool = False
) -> FieldContainer:
    """
    Get a container from a path.
    Parameters
    ----------
    element: str
    container: FieldContainer
    create_missing: bool

    Returns
    -------
    FieldContainer:
        container specified by path

    """
    keys = element.split("/")
    rv = container
    for key in keys:
        if key == "":
            continue
        if key not in rv._containers:
            if not create_missing:
                raise ValueError("Container '%s' not found in '%s'" % (key, rv))
            rv.add_container(key, name=key)
        rv = rv._containers[key]
    return rv


def return_hdf5cachepath(path, fileprefix: str | None = None) -> str:
    """
    Returns the path to the cache file for a given path.

    Parameters
    ----------
    path: str
        path to the dataset
    fileprefix: str | None
        Can be used to specify the fileprefix used for the dataset.

    Returns
    -------
    str

    """
    if fileprefix is not None:
        path = os.path.join(path, fileprefix)
    hsh = hash_path(path)
    fp = return_cachefile_path(os.path.join(hsh, "data.hdf5"))
    return fp


def path_hdf5cachefile_exists(path, **kwargs) -> bool:
    """
    Checks whether a cache file exists for given path.
    Parameters
    ----------
    path:
        path to the dataset
    kwargs:
        passed to return_hdf5cachepath
    Returns
    -------
    bool

    """
    fp = return_hdf5cachepath(path, **kwargs)
    if os.path.isfile(fp):
        return True
    return False


def return_cachefile_path(fname: str) -> str | None:
    """
    Return the path to the cache file, return None if path cannot be generated.

    Parameters
    ----------
    fname: str
        filename of cache file

    Returns
    -------
    str or None

    """
    config = get_config()
    if "cache_path" not in config:
        return None
    cp = config["cache_path"]
    cp = os.path.expanduser(cp)
    path = pathlib.Path(cp)
    path.mkdir(parents=True, exist_ok=True)
    fp = os.path.join(cp, fname)
    fp = os.path.expanduser(fp)
    bp = os.path.dirname(fp)
    if not os.path.exists(bp):
        try:
            os.mkdir(bp)
        except FileExistsError:
            pass  # can happen due to parallel access
    return fp


def map_interface_args(paths: list, *args, **kwargs):
    """
    Map arguments for interface if they are not lists.
    Parameters
    ----------
    paths
    args
    kwargs

    Returns
    -------
    generator
        yields path, args, kwargs

    """
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
    """
    Check whether a string can be converted to a float.
    Parameters
    ----------
    element: str
        string to check

    Returns
    -------
    bool

    """
    try:
        float(element)
        return True
    except ValueError:
        return False


def rectangular_cutout_mask(
    center, width, coords, pbc=True, boxsize=None, backend="dask", chunksize="auto"
):
    """
    Create a rectangular mask for a given set of coordinates.
    Parameters
    ----------
    center: list
        center of the rectangle
    width: list
        widths of the rectangle
    coords: ndarray
        coordinates to mask
    pbc: bool
        whether to apply PBC
    boxsize:
        boxsize for PBC
    backend: str
        backend to use (dask or numpy)
    chunksize: str
        chunksize for dask

    Returns
    -------
    ndarray:
        mask

    """
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
    mask = be.ones(coords.shape[0], dtype=bool, **kwargs)
    for i in range(3):
        mask &= dists[:, i] < (
            width[i] / 2.0
        )  # TODO: This interval is not closed on the left side.
    return mask


def check_config_for_dataset(metadata, path: str | None = None, unique: bool = True):
    """
    Check whether the given dataset can be identified to be a certain simulation (type) by its metadata.

    Parameters
    ----------
    metadata: dict
        metadata of the dataset used for identification
    path: str
        path to the dataset, sometimes helpful for identification
    unique: bool
        whether to expect return to be unique

    Returns
    -------
    List[str]
        Strings of Candidate identifiers

    """
    c = get_simulationconfig()

    candidates = []
    if "data" not in c:
        return candidates
    simdct = c["data"]
    if simdct is None:
        simdct = {}
    for k, vals in simdct.items():
        if vals is None:
            continue
        possible_candidate = True
        if "identifiers" in vals:
            idtfrs = vals["identifiers"]
            # special key not specifying identifying metadata
            specialkeys = ["name_contains"]
            allkeys = idtfrs.keys()
            keys = list([k for k in allkeys if k not in specialkeys])
            if "name_contains" in idtfrs and path is not None:
                p = pathlib.Path(path)
                # we only check the last three path elements
                dirnames = [p.name, p.parents[0].name, p.parents[1].name]
                substring = idtfrs["name_contains"]
                if not any([substring.lower() in d.lower() for d in dirnames]):
                    possible_candidate = False
            if len(allkeys) == 0:
                possible_candidate = False
            for grp in keys:
                v = idtfrs[grp]
                h5path = "/" + grp
                if h5path not in metadata:
                    possible_candidate = False
                    break
                attrs = metadata[h5path]
                for ikey, ival in v.items():
                    if ikey not in attrs:
                        possible_candidate = False
                        break
                    av = attrs[ikey]
                    matchtype = None
                    if isinstance(ival, dict):
                        matchtype = ival.get("match", matchtype)  # default means equal
                        ival = ival["content"]

                    if isinstance(av, bytes):
                        av = av.decode("UTF-8")
                    if matchtype is None:
                        if is_scalar(av):
                            if not np.isclose(av, ival):
                                possible_candidate = False
                                break
                        elif isinstance(av, (list, np.ndarray)):
                            if not np.all(av == ival):
                                possible_candidate = False
                                break
                        else:
                            if av != ival:
                                possible_candidate = False
                                break
                    elif matchtype == "substring":
                        if ival not in av:
                            possible_candidate = False
                            break
        else:
            possible_candidate = False
        if possible_candidate:
            candidates.append(k)
    if unique and len(candidates) > 1:
        raise ValueError("Multiple dataset candidates (set unique=False?):", candidates)
    return candidates


def deepdictkeycopy(olddict: object, newdict: object) -> None:
    """
    Recursively walk nested dictionary, only creating empty dictionaries for entries that are dictionaries themselves.
    Parameters
    ----------
    olddict
    newdict

    Returns
    -------
    None
    """
    cls = olddict.__class__
    for k, v in olddict.items():
        if isinstance(v, MutableMapping):
            newdict[k] = cls()
            deepdictkeycopy(v, newdict[k])


_sizeunits = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
}

_sizeunits = {k.lower(): v for k, v in _sizeunits.items()}


def parse_size(size):
    """
    Parse a size string to a number in bytes.
    Parameters
    ----------
    size: str

    Returns
    -------
    int
        size in bytes
    """
    idx = 0
    for c in size:
        if c.isnumeric():
            continue
        idx += 1
    number = size[:idx]
    unit = size[idx:]
    return int(float(number) * _sizeunits[unit.lower().strip()])


def is_scalar(value):
    is_sclr = isinstance(value, (int, float, np.integer, np.floating))
    # also allow 0d arrays
    is_sclr |= isinstance(value, np.ndarray) and value.ndim == 0
    return is_sclr


def get_scalar(value):
    scalar = None
    if isinstance(value, np.ndarray) and value.ndim > 0:
        scalar = value[0]
    else:
        scalar = float(value)
    return scalar


def group_by_common_prefix(file_names):
    groups = defaultdict(list)
    for file in file_names:
        # Split by both "_" and "."
        parts = re.split(r"[_\.]", file)
        prefix = parts[0]  # Use the first part as the prefix
        groups[prefix].append(file)

    return dict(groups)
