import logging
import os
import pathlib
from collections.abc import MutableMapping
from typing import Optional

import dask.array as da
import numpy as np

from scida.config import get_config, get_simulationconfig
from scida.fields import FieldContainer
from scida.helpers_misc import hash_path

log = logging.getLogger(__name__)


def get_container_from_path(
    element: str, container: FieldContainer = None, create_missing: bool = False
) -> FieldContainer:
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


def return_hdf5cachepath(path_original, fileprefix=None) -> str:
    path = path_original
    if fileprefix is not None:
        path = os.path.join(path, fileprefix)
    hsh = hash_path(path)
    fp = return_cachefile_path(os.path.join(hsh, "data.hdf5"))
    return fp


def path_hdf5cachefile_exists(path, **kwargs) -> bool:
    """Checks whether a cache file exists for given path."""
    fp = return_hdf5cachepath(path, **kwargs)
    if os.path.isfile(fp):
        return True
    return False


def return_cachefile_path(fname: str) -> Optional[str]:
    """If path cannot be generated, return None"""
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


def check_config_for_dataset(metadata, path: Optional[str] = None, unique=True):
    """Check whether the given dataset can be identified to be a certain simulation (type) by its metadata"""
    c = get_simulationconfig()

    candidates = []
    if "data" not in c:
        return candidates
    for k, vals in c["data"].items():
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
                if not any([substring in d for d in dirnames]):
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

    """
    cls = olddict.__class__
    for k, v in olddict.items():
        if isinstance(v, MutableMapping):
            newdict[k] = cls()
            deepdictkeycopy(v, newdict[k])
