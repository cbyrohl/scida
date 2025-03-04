import hashlib
import inspect
import io
import logging
import re
import types

import dask.array as da
import numpy as np

log = logging.getLogger(__name__)


def hash_path(path):
    """
    Hash a path to a fixed length string.

    Parameters
    ----------
    path: str or pathlib.Path

    Returns
    -------
    str
    """
    sha = hashlib.sha256()
    sha.update(str(path).strip("/ ").encode())
    return sha.hexdigest()[:16]


# see https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
class RecursiveNamespace(types.SimpleNamespace):
    """A SimpleNamespace that can be created recursively from a dict"""

    def __init__(self, **kwargs):
        """
        Create a SimpleNamespace recursively
        """
        super().__init__(**kwargs)
        self.__dict__.update({k: self.__elt(v) for k, v in kwargs.items()})

    def __elt(self, elt):
        """
        Recurse into elt to create leaf namespace objects.
        """
        if isinstance(elt, dict):
            return type(self)(**elt)
        if type(elt) in (list, tuple):
            return [self.__elt(i) for i in elt]
        return elt


def make_serializable(v):
    """
    Make a value JSON serializable.

    Parameters
    ----------
    v: any
        Object to make JSON serializable

    Returns
    -------
    any
        JSON serializable object
    """

    # Attributes need to be JSON serializable. No numpy types allowed.
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return v


def get_kwargs(func):
    """
    Get the keyword arguments of a function.

    Parameters
    ----------
    func: callable
        Function to get keyword arguments from

    Returns
    -------
    dict
        Keyword arguments of the function
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_args(func):
    """
    Get the positional arguments of a function.

    Parameters
    ----------
    func: callable

    Returns
    -------
    list
        Positional arguments of the function
    """
    signature = inspect.signature(func)
    return [k for k, v in signature.parameters.items() if v.default is inspect.Parameter.empty]


def computedecorator(func):
    """
    Decorator introducing compute keyword to evalute dask array returns.

    Parameters
    ----------
    func: callable
        Function to decorate

    Returns
    -------
    callable
        Decorated function
    """

    def wrapper(*args, compute=False, **kwargs):
        res = func(*args, **kwargs)
        if compute:
            return res.compute()
        else:
            return res

    return wrapper


# based on https://stackoverflow.com/a/60708339 & https://stackoverflow.com/a/42865957/2002471
units = {"B": 1, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}


# needed for processing dask arrays
def parse_humansize(size):
    """
    Parse a human-readable size string to bytes.

    Parameters
    ----------
    size: str
        Human readable size string, e.g. 1.5GiB

    Returns
    -------
    int
        Size in bytes
    """
    size = size.upper()
    if not re.match(r" ", size):
        size = re.sub(r"([KMGT]?I*B)", r" \1", size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])


def sprint(*args, end="\n", **kwargs):
    """
    Print to a string.

    Parameters
    ----------
    args: any
        Arguments to print
    end: str
        String to append at the end
    kwargs: any
        Keyword arguments to pass to print

    Returns
    -------
    str
       String to print
    """
    output = io.StringIO()
    print(*args, file=output, end=end, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def map_blocks(
    func,
    *args,
    name=None,
    token=None,
    dtype=None,
    chunks=None,
    drop_axis=None,
    new_axis=None,
    enforce_ndim=False,
    meta=None,
    output_units=None,
    **kwargs,
):
    """map_blocks with units"""
    da_kwargs = dict(
        name=name,
        token=token,
        dtype=dtype,
        chunks=chunks,
        drop_axis=drop_axis,
        new_axis=new_axis,
        enforce_ndim=enforce_ndim,
        meta=meta,
    )
    res = da.map_blocks(
        func,
        *args,
        **da_kwargs,
        **kwargs,
    )
    if output_units is not None:
        if hasattr(res, "magnitude"):
            log.info("map_blocks output already has units, overwriting.")
            res = res.magnitude * output_units
        res = res * output_units

    return res
