import hashlib
import inspect
import io
import re
import types

import dask.array as da
import numpy as np


def hash_path(path):
    sha = hashlib.sha256()
    sha.update(str(path).strip("/ ").encode())
    return sha.hexdigest()[:16]


class RecursiveNamespace(types.SimpleNamespace):
    # see https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    def __init__(self, **kwargs):
        """Create a SimpleNamespace recursively"""
        super().__init__(**kwargs)
        self.__dict__.update({k: self.__elt(v) for k, v in kwargs.items()})

    def __elt(self, elt):
        """Recurse into elt to create leaf namespace objects"""
        if type(elt) is dict:
            return type(self)(**elt)
        if type(elt) in (list, tuple):
            return [self.__elt(i) for i in elt]
        return elt


def make_serializable(v):
    # Attributes need to be JSON serializable. No numpy types allowed.
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return v


def get_kwargs(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_args(func):
    signature = inspect.signature(func)
    return [
        k
        for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty
    ]


def computedecorator(func):
    def wrapper(*args, compute=False, **kwargs):
        res = func(*args, **kwargs)
        if compute:
            return res.compute()
        else:
            return res

    return wrapper


# based on https://stackoverflow.com/a/60708339 & https://stackoverflow.com/a/42865957/2002471
units = {"B": 1, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}


def parse_humansize(size):
    size = size.upper()
    if not re.match(r" ", size):
        size = re.sub(r"([KMGT]?I*B)", r" \1", size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])


def sprint(*args, end="\n", **kwargs):
    """print to string"""
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
        res = res * output_units

    return res
