import io
import os
from typing import Optional

from astrodask.config import get_config


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
