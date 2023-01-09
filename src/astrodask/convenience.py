import copy
import os
from collections import Counter
from functools import reduce
from inspect import getmro
from os.path import join
from typing import List

from astrodask.config import get_config
from astrodask.registries import dataset_type_registry


def load(path: str, strict=False, **kwargs):
    if os.path.exists(path):
        # datasets on disk
        pass
    elif len(path.split(":")) > 1:
        # check different alternative backends
        databackend = path.split(":")[0]
        if databackend in ["http", "https"]:
            # dataset on the internet
            raise NotImplementedError("TODO.")
        else:
            # potentially custom dataset.
            # TODO: The following hardcoded cases should be loaded through a config file
            # config = get_config()
            if databackend in ["mpcdf", "virgotng"]:
                bp = "/virgotng/universe/IllustrisTNG"
                pathdict = {"TNG50-%i" % i: join(bp, "TNG50-%i" % i) for i in range(4)}
                path = pathdict[path.split(":")[1]]
    else:
        raise ValueError("Specified path unknown.")

    available_dtypes: List[str] = []
    reg = dataset_type_registry
    for k, dtype in reg.items():
        if dtype.validate_path(path):
            available_dtypes.append(k)
    if len(available_dtypes) == 0:
        raise ValueError("Unknown data type.")
    if len(available_dtypes) > 1:
        # reduce candidates by looking at most specific ones.
        inheritancecounters = [Counter(getmro(reg[k])) for k in available_dtypes]
        # try to find candidate that shows up only once across all inheritance trees.
        # => this will be the most specific candidate(s).
        count = reduce(lambda x, y: x + y, inheritancecounters)
        available_dtypes = [k for k in available_dtypes if count[reg[k]] == 1]
        if len(available_dtypes) > 1:
            # after looking for the most specific candidate(s), do we still have multiple?
            if strict:
                raise ValueError("Ambiguous data type.")
            else:
                print("Note: Multiple dataset candidates: ", available_dtypes)
    dtype = available_dtypes[0]

    return dataset_type_registry[dtype](path, **kwargs)


def get_dataset_by_name(name):
    dname = None
    c = get_config()
    if "datasets" not in c:
        return dname
    datasets = copy.deepcopy(c["datasets"])
    if name in datasets:
        dname = name
    else:
        # could still be an alias
        for k, v in datasets.items():
            if "aliases" not in v:
                continue
            if name in v["aliases"]:
                dname = k
                break
    return dname


def get_datasets_by_props(**kwargs):
    dnames = []
    c = get_config()
    datasets = copy.deepcopy(c["datasets"])
    if "datasets" not in c:
        return dnames
    for k, v in datasets.items():
        props = v.get("properties", {})
        match = True
        for pk, pv in kwargs.items():
            if pk not in props:
                match = False
                break
            if props[pk] != pv:
                match = False
                break
        if match:
            dnames.append(k)
    return dnames


def get_dataset_candidates(name=None, props=None):
    if name is not None:
        dnames = [get_dataset_by_name(name)]
        return dnames
    if props is not None:
        dnames = get_datasets_by_props(**props)
        return dnames
    raise ValueError("Need to specify name or properties.")


def get_dataset(name=None, props=None):
    dnames = get_dataset_candidates(name=name, props=props)
    if len(dnames) > 1:
        raise ValueError("Too many dataset candidates.")
    elif len(dnames) == 0:
        raise ValueError("No dataset candidate found.")
    return dnames[0]
