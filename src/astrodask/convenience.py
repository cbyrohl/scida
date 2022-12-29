import os
from os.path import join
from typing import List

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
            if databackend in ["mpcdf", "virgotng"]:
                bp = "/virgotng/universe/IllustrisTNG"
                pathdict = {"TNG50-%i" % i: join(bp, "TNG50-%i" % i) for i in range(4)}
                path = pathdict[path.split(":")[1]]
    else:
        raise ValueError("Specified path unknown.")

    available_dtypes: List[str] = []
    for k, dtype in dataset_type_registry.items():
        if dtype.validate_path(path):
            available_dtypes.append(k)
            print("AVAIL:", k)
        else:
            print("UNAVAIL:", k)
    if len(available_dtypes) == 0:
        raise ValueError("Unknown data type.")
    elif strict and len(available_dtypes) > 1:
        raise ValueError("Ambiguous data type.")
    dtype = available_dtypes[0]
    # TODO: Check for most specific class to take if ambiguous.

    return dataset_type_registry[dtype](path, **kwargs)
