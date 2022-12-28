import os
from os.path import join

from astrodask.interface import BaseSnapshot
from astrodask.registries import dataset_type_registry


def load(path: str, **kwargs):
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

    print(dataset_type_registry)
    return BaseSnapshot(path, **kwargs)
