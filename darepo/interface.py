import os
import logging
import tempfile

import dask.array as da
import numpy as np
import h5py

from .config import _config

from .helpers_hdf5 import create_virtualfile, walk_hdf5file


class BaseSnapshot(object):
    def __init__(self, path):
        self.data = None
        self.path = path
        self.file = None # File reference
        self.fn = "" # Filename
        self.datadict = {}

        self.load_data()
        self.data = RecursiveNamespace(**self.datadict)

    def load_data(self):
        if os.path.isdir(self.path):
            # we are given a directory with multiple files. Create a virtual file first.
            files = np.array([os.path.join(self.path, f) for f in os.listdir(self.path)])
            nmbrs = [int(f.split(".")[-2]) for f in files]
            sortidx = np.argsort(nmbrs)
            files = files[sortidx]

            if "cachedir" in _config:
                # we create a virtual file in the cache directory
                fn = hash_path(self.path) + ".hdf5"
                file = os.path.join(_config["cachedir"], fn)
                if not os.path.isfile(file):
                    create_virtualfile(file, files)
                else:
                    # TODO
                    logging.warning("Virtual file already existing? TODO.")
                    raise NotImplementedError
            else:
                # otherwise, we create a temporary file
                self.file = tempfile.NamedTemporaryFile("wb", suffix=".hdf5")
                self.fn = self.file.name
                create_virtualfile(self.file.name, files)
                logging.warning("No caching directory specified. Initial file read will remain slow.")
        else:
            # we are directly given a target file
            # TODO
            raise NotImplementedError
        self.h5file = h5py.File(self.fn,"r")

        # Populate instance with hdf5 data
        tree = {}
        walk_hdf5file(self.fn, tree)
        ## groups
        for group in tree["groups"]:
            # TODO: Do not hardcode dataset/field groups
            if group.startswith("/PartType") or group.startswith("/Group") or group.startswith("/Subhalo"):
                self.datadict[group.split("/")[1]] = {}
        ## datasets
        for dataset in tree["datasets"]:
            if dataset[0].startswith("/PartType") or dataset[0].startswith("/Group") or dataset[0].startswith("/Subhalo"):
                group = dataset[0].split("/")[1]  # TODO: Still dont support more nested groups
                ds = da.from_array(self.h5file[dataset[0]])
                #ds = load_hdf5dataset_as_daskarr((self.file, dataset[0],), shape=dataset[1], dtype=dataset[2])
                self.datadict[group][dataset[0].split("/")[-1]] = ds

        ## attrs
        if "/Config" in tree["attrs"]:
            self.config = tree["attrs"]["/Config"]
        if "/Header" in tree["attrs"]:
            self.header = tree["attrs"]["/Header"]
        if "/Parameters" in tree["attrs"]:
            self.parameters = tree["attrs"]["/Parameters"]


class ArepoSnapshot(BaseSnapshot):
    def __init__(self, path):
        super().__init__(path)
