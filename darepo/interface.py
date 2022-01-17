import os
import logging
import tempfile

import dask.array as da
import numpy as np
import h5py
import zarr

from .config import _config

from .helpers_hdf5 import create_virtualfile, walk_hdf5file
from .helpers_misc import hash_path, RecursiveNamespace, make_serializable
from .fields import FieldContainer
from .helpers_misc import partTypeNum


class Dataset(object):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.file = None
        self.tempfile = None # need this reference to keep garbage collection away for the tempfile
        self.location = None

        # Let's find the data and metadata for the object at 'path'
        self.metadata = {}
        self.data = {}

        if not os.path.exists(self.path):
            raise Exception("Specified path does not exist.")

        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path,".zgroup")):
                # object is a zarr object
                raise NotImplementedError # TODO
            else:
                # otherwise expect this is a chunked HDF5 file
                self.load_chunkedhdf5()
        else:
            # we are directly given a target file
            raise NotImplementedError # TODO


    def load_chunkedhdf5(self):
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
            self.tempfile = tempfile.NamedTemporaryFile("wb", suffix=".hdf5")
            self.location = self.tempfile.name
            create_virtualfile(self.location, files)
            logging.warning("No caching directory specified. Initial file read will remain slow.")

        self.file = h5py.File(self.location,"r")
        # Populate instance with hdf5 data
        tree = {}
        walk_hdf5file(self.location, tree)
        ## groups
        for group in tree["groups"]:
            # TODO: Do not hardcode dataset/field groups
            if group.startswith("/PartType") or group.startswith("/Group") or group.startswith("/Subhalo"):
                self.data[group.split("/")[1]] = {}
        ## datasets
        for dataset in tree["datasets"]:
            if dataset[0].startswith("/PartType") or dataset[0].startswith("/Group") or dataset[0].startswith("/Subhalo"):
                group = dataset[0].split("/")[1]  # TODO: Still dont support more nested groups
                ds = da.from_array(self.file[dataset[0]])
                #ds = load_hdf5dataset_as_daskarr((self.file, dataset[0],), shape=dataset[1], dtype=dataset[2])
                self.data[group][dataset[0].split("/")[-1]] = ds

        ## Make each datadict entry a FieldContainer
        for k in self.data:
            self.data[k] = FieldContainer(**self.data[k])

        ## attrs
        self.metadata = tree["attrs"]

    def save(self, fname, overwrite=True):
        """Saving into zarr format."""
        # We use zarr, as this way we have support to directly write into the file by the workers
        # (rather than passing back the data chunk over the scheduler to the interface)
        # Also, this way we can leverage new features, such as a large variety of compression methods.
        store = zarr.DirectoryStore(fname)
        root = zarr.group(store, overwrite=overwrite)

        # Metadata
        defaultattributes = ["config","header","parameters"]
        for dctname in defaultattributes:
            if dctname in self.__dict__:
                grp = root.create_group(dctname)
                dct = self.__dict__[dctname]
                for k, v in dct.items():
                    v = make_serializable(v)
                    grp.attrs[k] = v
        # Data
        datagrp = root.create_group("data")
        for p in self.data:
            datagrp.create_group(p)
            for k in self.data[p]:
                da.to_zarr(self.data[p][k], os.path.join(fname, "data", p, k), overwrite=True)


class BaseSnapshot(Dataset):
    def __init__(self, path):
        super().__init__(path)

        defaultattributes = ["config","header","parameters"]
        for k in self.metadata:
            name = k.strip("/").lower()
            if name in defaultattributes:
                self.__dict__[name] = self.metadata[k]

        self.d = RecursiveNamespace(**self.data)

        self.boxsize = np.full(3,np.nan)

    def register_field(self, parttype, name=None, construct=True):
        num = partTypeNum(parttype)
        return self.data["PartType"+str(num)].register_field(name=name)



class ArepoSnapshot(BaseSnapshot):
    def __init__(self, path):
        self.header = {}
        self.config = {}
        self.parameters = {}
        super().__init__(path)

        if isinstance(self.header["BoxSize"],float):
            self.boxsize[:] = self.header["BoxSize"]
        else:
            # Have not thought about non-cubic cases yet.
            raise NotImplementedError

