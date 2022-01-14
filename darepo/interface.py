import os
import logging
import tempfile

import dask.array as da
import numpy as np
import h5py
import zarr

from .config import _config

from .helpers_hdf5 import create_virtualfile, walk_hdf5file
from .helpers_misc import hash_path, RecursiveNamespace
from .fields import FieldContainer


class BaseSnapshot(object):
    def __init__(self, path):
        self.data = None
        self.path = path
        self.file = None # File reference
        self.fn = "" # Filename
        self.data = {}

        self.load_data()
        self.d = RecursiveNamespace(**self.data)

        self.boxsize = np.full(3,np.nan)

    def load_data(self):
        if not os.path.exists(self.path):
            raise Exception("Specified path does not exist.")

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
                self.data[group.split("/")[1]] = {}
        ## datasets
        for dataset in tree["datasets"]:
            if dataset[0].startswith("/PartType") or dataset[0].startswith("/Group") or dataset[0].startswith("/Subhalo"):
                group = dataset[0].split("/")[1]  # TODO: Still dont support more nested groups
                ds = da.from_array(self.h5file[dataset[0]])
                #ds = load_hdf5dataset_as_daskarr((self.file, dataset[0],), shape=dataset[1], dtype=dataset[2])
                self.data[group][dataset[0].split("/")[-1]] = ds

        ## Make each datadict entry a FieldContainer
        for k in self.data:
            self.data[k] = FieldContainer(**self.data[k])

        ## attrs
        if "/Config" in tree["attrs"]:
            self.config = tree["attrs"]["/Config"]
        if "/Header" in tree["attrs"]:
            self.header = tree["attrs"]["/Header"]
        if "/Parameters" in tree["attrs"]:
            self.parameters = tree["attrs"]["/Parameters"]
            
            
    def save(self, fname, overwrite=True):
        """Saving into zarr format."""
        # We use zarr, as this way we have support to directly write into the file by the workers
        # (rather than passing back the data chunk over the scheduler to the interface)
        # Also, this way we can leverage new features, such as a large variety of compression methods.
        store = zarr.DirectoryStore(fname)
        root = zarr.group(store, overwrite=overwrite)

        # Metadata
        attrsdict = dict(Config=self.config,Header=self.header,Parameters=self.parameters)
        for dctname, dct in attrsdict.items():
            if isinstance(dct, dict):
                grp = root.create_group(dctname)
                for k,v in dct.items():
                    v = make_serializable(v)
                    grp.attrs[k] = v
        # Data
        datagrp = root.create_group("data")
        for p in self.data:
            grp = datagrp.create_group(p)
            for k in self.data[p]:
                da.to_zarr(self.data[p][k],os.path.join(fname,"data",p),overwrite=True)

    def save_header(self,fname):
        with h5py.File(fname, 'r+') as hf:
            grp = hf.create_group("Header")
            for h in self.header:
                grp.attrs[h] = self.header[h]
            # overwrite relevant header attributes.
            grp.attrs["NumFilesPerSnapshot"] = 1
            h = (self.cosmology.H0/100).value
            grp.attrs["BoxSize"] = (self.boxsize/h*u.kpc).to(u.cm).value/(1+grp.attrs["Redshift"])


class ArepoSnapshot(BaseSnapshot):
    def __init__(self, path):
        super().__init__(path)

        if isinstance(self.header["BoxSize"],float):
            self.boxsize[:] = self.header["BoxSize"]
        else:
            # Have not thought about non-cubic cases yet.
            raise NotImplementedError

def make_serializable(v):
    # Attributes need to be JSON serializable. No numpy types allowed.
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return v

