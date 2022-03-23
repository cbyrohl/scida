import os
import logging
import tempfile
import hashlib

import dask.array as da
import dask
import numpy as np
import h5py
import zarr

from .config import _config

from .helpers_hdf5 import create_mergedhdf5file, walk_hdf5file, walk_zarrfile
from .helpers_misc import hash_path, make_serializable
from .fields import FieldContainerCollection
from collections.abc import MutableMapping


class Dataset(object):
    def __init__(self, path, chunksize="auto", virtualcache=True, overwritecache = False):
        super().__init__()
        self.path = path
        self.file = None
        self.tempfile = None  # need this reference to keep garbage collection away for the tempfile
        self.location = None
        self.chunksize = chunksize
        self.virtualcache = virtualcache
        self.overwritecache = overwritecache

        # Let's find the data and metadata for the object at 'path'
        self.metadata = {}
        self.data = FieldContainerCollection()

        if not os.path.exists(self.path):
            raise Exception("Specified path does not exist.")

        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path,".zgroup")):
                # object is a zarr object
                self.load_zarr()
            else:
                # otherwise expect this is a chunked HDF5 file
                self.load_chunkedhdf5(overwrite=self.overwritecache)
        else:
            # we are directly given a target file
            self.load_hdf5()

    def __del__(self):
        """
        On deletion of object, make sure we close file.
        """
        try:
            self.file.close()
        except AttributeError:
            pass  # arises when __init__ fails.

    def __hash__(self):
        """Hash for Dataset instance to be derived from the file location."""
        # determinstic hash; note that hash() on a string is no longer deterministic in python3.
        hash_value = int(hashlib.sha256(self.location.encode('utf-8')).hexdigest(), 16) % 10**10
        return hash_value

        
    def __dask_tokenize__(self):
        return self.__hash__()


    def load_hdf5(self):
        self.location = self.path
        tree = {}
        walk_hdf5file(self.location,tree=tree)
        try:
            import h5pickle
            self.file = h5pickle.File(self.location,"r")
        except ImportError:
            self.file = h5py.File(self.location,"r")
        self.load_from_tree(tree)

    def load_zarr(self):
        self.location = self.path
        tree = {}
        walk_zarrfile(self.location,tree=tree)
        self.file = zarr.open(self.location)
        self.load_from_tree(tree)

    def load_chunkedhdf5(self,overwrite=False):
        files = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        files = np.array([f for f in files if os.path.isfile(f)]) # ignore subdirectories
        nmbrs = [int(f.split(".")[-2]) for f in files]
        sortidx = np.argsort(nmbrs)
        files = files[sortidx]

        if "cachedir" in _config:
            # we create a virtual file in the cache directory
            dataset_cachedir = os.path.join(_config["cachedir"], hash_path(self.path))
            fp = os.path.join(dataset_cachedir,"data.hdf5")
            if not os.path.exists(dataset_cachedir):
                os.mkdir(dataset_cachedir)
            if not os.path.isfile(fp) or overwrite:
                create_mergedhdf5file(fp, files, virtual=self.virtualcache)
            self.location = fp
        else:
            # otherwise, we create a temporary file
            self.tempfile = tempfile.NamedTemporaryFile("wb", suffix=".hdf5")
            self.location = self.tempfile.name
            create_mergedhdf5file(self.location, files, virtual=self.virtualcache)
            logging.warning("No caching directory specified. Initial file read will remain slow.")


        try:
            import h5pickle
            self.file = h5pickle.File(self.location,"r")
        except ImportError:
            self.file = h5py.File(self.location,"r")

        tree = {}
        walk_hdf5file(self.location, tree)
        self.load_from_tree(tree)


    def load_from_tree(self,tree,groups_load=None):
        """groups_load: list of groups to load; all groups with datasets are loaded if groups_load==None"""
        inline_array = False  # inline arrays in dask; heavily reduces memory usage. However, doesnt work with h5py (need h5pickle wrapper or zarr).
        if type(self.file) == h5py._hl.files.File:
            logging.warning("Install h5pickle ('pip install h5pickle') to allow inlining arrays. This will increase performance significantly.")
            inline_array = False

        if groups_load is None:
            toload = True
            groups_with_datasets = set([d[0].split("/")[1] for d in tree["datasets"]])
        ## groups
        for group in tree["groups"]:
            # TODO: Do not hardcode dataset/field groups
            if groups_load is not None:
                toload = any([group.startswith(gname) for gname in groups_load])
            else:
                toload = any([group=="/"+g for g in groups_with_datasets])
            if toload:
                self.data[group.split("/")[1]] = {}
        ## datasets
        for dataset in tree["datasets"]:
            if groups_load is not None:
                toload = any([dataset[0].startswith(gname) for gname in groups_load])
            else:
                toload = any([dataset[0].startswith("/"+g+"/") for g in groups_with_datasets])
            if toload:
                group = dataset[0].split("/")[1]  # TODO: Still dont support more nested groups
                name = "Dataset"+str(self.__hash__())+dataset[0].replace("/","_")
                if "__dask_tokenize__" in self.file: # check if the file contains the dask name.
                    name = self.file["__dask_tokenize__"].attrs.get(dataset[0].strip("/"), name)
                ds = da.from_array(self.file[dataset[0]], chunks=self.chunksize, name=name, inline_array=inline_array)
                self.data[group][dataset[0].split("/")[-1]] = ds

        ## Make each datadict entry a FieldContainer
        data = FieldContainerCollection(self.data.keys(), derivedfields_kwargs=dict(snap=self))
        for k in self.data:
            data.new_container(k, **self.data[k])
        self.data = data

        # Add a unique identifier for each element for each data type
        for p in self.data:
            for k, v in self.data[p].items():
                nparts = v.shape[0]
                if len(v.chunks)==1:
                    # make same shape as other 1D arrays.
                    chunks = v.chunks
                    break
                else:
                    # alternatively, attempt to use chunks from multi-dim array until found something better.
                    chunks = (v.chunks[0],)

            self.data[p]["uid"] = da.arange(nparts, chunks=chunks)

        ## attrs
        self.metadata = tree["attrs"]



    def return_data(self):
        return self.data

    def save(self, fname, overwrite=True, zarr_kwargs=None, cast_uints=False):
        """Saving into zarr format."""
        # We use zarr, as this way we have support to directly write into the file by the workers
        # (rather than passing back the data chunk over the scheduler to the interface)
        # Also, this way we can leverage new features, such as a large variety of compression methods.
        # cast_uints: if true, we cast uints to ints; needed for some compressions (particularly zfp)
        if zarr_kwargs is None:
            zarr_kwargs = {}
        store = zarr.DirectoryStore(fname,**zarr_kwargs)
        root = zarr.group(store, overwrite=overwrite)

        # Metadata
        defaultattributes = ["config", "header", "parameters"]
        for dctname in defaultattributes:
            if dctname in self.__dict__:
                grp = root.create_group(dctname)
                dct = self.__dict__[dctname]
                for k, v in dct.items():
                    v = make_serializable(v)
                    grp.attrs[k] = v
        # Data
        tasks = []
        for p in self.data:
            root.create_group(p)
            for k in self.data[p]:
                arr = self.data[p][k]
                if cast_uints:
                    if arr.dtype==np.uint64:
                        arr = arr.astype(np.int64)
                    elif arr.dtype==np.uint32:
                        arr = arr.astype(np.int32)
                task = da.to_zarr(arr, os.path.join(fname, p, k), overwrite=True, compute=False)
                tasks.append(task)
        dask.compute(tasks)


class BaseSnapshot(Dataset):
    def __init__(self, path, chunksize="auto", virtualcache=True):
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache)

        defaultattributes = ["config","header","parameters"]
        for k in self.metadata:
            name = k.strip("/").lower()
            if name in defaultattributes:
                self.__dict__[name] = self.metadata[k]

        self.boxsize = np.full(3,np.nan)


    def register_field(self, parttype, name=None, description=""):
        return self.data.register_field(parttype, name=name, description=description)


def deepdictkeycopy(olddict,newdict):
    """Recursively walk nested dictionary, only creating empty dictionaries
    for entries that are dictionaries themselves"""
    for k,v in olddict.items():
        if isinstance(v,MutableMapping):
            newdict[k] = {}
            deepdictkeycopy(v,newdict[k])


class Selector(object):
    """ Base Class for data selection decorator factory"""
    def __init__(self):
        self.keys = None  # the keys we check for.
        self.data_backup = {}  # holds a copy of the species' fields
        self.data = {}  # holds the species' fields we operate on

    def __call__(self, fn, *args, **kwargs):
        def newfn(*args, **kwargs):
            # TODO: Add graceful exit/restore after exception in self.prepare
            self.data_backup = args[0].data
            self.data = {}
            deepdictkeycopy(self.data_backup,self.data)

            self.prepare(*args, **kwargs)
            if self.keys is None:
                raise NotImplementedError("Subclass implementation needed for self.keys!")
            if kwargs.pop("dropkeys", True):
                for k in self.keys:
                    kwargs.pop(k, None)
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                self.finalize(*args, **kwargs)
        return newfn

    def prepare(self, *args, **kwargs):
        raise NotImplementedError("Subclass implementation needed!")

    def finalize(self, *args, **kwargs):
        args[0].data = self.data_backup

