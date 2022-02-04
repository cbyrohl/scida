import os
import logging
import tempfile

import dask.array as da
import dask
import numpy as np
import h5py
import zarr

from .config import _config

from .helpers_hdf5 import create_mergedhdf5file, walk_hdf5file, walk_zarrfile
from .helpers_misc import hash_path, RecursiveNamespace, make_serializable
from .fields import FieldContainerCollection


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

    def load_hdf5(self):
        self.location = self.path
        tree = {}
        walk_hdf5file(self.location,tree=tree)
        self.file = h5py.File(self.location,"r")
        self.load_from_tree(tree)

    def load_zarr(self):
        self.location = self.path
        tree = {}
        walk_zarrfile(self.location,tree=tree)
        self.file = zarr.open(self.location)
        self.load_from_tree(tree)

    def load_chunkedhdf5(self,overwrite=False):
        files = np.array([os.path.join(self.path, f) for f in os.listdir(self.path)])
        nmbrs = [int(f.split(".")[-2]) for f in files]
        sortidx = np.argsort(nmbrs)
        files = files[sortidx]

        if "cachedir" in _config:
            # we create a virtual file in the cache directory
            fn = hash_path(self.path) + ".hdf5"
            fp = os.path.join(_config["cachedir"], fn)
            if not os.path.isfile(fp) or overwrite:
                create_mergedhdf5file(fp, files, virtual=self.virtualcache)
            self.location = fp
        else:
            # otherwise, we create a temporary file
            self.tempfile = tempfile.NamedTemporaryFile("wb", suffix=".hdf5")
            self.location = self.tempfile.name
            create_mergedhdf5file(self.location, files, virtual=self.virtualcache)
            logging.warning("No caching directory specified. Initial file read will remain slow.")

        self.file = h5py.File(self.location,"r")

        tree = {}
        walk_hdf5file(self.location, tree)
        self.load_from_tree(tree)


    def load_from_tree(self,tree,groups_load=None):
        """groups_load: list of groups to load; all groups with datasets are loaded if groups_load==None"""
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
                ds = da.from_array(self.file[dataset[0]], chunks=self.chunksize)
                self.data[group][dataset[0].split("/")[-1]] = ds

        ## Make each datadict entry a FieldContainer
        data = FieldContainerCollection(self.data.keys(), derivedfields_kwargs=dict(snap=self))
        for k in self.data:
            data.new_container(k, **self.data[k])
        self.data = data

        # Add a unique identifier for each element for each data type
        for p in self.data:
            k = next(iter(self.data[p]))
            arr = self.data[p][k]
            nparts = arr.shape[0]
            chunks = arr.chunks
            if len(chunks) == 2:
                chunks = (chunks[0],)
            self.data[p]["uid"] = da.arange(nparts, chunks=chunks)

        ## attrs
        self.metadata = tree["attrs"]



    def return_data(self):
        return self.data

    def save(self, fname, overwrite=True, zarr_kwargs={}, cast_uints=False):
        """Saving into zarr format."""
        # We use zarr, as this way we have support to directly write into the file by the workers
        # (rather than passing back the data chunk over the scheduler to the interface)
        # Also, this way we can leverage new features, such as a large variety of compression methods.
        # cast_uints: if true, we cast uints to ints; needed for some compressions (particularly zfp)
        store = zarr.DirectoryStore(fname,**zarr_kwargs)
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

                task = da.to_zarr(arr, os.path.join(fname, p, k), overwrite=True,compute=False)
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

        self.d = RecursiveNamespace(**self.data)

        self.boxsize = np.full(3,np.nan)


    def register_field(self, parttype, name=None, description=""):
        return self.data.register_field(parttype, name=name, description=description)


def deepdictkeycopy(olddict,newdict):
    """Recursively walk nested dictionary, only creating empty dictionaries
    for entries that are dictionaries themselves"""
    for k,v in olddict.items():
        if isinstance(v,dict):
            newdict[k] = {}
            deepdictkeycopy(v,newdict[k])

class Selector(object):
    """ Base Class for data selection decorator factory"""
    def __init__(self):
        self.keys = None
        self.data_backup = {}
        self.data = {}

    def __call__(self, fn, *args, **kwargs):
        def newfn(*args, **kwargs):
            # TODO: Add graceful exit/restore after exception in self.prepare
            self.data_backup = args[0].data
            self.data = {}
            deepdictkeycopy(self.data_backup,self.data)

            self.prepare(*args,**kwargs)
            if self.keys is None:
                raise NotImplementedError("Subclass implementation needed for self.keys!")
            if(kwargs.pop("dropkeys",True)):
                for k in self.keys:
                    kwargs.pop(k, None)
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                self.finalize(*args,**kwargs)
        return newfn

    def prepare(self):
        raise NotImplementedError("Subclass implementation needed!")

    def finalize(self,*args,**kwargs):
        args[0].data = self.data_backup

