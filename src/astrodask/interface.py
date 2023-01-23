import abc
import hashlib
import os
from collections.abc import MutableMapping

import dask
import dask.array as da
import numpy as np
import zarr

from astrodask.fields import FieldContainer, FieldContainerCollection
from astrodask.helpers_misc import make_serializable
from astrodask.io import BaseLoader
from astrodask.misc import sprint
from astrodask.registries import dataset_type_registry


class Dataset(abc.ABC):
    @classmethod
    @property  # TODO: chaining property and classmethod deprecated in py 3.11: remove!
    @abc.abstractmethod
    def _loader(cls):  # each class has to have a loader
        raise NotImplementedError

    def __init__(
        self,
        path,
        chunksize="auto",
        virtualcache=True,
        overwritecache=False,
        fileprefix="",
        **kwargs
    ):
        super().__init__()
        self.path = path
        self.file = None
        self.tempfile = (
            None  # need this reference to keep garbage collection away for the tempfile
        )
        self.location = None
        self.chunksize = chunksize
        self.virtualcache = virtualcache
        self.overwritecache = overwritecache

        # Let's find the data and metadata for the object at 'path'
        self.metadata = {}
        self.data = FieldContainerCollection()

        if not os.path.exists(self.path):
            raise Exception("Specified path '%s' does not exist." % self.path)

        loadkwargs = dict(
            overwrite=self.overwritecache,
            fileprefix=fileprefix,
            virtualcache=virtualcache,
            derivedfields_kwargs=dict(snap=self),
        )

        loader = self._loader(path)
        self.data, self.metadata = loader.load(**loadkwargs)
        self.tempfile = loader.tempfile
        self.file = loader.file

    def info(self, listfields=False):
        rep = ""
        rep += sprint(self.__class__.__name__)
        props = self._repr_dict()
        for k, v in props.items():
            rep += sprint("%s: %s" % (k, v))
        rep += sprint("=== data ===")
        for k, v in self.data.items():
            if isinstance(v, FieldContainer):
                length = v.fieldlength
                count = v.fieldcount
                if length is not None:
                    rep += sprint("+", k, "(fields: %i, entries: %i)" % (count, length))
                else:
                    rep += sprint("+", k)
                if not listfields:
                    continue
                for k2, v2 in v.items():
                    rep += sprint("--", k2)
            else:
                rep += sprint("--", k)
        rep += sprint("============")
        print(rep)

    def _repr_dict(self):
        props = dict()
        if self.file is not None:
            if hasattr(self.file, "filename"):
                props["cache"] = self.file.filename
        props["source"] = self.path
        return props

    def __repr__(self):
        props = self._repr_dict()
        clsname = self.__class__.__name__
        result = clsname + "["
        for k, v in props.items():
            result += "%s=%s, " % (k, v)
        result = result[:-2] + "]"
        return result

    def _repr_pretty_(self, p, cycle):
        repr = self.__repr__()
        p.text(repr)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        dataset_type_registry[cls.__name__] = cls

    @classmethod
    @abc.abstractmethod
    def validate_path(cls, path, *args, **kwargs):
        # estimate whether we have a valid path for this dataset
        return False

    def __hash__(self):
        """Hash for Dataset instance to be derived from the file location."""
        # determinstic hash; note that hash() on a string is no longer deterministic in python3.
        hash_value = (
            int(hashlib.sha256(self.location.encode("utf-8")).hexdigest(), 16)
            % 10**10
        )
        return hash_value

    def __dask_tokenize__(self):
        return self.__hash__()

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
        store = zarr.DirectoryStore(fname, **zarr_kwargs)
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
                    if arr.dtype == np.uint64:
                        arr = arr.astype(np.int64)
                    elif arr.dtype == np.uint32:
                        arr = arr.astype(np.int32)
                task = da.to_zarr(
                    arr, os.path.join(fname, p, k), overwrite=True, compute=False
                )
                tasks.append(task)
        dask.compute(tasks)


class BaseSnapshot(Dataset):
    # TODO: Think more careful about hierarchy here. Right now BaseSnapshot is for Gadget-style sims
    _loader = BaseLoader

    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs):
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

        defaultattributes = ["config", "header", "parameters"]
        for k in self.metadata:
            name = k.strip("/").lower()
            if name in defaultattributes:
                self.__dict__[name] = self.metadata[k]

        self.boxsize = np.full(3, np.nan)

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        if path.endswith(".hdf5") or path.endswith(".zarr"):
            return True
        if os.path.isdir(path):
            files = os.listdir(path)
            prfxs = [f.split(".")[0] for f in files]
            sufxs = [f.split(".")[-1] for f in files]
            if len(set(prfxs)) > 1 or len(set(sufxs)) > 1:
                return False
            if sufxs[0] in ["hdf5", "zarr"]:
                return True
        return False

    def register_field(self, parttype, name=None, description=""):
        return self.data.register_field(parttype, name=name, description=description)


def deepdictkeycopy(olddict, newdict):
    """Recursively walk nested dictionary, only creating empty dictionaries
    for entries that are dictionaries themselves"""
    for k, v in olddict.items():
        if isinstance(v, MutableMapping):
            newdict[k] = {}
            deepdictkeycopy(v, newdict[k])


class Selector(object):
    """Base Class for data selection decorator factory"""

    def __init__(self):
        self.keys = None  # the keys we check for.
        self.data_backup = {}  # holds a copy of the species' fields
        self.data = {}  # holds the species' fields we operate on

    def __call__(self, fn, *args, **kwargs):
        def newfn(*args, **kwargs):
            # TODO: Add graceful exit/restore after exception in self.prepare
            self.data_backup = args[0].data
            self.data = {}
            deepdictkeycopy(self.data_backup, self.data)

            self.prepare(*args, **kwargs)
            if self.keys is None:
                raise NotImplementedError(
                    "Subclass implementation needed for self.keys!"
                )
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
