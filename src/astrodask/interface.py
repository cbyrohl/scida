import abc
import hashlib
import logging
import os
from collections.abc import MutableMapping
from typing import Dict, List, Union

import dask
import dask.array as da
import numpy as np
import zarr

import astrodask.io
from astrodask.fields import FieldContainer, FieldContainerCollection
from astrodask.helpers_misc import make_serializable
from astrodask.misc import check_config_for_dataset, sprint
from astrodask.registries import dataset_type_registry

log = logging.getLogger(__name__)


class MixinMeta(type):
    def __call__(cls, *args, **kwargs):
        newcls = cls
        mixins = kwargs.pop("mixins", None)
        # print("mixins", mixins, cls)
        if isinstance(mixins, list) and len(mixins) > 0:
            name = cls.__name__ + "With" + "And".join([m.__name__ for m in mixins])
            # adjust entry point if __init__ available in some mixin
            nms = dict(cls.__dict__)
            # need to make sure first mixin init is called over cls init
            nms["__init__"] = mixins[0].__init__
            newcls = type(name, (*mixins, cls), nms)
        return type.__call__(newcls, *args, **kwargs)


class Dataset(metaclass=MixinMeta):
    def __init__(
        self,
        path,
        chunksize="auto",
        virtualcache=True,
        overwritecache=False,
        fileprefix="",
        hints=None,
        **kwargs
    ):
        super().__init__()
        self.hints = hints if hints is not None else {}
        self.path = path
        self.file = None
        # need this 'tempfile' reference to keep garbage collection away for the tempfile
        self.tempfile = None
        self.location = str(path)
        self.chunksize = chunksize
        self.virtualcache = virtualcache
        self.overwritecache = overwritecache

        # Let's find the data and metadata for the object at 'path'
        self.metadata = {}
        self._metadata_raw = {}
        self.data = FieldContainerCollection()

        if not os.path.exists(self.path):
            raise Exception("Specified path '%s' does not exist." % self.path)

        loadkwargs = dict(
            overwrite=self.overwritecache,
            fileprefix=fileprefix,
            virtualcache=virtualcache,
            derivedfields_kwargs=dict(snap=self),
            token=self.__dask_tokenize__(),
        )

        res = astrodask.io.load(path, **loadkwargs)
        self.data = res[0]
        self._metadata_raw = res[1]
        self.file = res[2]
        self.tempfile = res[3]
        self._cached = False

        # any identifying metadata?
        candidates = check_config_for_dataset(self._metadata_raw)
        if len(candidates) > 0:
            dsname = candidates[0]
            print("Dataset is identified as '%s'." % dsname)
            self.hints["dsname"] = dsname

    def _info_custom(self):
        """
        Custom information to be printed by info() method.
        Returns
        -------

        """
        return None

    def info(self, listfields: bool = False):
        """
        Print information about the dataset.
        Parameters
        ----------
        listfields: bool
            If True, list all fields in the dataset.

        Returns
        -------

        """
        rep = ""
        rep += sprint(self.__class__.__name__)
        props = self._repr_dict()
        for k, v in props.items():
            rep += sprint("%s: %s" % (k, v))
        if self._info_custom() is not None:
            rep += self._info_custom()
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

    def _repr_dict(self) -> Dict[str, str]:
        """
        Return a dictionary of properties to be printed by __repr__ method.
        Returns
        -------
        dict
        """
        props = dict()
        props["source"] = self.path
        return props

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        Returns
        -------
        str
        """
        props = self._repr_dict()
        clsname = self.__class__.__name__
        result = clsname + "["
        for k, v in props.items():
            result += "%s=%s, " % (k, v)
        result = result[:-2] + "]"
        return result

    def _repr_pretty_(self, p, cycle):
        """
        Pretty print representation for IPython.
        Parameters
        ----------
        p
        cycle

        Returns
        -------

        """
        rpr = self.__repr__()
        p.text(rpr)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if cls.__name__ == "Delay":
            return  # nothing to register for Delay objects
        if "Mixin" in cls.__name__:
            return  # do not register classes with Mixins
        dataset_type_registry[cls.__name__] = cls

    @classmethod
    @abc.abstractmethod
    def validate_path(cls, path, *args, **kwargs):
        """
        Validate whether the given path is a valid path for this dataset.
        Parameters
        ----------
        path
        args
        kwargs

        Returns
        -------

        """
        return False

    def __hash__(self) -> int:
        """
        Hash for Dataset instance to be derived from the file location.

        Returns
        -------
        int
        """
        # determinstic hash; note that hash() on a string is no longer deterministic in python3.
        hash_value = (
            int(hashlib.sha256(self.location.encode("utf-8")).hexdigest(), 16)
            % 10**10
        )
        return hash_value

    def __dask_tokenize__(self) -> int:
        """
        Token for dask to be derived -- naively from the file location.
        Returns
        -------
        int
        """
        return self.__hash__()

    def return_data(self) -> FieldContainerCollection:
        """
        Return the data container.
        Returns
        -------

        """
        return self.data

    def save(
        self,
        fname,
        fields: Union[str, Dict[str, Union[List[str], Dict[str, da.Array]]]] = "all",
        overwrite=True,
        zarr_kwargs=None,
        cast_uints=False,
    ) -> None:
        """
        Save the dataset to a file using the 'zarr' format.
        Parameters
        ----------
        fname: str
            Filename to save to.
        fields: str or dict
            dictionary of dask arrays to save. If equal to 'all', save all fields in current dataset.
        overwrite
            overwrite existing file
        zarr_kwargs
            optional arguments to pass to zarr
        cast_uints
            need to potentially cast uints to ints for some compressions; TODO: clean this up

        Returns
        -------

        """
        # We use zarr, as this way we have support to directly write into the file by the workers
        # (rather than passing back the data chunk over the scheduler to the interface)
        # Also, this way we can leverage new features, such as a large variety of compression methods.
        # cast_uints: if true, we cast uints to ints; needed for some compressions (particularly zfp)
        if zarr_kwargs is None:
            zarr_kwargs = {}
        store = zarr.DirectoryStore(fname, **zarr_kwargs)
        root = zarr.group(store, overwrite=overwrite)

        # Metadata
        defaultattributes = ["Config", "Header", "Parameters"]
        for dctname in defaultattributes:
            if dctname.lower() in self.__dict__:
                grp = root.create_group(dctname)
                dct = self.__dict__[dctname.lower()]
                for k, v in dct.items():
                    v = make_serializable(v)
                    grp.attrs[k] = v
        # Data
        tasks = []
        ptypes = self.data.keys()
        if isinstance(fields, dict):
            ptypes = fields.keys()
        elif isinstance(fields, str):
            if not fields == "all":
                raise ValueError("Invalid field specifier.")
        else:
            raise ValueError("Invalid type for fields.")
        for p in ptypes:
            root.create_group(p)
            if fields == "all":
                fieldkeys = self.data[p]
            else:
                if isinstance(fields[p], dict):
                    fieldkeys = fields[p].keys()
                else:
                    fieldkeys = fields[p]
            for k in fieldkeys:
                if not isinstance(fields, str) and isinstance(fields[p], dict):
                    arr = fields[p][k]
                else:
                    arr = self.data[p][k]
                if np.any(np.isnan(arr.shape)):
                    arr.compute_chunk_sizes()  # very inefficient (have to do it separately for every array)
                    arr = arr.rechunk(chunks="auto")
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
    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        self.boxsize = np.full(3, np.nan)
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

        defaultattributes = ["config", "header", "parameters"]
        for k in self._metadata_raw:
            name = k.strip("/").lower()
            if name in defaultattributes:
                self.__dict__[name] = self._metadata_raw[k]

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> bool:
        path = str(path)
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


def deepdictkeycopy(olddict, newdict) -> None:
    """
    Recursively walk nested dictionary, only creating empty dictionaries for entries that are dictionaries themselves.
    Parameters
    ----------
    olddict
    newdict

    Returns
    -------

    """
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

    def prepare(self, *args, **kwargs) -> None:
        raise NotImplementedError("Subclass implementation needed!")

    def finalize(self, *args, **kwargs) -> None:
        args[0].data = self.data_backup
