"""
Base dataset class and its handling.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
from typing import Any

import dask
import dask.array as da
import numpy as np
import zarr

import scida.io
from scida.fields import FieldContainer
from scida.helpers_misc import make_serializable, sprint
from scida.interfaces.mixins import UnitMixin
from scida.misc import check_config_for_dataset
from scida.registries import dataset_type_registry

log = logging.getLogger(__name__)


class MixinMeta(type):
    """
    Metaclass for Mixin classes.
    """

    def __call__(cls, *args, **kwargs):
        mixins = kwargs.pop("mixins", None)
        newcls = create_datasetclass_with_mixins(cls, mixins)
        return type.__call__(newcls, *args, **kwargs)


class BaseDataset(metaclass=MixinMeta):
    """
    Base class for all datasets.
    """

    def __init__(
        self,
        path: str,
        chunksize: str | int = "auto",
        virtualcache: bool = True,
        overwrite_cache: bool = False,
        fileprefix: str = "",
        hints: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a dataset object.

        Parameters
        ----------
        path: str
            Path to the dataset.
        chunksize: int or str
            Chunksize for dask arrays.
        virtualcache: bool
            Whether to use virtual caching.
        overwrite_cache: bool
            Whether to overwrite existing cache.
        fileprefix: str
            Prefix for files to scan for.
        hints: dict
            Hints for the dataset.
        kwargs: dict
            Additional keyword arguments.
        """
        super().__init__()
        self.hints = hints if hints is not None else {}
        self.path = path
        self.file = None
        # need this 'tempfile' reference to keep garbage collection away for the tempfile
        self.tempfile = None
        self.location = str(path)
        self.chunksize = chunksize
        self.virtualcache = virtualcache
        self.overwrite_cache = overwrite_cache
        self.withunits = kwargs.get("units", False)

        # Let's find the data and metadata for the object at 'path'
        self.metadata: dict[str, Any] = {}
        self._metadata_raw: dict[str, Any] = {}
        self.data = FieldContainer(withunits=self.withunits)

        if not os.path.exists(self.path):
            raise Exception("Specified path '%s' does not exist." % self.path)

        loadkwargs = dict(
            overwrite_cache=self.overwrite_cache,
            fileprefix=fileprefix,
            virtualcache=virtualcache,
            nonvirtual_datasets=kwargs.get("nonvirtual_datasets", None),
            derivedfields_kwargs=dict(snap=self),
            token=self.__dask_tokenize__(),
            withunits=self.withunits,
        )
        if "choose_prefix" in kwargs:
            loadkwargs["choose_prefix"] = kwargs["choose_prefix"]

        res = scida.io.load(path, **loadkwargs)
        self.data = res[0]
        self._metadata_raw = res[1]
        self.file = res[2]
        self.tempfile = res[3]
        self._cached = False

        # any identifying metadata?
        if "dsname" not in self.hints:
            candidates = check_config_for_dataset(self._metadata_raw, path=self.path)
            if len(candidates) > 0:
                dsname = candidates[0]
                log.debug("Dataset is identified as '%s'." % dsname)
                self.hints["dsname"] = dsname

    def _info_custom(self):
        """
        Custom information to be printed by info() method.

        Returns
        -------
        None
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
        None
        """
        rep = ""
        rep += "class: " + sprint(self.__class__.__name__)
        props = self._repr_dict()
        for k, v in props.items():
            rep += sprint("%s: %s" % (k, v))
        if self._info_custom() is not None:
            rep += self._info_custom()
        rep += sprint("=== data ===")
        rep += self.data.info(name="root")
        rep += sprint("============")
        print(rep)

    def _repr_dict(self) -> dict[str, str]:
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
        None
        """
        rpr = self.__repr__()
        p.text(rpr)

    def __init_subclass__(cls, *args, **kwargs):
        """
        Register subclasses in the dataset type registry.
        Parameters
        ----------
        args: list
        kwargs: dict

        Returns
        -------
        None
        """
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
        bool

        """
        return False

    def __hash__(self) -> int:
        """
        Hash for Dataset instance to be derived from the file location.

        Returns
        -------
        int
        """
        # deterministic hash; note that hash() on a string is no longer deterministic in python3.
        hash_value = (
            int(hashlib.sha256(self.location.encode("utf-8")).hexdigest(), 16) % 10**10
        )
        return hash_value

    def __getitem__(self, item):
        return self.data[item]

    def __dask_tokenize__(self) -> int:
        """
        Token for dask to be derived -- naively from the file location.

        Returns
        -------
        int
        """
        return self.__hash__()

    def return_data(self) -> FieldContainer:
        """
        Return the data container.

        Returns
        -------
        FieldContainer
        """
        return self.data

    def save(
        self,
        fname: str,
        fields: (
            str | dict[str, list[str] | dict[str, da.Array]] | FieldContainer
        ) = "all",
        overwrite: bool = True,
        zarr_kwargs: dict[str, Any] | None = None,
        cast_uints: bool = False,
        extra_attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        Save the dataset to a file using the 'zarr' format.
        Parameters
        ----------
        extra_attrs: dict
            additional attributes to save in the root group
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
        None
        """
        # We use zarr, as this way we have support to directly write into the file by the workers
        # (rather than passing back the data chunk over the scheduler to the interface)
        # Also, this way we can leverage new features, such as a large variety of compression methods.
        # cast_uints: if true, we cast uints to ints; needed for some compressions (particularly zfp)
        if zarr_kwargs is None:
            zarr_kwargs = {}
        if overwrite and os.path.exists(fname):
            if os.path.isdir(fname) and len(os.listdir(fname)) > 0:
                # check if it is a zarr group/array
                is_zarr = os.path.exists(os.path.join(fname, ".zgroup"))
                is_zarr |= os.path.exists(os.path.join(fname, ".zarray"))
                if not is_zarr:
                    raise Exception(
                        f"Directory '{fname}' exists and is not a zarr group. "
                        "Refusing to overwrite for safety."
                    )
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
        if extra_attrs is not None:
            for k, v in extra_attrs.items():
                root.attrs[k] = v
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
                if hasattr(arr, "magnitude"):  # if we have units, remove those here
                    # TODO: save units in metadata!
                    arr = arr.magnitude
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


class Dataset(BaseDataset):
    """
    Base class for datasets with some functions to be overwritten by subclass.
    """

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        """
        Validate whether the given path is a valid path for this dataset.

        Parameters
        ----------
        path: str
            Path to the dataset.
        args: list
        kwargs: dict

        Returns
        -------
        bool
        """
        return True

    @classmethod
    def _clean_metadata_from_raw(cls, rawmetadata):
        """Clean metadata from raw metadata"""
        return {}


# TODO: remove this class (?), should be dynamically created in load() where needed
class DatasetWithUnitMixin(UnitMixin, Dataset):
    """
    Dataset with units.
    """

    def __init__(self, *args, **kwargs):
        """Initialize dataset with units."""
        super().__init__(*args, **kwargs)


class Selector(object):
    """Base Class for data selection decorator factory"""

    def __init__(self):
        """
        Initialize the selector.
        """
        self.keys = None  # the keys we check for.
        # holds a copy of the species' fields
        self.data_backup = FieldContainer()
        # holds the species' fields we operate on
        self.data: FieldContainer = FieldContainer()

    def __call__(self, fn, *args, **kwargs):
        """
        Call the selector.

        Parameters
        ----------
        fn: function
            Function to be decorated.
        args: list
        kwargs: dict

        Returns
        -------
        function
            Decorated function.

        """

        def newfn(*args, **kwargs):
            # TODO: Add graceful exit/restore after exception in self.prepare
            self.data_backup = args[0].data
            self.data = args[0].data.copy_skeleton()
            # deepdictkeycopy(self.data_backup, self.data)

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
        """
        Prepare the data for selection. To be implemented in subclasses.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        raise NotImplementedError("Subclass implementation needed!")

    def finalize(self, *args, **kwargs) -> None:
        """
        Finalize the data after selection. To be implemented in subclasses.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        args[0].data = self.data_backup


def create_datasetclass_with_mixins(cls: type, mixins: list | None) -> type:
    """
    Create a new class from a given class and a list of mixins.

    Parameters
    ----------
    cls:
        dataset class to be extended
    mixins:
        list of mixin classes to be added

    Returns
    -------
    Type[BaseDataset]
        new class with mixins
    """
    newcls = cls
    if isinstance(mixins, list) and len(mixins) > 0:
        # check for duplicates within the mixins list itself
        seen = set()
        for m in mixins:
            if m in seen:
                raise ValueError(
                    "Mixin '%s' appears multiple times in the mixins list. "
                    "Please remove duplicates." % getattr(m, "__name__", str(m))
                )
            seen.add(m)

        # check whether any mixin already in cls hierarchy recursively
        def has_mixin_in_hierarchy(cls, m):
            if m in cls.__bases__:
                return True
            for b in cls.__bases__:
                if has_mixin_in_hierarchy(b, m):
                    return True
            return False

        for m in mixins:
            if has_mixin_in_hierarchy(cls, m):
                raise ValueError(
                    "Mixin '%s' is already present in the class hierarchy of '%s'. "
                    "Please remove it from the mixins list to avoid duplication."
                    % (getattr(m, "__name__", str(m)), cls.__name__)
                )
        name = cls.__name__ + "With" + "And".join([m.__name__ for m in mixins])
        # adjust entry point if __init__ available in some mixin
        nms = dict(cls.__dict__)
        # need to make sure first mixin init is called over cls init
        nms["__init__"] = mixins[0].__init__
        newcls = type(name, (*mixins, cls), nms)
    return newcls
