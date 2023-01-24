import abc
import logging
import os
import tempfile
from os.path import join

import dask.array as da
import h5py
import numpy as np
import zarr

from astrodask.fields import FieldContainerCollection
from astrodask.helpers_hdf5 import create_mergedhdf5file, walk_hdf5file, walk_zarrfile
from astrodask.misc import return_hdf5cachepath

log = logging.getLogger(__name__)


class Loader(abc.ABC):
    def __init__(self, path):
        self.path = path
        self.file = None  # open file handle if available
        self.tempfile = None  # reference to tempfile where needed

    @abc.abstractmethod
    def load(self):
        pass


class HDF5Loader(Loader):
    def __init__(self, path):
        super().__init__(path)
        self.location = ""

    def load(
        self,
        overwrite=False,
        fileprefix="",
        token="",
        chunksize="auto",
        virtualcache=False,
        derivedfields_kwargs=None,
    ):
        self.location = self.path
        tree = {}
        walk_hdf5file(self.location, tree=tree)
        file = h5py.File(self.location, "r")
        datadict = load_datadict_old(
            self.path,
            file,
            token=token,
            chunksize=chunksize,
            derivedfields_kwargs=derivedfields_kwargs,
        )
        self.file = file
        return datadict


class ZarrLoader(Loader):
    def __init__(self, path):
        super().__init__(path)
        self.location = ""

    def load(
        self,
        overwrite=False,
        fileprefix="",
        token="",
        chunksize="auto",
        virtualcache=False,
        derivedfields_kwargs=None,
    ):
        self.location = self.path
        tree = {}
        walk_zarrfile(self.location, tree=tree)
        self.file = zarr.open(self.location)
        self.load_from_tree(tree)


class ChunkedHDF5Loader(Loader):
    def __init__(self, path):
        super().__init__(path)
        self.tempfile = ""
        self.location = ""

    def load(
        self,
        overwrite=False,
        fileprefix="",
        token="",
        chunksize="auto",
        virtualcache=False,
        derivedfields_kwargs=None,
    ):

        cachefp = return_hdf5cachepath(self.path)
        if cachefp is not None and os.path.isfile(cachefp):
            if not overwrite:
                # we are done; just use cached version
                datadict = self.load_singlehdf5(
                    cachefp,
                    token=token,
                    chunksize=chunksize,
                    derivedfields_kwargs=derivedfields_kwargs,
                )
                return datadict

        files = [join(self.path, f) for f in os.listdir(self.path)]
        files = [f for f in files if os.path.isfile(f)]  # ignore subdirectories
        files = np.array([f for f in files if f.split("/")[-1].startswith(fileprefix)])
        prfxs = [f.split(".")[0] for f in files]
        if len(set(prfxs)) > 1:
            print("Available prefixes:", set(prfxs))
            raise ValueError(
                "More than one file prefix in directory, specify 'fileprefix'."
            )
        nmbrs = [int(f.split(".")[-2]) for f in files]
        sortidx = np.argsort(nmbrs)
        files = files[sortidx]

        self.location = cachefp
        if cachefp is None:
            # no filepath available
            self.tempfile = tempfile.NamedTemporaryFile("wb", suffix=".hdf5")
            self.location = self.tempfile.name
            log.warning(
                "No caching directory specified. Initial file read will remain slow."
            )

        try:
            create_mergedhdf5file(cachefp, files, virtual=virtualcache)
        except Exception as ex:
            os.remove(cachefp)  # remove failed attempt at merging file
            raise ex
        datadict = self.load_singlehdf5(
            self.location,
            token=token,
            chunksize=chunksize,
            derivedfields_kwargs=derivedfields_kwargs,
        )
        return datadict

    def load_singlehdf5(
        self, location, token="", chunksize="auto", derivedfields_kwargs=None
    ):
        self.file = h5py.File(location, "r")
        datadict = load_datadict_old(
            location,
            self.file,
            token=token,
            chunksize=chunksize,
            derivedfields_kwargs=derivedfields_kwargs,
        )
        return datadict


def load_datadict_old(
    location,
    file,
    groups_load=None,
    token="",
    chunksize=None,
    derivedfields_kwargs=None,
):
    # TODO: Refactor and rename
    data = {}
    tree = {}
    walk_hdf5file(location, tree)
    """groups_load: list of groups to load; all groups with datasets are loaded if groups_load==None"""
    inline_array = False  # inline arrays in dask; intended to improve dask scheduling (?). However, doesnt work with h5py (need h5pickle wrapper or zarr).
    if type(file) == h5py._hl.files.File:
        inline_array = False
    if groups_load is None:
        toload = True
        groups_with_datasets = sorted(
            set([d[0].split("/")[1] for d in tree["datasets"]])
        )
    # groups
    for group in tree["groups"]:
        # TODO: Do not hardcode dataset/field groups
        if groups_load is not None:
            toload = any([group.startswith(gname) for gname in groups_load])
        else:
            toload = any([group == "/" + g for g in groups_with_datasets])
        if toload:
            data[group.split("/")[1]] = {}
    # datasets
    for dataset in tree["datasets"]:
        if groups_load is not None:
            toload = any([dataset[0].startswith(gname) for gname in groups_load])
        else:
            toload = any(
                [dataset[0].startswith("/" + g + "/") for g in groups_with_datasets]
            )
        if toload:
            # TODO: Still dont support more nested groups
            group = dataset[0].split("/")[1]
            name = "Dataset" + str(token) + dataset[0].replace("/", "_")
            if "__dask_tokenize__" in file:  # check if the file contains the dask name.
                name = file["__dask_tokenize__"].attrs.get(dataset[0].strip("/"), name)
            ds = da.from_array(
                file[dataset[0]],
                chunks=chunksize,
                name=name,
                inline_array=inline_array,
            )
            data[group][dataset[0].split("/")[-1]] = ds

    # Make each datadict entry a FieldContainer
    datanew = FieldContainerCollection(
        data.keys(),
        derivedfields_kwargs=derivedfields_kwargs,
    )
    for k in data:
        datanew.new_container(k, **data[k])
    data = datanew

    # Add a unique identifier for each element for each data type
    for p in data:
        for k, v in data[p].items():
            nparts = v.shape[0]
            if len(v.chunks) == 1:
                # make same shape as other 1D arrays.
                chunks = v.chunks
                break
            else:
                # alternatively, attempt to use chunks from multi-dim array until found something better.
                chunks = (v.chunks[0],)

        data[p]["uid"] = da.arange(nparts, chunks=chunks)

    # attrs
    metadata = tree["attrs"]
    return data, metadata


def determine_loader(path):
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, ".zgroup")):
            # object is a zarr object
            loader = ZarrLoader(path)
        else:
            # otherwise expect this is a chunked HDF5 file
            loader = ChunkedHDF5Loader(path)
    else:
        # we are directly given a target file
        loader = HDF5Loader(path)
    return loader


def load(path, **kwargs):
    loader = determine_loader(path)
    data, metadata = loader.load(**kwargs)
    file = loader.file
    tmpfile = loader.tempfile
    return data, metadata, file, tmpfile
