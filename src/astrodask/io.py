import abc
import logging
import os
import tempfile
from functools import partial
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

    def load_metadata(self, **kwargs):
        """Take a quick glance at the metadata."""
        tree = {}
        walk_hdf5file(self.path, tree)
        return tree["attrs"]


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
        datadict = load_datadict_old(
            self.path,
            self.file,
            token=token,
            chunksize=chunksize,
            derivedfields_kwargs=derivedfields_kwargs,
            filetype="zarr",
        )
        return datadict


class ChunkedHDF5Loader(Loader):
    def __init__(self, path):
        super().__init__(path)
        self.tempfile = ""
        self.location = ""

    def load_metadata(self, fileprefix=""):
        """Take a quick glance at the metadata."""
        cachefp = return_hdf5cachepath(self.path)
        if cachefp is not None and os.path.isfile(cachefp):
            path = cachefp
        else:
            # get data from first file in list
            path = self.get_chunkedfiles(fileprefix)[0]
        tree = {}
        walk_hdf5file(path, tree)
        return tree["attrs"]

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
                datadict = self.load_cachefile(
                    cachefp,
                    token=token,
                    chunksize=chunksize,
                    derivedfields_kwargs=derivedfields_kwargs,
                )
                return datadict

        self.create_cachefile(fileprefix=fileprefix, virtualcache=virtualcache)

        datadict = self.load_cachefile(
            self.location,
            token=token,
            chunksize=chunksize,
            derivedfields_kwargs=derivedfields_kwargs,
        )
        return datadict

    def get_chunkedfiles(self, fileprefix) -> list:
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
        return files

    def create_cachefile(self, fileprefix="", virtualcache=False):
        cachefp = return_hdf5cachepath(self.path)
        files = self.get_chunkedfiles(fileprefix)

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

    def load_cachefile(
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
    lazy=True,  # if true, call da.from_array delayed
    filetype="hdf5",
):
    # TODO: Refactor and rename
    data = {}
    tree = {}
    if filetype == "hdf5":
        walk_hdf5file(location, tree)
    elif filetype == "zarr":
        walk_zarrfile(location, tree)
    else:
        raise ValueError("Unknown filetype ''" % filetype)
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

    # Make each datadict entry a FieldContainer
    datanew = FieldContainerCollection(
        data.keys(),
        derivedfields_kwargs=derivedfields_kwargs,
    )
    for k in data:
        datanew.new_container(k)

    # datasets

    for i, dataset in enumerate(tree["datasets"]):
        if groups_load is not None:
            toload = any([dataset[0].startswith(gname) for gname in groups_load])
        else:
            toload = any(
                [dataset[0].startswith("/" + g + "/") for g in groups_with_datasets]
            )
        if toload:
            # TODO: Still dont support more nested groups
            splt = dataset[0].rstrip("/").split("/")
            group = splt[1]
            fieldname = splt[-1]
            name = "Dataset" + str(token) + dataset[0].replace("/", "_")
            if "__dask_tokenize__" in file:  # check if the file contains the dask name.
                name = file["__dask_tokenize__"].attrs.get(dataset[0].strip("/"), name)
            if lazy and i > 0:  # need one non-lazy entry though later on...
                hds = file[dataset[0]]

                def field(
                    arrs,
                    snap=None,
                    h5path="",
                    chunksize="",
                    name="",
                    inline_array=False,
                    file=None,
                    **kwargs
                ):
                    hds = file[h5path]
                    arr = da.from_array(
                        hds,
                        chunks=chunksize,
                        name=name,
                        inline_array=inline_array,
                    )
                    return arr

                fnc = partial(
                    field,
                    h5path=dataset[0],
                    chunksize=chunksize,
                    name=name,
                    inline_array=inline_array,
                    file=file,
                )
                datanew[group].register_field(
                    name=fieldname, description=fieldname + ": lazy field from disk"
                )(fnc)
            else:
                hds = file[dataset[0]]
                ds = da.from_array(
                    hds,
                    chunks=chunksize,
                    name=name,
                    inline_array=inline_array,
                )
                datanew[group][fieldname] = ds
    data = datanew

    # Add a unique identifier for each element for each data type
    for p in data:
        for k in data[p].keys(allfields=True):
            v = data[p][k]
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
