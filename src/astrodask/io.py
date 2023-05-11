import abc
import logging
import os
import pathlib
import tempfile
from functools import partial
from os.path import join
from typing import Optional

import dask.array as da
import h5py
import numpy as np
import zarr

from astrodask.fields import FieldContainer
from astrodask.helpers_hdf5 import create_mergedhdf5file, walk_hdf5file, walk_zarrfile
from astrodask.misc import get_container_from_path, return_hdf5cachepath

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

    def load_metadata(self, **kwargs):
        """Take a quick glance at the metadata."""
        tree = {}
        walk_zarrfile(self.path, tree)
        return tree["attrs"]


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

    def get_chunkedfiles(self, fileprefix: Optional[str] = "") -> list:
        """
        Get all files in directory with given prefix.
        Parameters
        ----------
        fileprefix: Optional[str]
            Prefix of files to be loaded. If None, we take the first prefix.

        Returns
        -------

        """

        files = [join(self.path, f) for f in os.listdir(self.path)]
        files = [f for f in files if os.path.isfile(f)]  # ignore subdirectories
        if fileprefix is not None:
            files = np.array(
                [f for f in files if f.split("/")[-1].startswith(fileprefix)]
            )
        prfxs = sorted([f.split(".")[0] for f in files])
        if fileprefix is None:
            prfx = prfxs[0]
            prfxs = [prfx]
            files = np.array([f for f in files if f.startswith(prfx)])
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
    """groups_load: list of groups to load; all groups with datasets are loaded if groups_load==None"""
    # TODO: Refactor and rename
    inline_array = False  # inline arrays in dask; intended to improve dask scheduling (?). However, doesnt work with h5py (need h5pickle wrapper or zarr).
    if type(file) == h5py._hl.files.File:
        inline_array = False

    data = {}
    tree = {}
    if filetype == "hdf5":
        walk_hdf5file(location, tree)
    elif filetype == "zarr":
        walk_zarrfile(location, tree)
    else:
        raise ValueError("Unknown filetype ''" % filetype)

    # hosting all data
    rootcontainer = FieldContainer(fieldrecipes_kwargs=derivedfields_kwargs)

    datagroups = []  # names of groups with datasets
    # othergroups = []  # names of groups without datasets
    # datasets = []  # list of datasets (relative path from root)
    # each dataset entry is a tuple of (relpath, shape, dtype)
    # attributes = {}  # dictionary of attributes
    # key holds the relpath to the group and the value a dictionary of the attribute data

    grps = tree["groups"]
    dts = tree["datasets"]
    for grp in grps:
        for dt in dts:
            dn = dt[0]
            if dn.startswith(grp):
                datagroups.append(grp)
                break

    # groups
    if groups_load is not None:
        datagroups = sorted([grp for grp in datagroups if grp in groups_load])

    # Make each datadict entry a FieldContainer
    for group in datagroups:
        get_container_from_path(group, rootcontainer, create_missing=True)

    # datasets
    for i, dataset in enumerate(tree["datasets"]):
        # fpath is the path to the group containing given field
        fpath = "/".join(dataset[0].split("/")[:-1])
        toload = fpath == "" or fpath in datagroups
        if not toload:
            continue
        container = get_container_from_path(fpath, rootcontainer)
        # TODO: still change for nesting
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
            container.register_field(
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
            container[fieldname] = ds
    data = rootcontainer

    def walk_container(
        cntr, path="", handler_field=None, handler_group=None, withrecipes=False
    ):
        keykwargs = dict(withgroups=True, withrecipes=withrecipes)
        for ck in cntr.keys(**keykwargs):
            # we do not want to instantiate entry from recipe by calling cntr[ck] here
            entry = cntr[ck]
            newpath = path + "/" + ck
            if isinstance(entry, FieldContainer):
                if handler_group is not None:
                    handler_group(entry, newpath)
                walk_container(
                    entry,
                    newpath,
                    handler_field,
                    handler_group,
                    withrecipes=withrecipes,
                )
            else:
                if handler_field is not None:
                    handler_field(entry, newpath)

    # instantiate at least one field

    def instantiate_one_field(container: FieldContainer, path: str):
        keys = container.keys(withgroups=False, withrecipes=False)
        if len(keys) > 0:
            return
        keys = container.keys(withgroups=False, withrecipes=True)
        if len(keys) == 0:
            return
            # raise ValueError("Data container is empty, this is unexpected.")
        for k in keys:
            p = pathlib.Path(path) / k
            res = [k for k in tree["datasets"] if k[0] == str(p)]
            if len(res) == 0:
                continue  # no dataset info
            # tree_entry = res[0]
            # calling will instantiate.
            container.__getitem__(k)
            break

    walk_container(data, handler_group=instantiate_one_field)

    # create uids fields for all containers
    def create_uids(container: FieldContainer, path: str):
        nparts = -1
        chunks = None
        keys = container.keys(withgroups=False, withrecipes=False)
        for k in keys:
            v = container[k]
            nparts = v.shape[0]
            if len(v.chunks) == 1:
                # make same shape as other 1D arrays.
                chunks = v.chunks
                break
            else:
                # alternatively, attempt to use chunks from multi-dim array until found something better.
                chunks = (v.chunks[0],)
        if nparts > -1:
            container["uid"] = da.arange(nparts, chunks=chunks)
        else:
            print("no uid created for %s" % container.name)

    walk_container(data, handler_group=create_uids)

    # walk_container(data, handler_field=lambda x, y: print(x, y))

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


def load_metadata(path, **kwargs):
    loader = determine_loader(path)
    metadata = loader.load_metadata(**kwargs)
    return metadata


def load(path, **kwargs):
    loader = determine_loader(path)
    data, metadata = loader.load(**kwargs)
    file = loader.file
    tmpfile = loader.tempfile
    return data, metadata, file, tmpfile
