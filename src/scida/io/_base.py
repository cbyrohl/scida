"""
Basic IO functionality for scida.
"""

import abc
import logging
import os
import pathlib
import tempfile
from functools import partial
from os.path import join
from typing import Optional, Union

import dask.array as da
import h5py
import numpy as np
import zarr

from scida.config import get_config
from scida.fields import FieldContainer, walk_container
from scida.helpers_hdf5 import create_mergedhdf5file, walk_hdf5file, walk_zarrfile
from scida.io.fits import fitsrecords_to_daskarrays
from scida.misc import get_container_from_path, return_hdf5cachepath

log = logging.getLogger(__name__)


class InvalidCacheError(Exception):
    """
    Raised if a cache file is invalid.
    """

    pass


class Loader(abc.ABC):
    """
    Base class for loaders.
    """

    def __init__(self, path):
        """
        Initialize loader by passing target path.
        Parameters
        ----------
        path: str
            path to the dataset
        """
        self.path = path
        self.file = None  # open file handle if available
        self.tempfile = None  # reference to tempfile where needed

    @abc.abstractmethod
    def load(self):
        """
        Load data from file.
        Returns
        -------

        """
        pass


class FITSLoader(Loader):
    """
    FITS file loader.
    """

    def __init__(self, path):
        """
        Initialize loader by passing target path.
        Parameters
        ----------
        path: str
           path to resource
        """
        super().__init__(path)
        self.location = self.path

    def load(
        self,
        overwrite_cache=False,
        fileprefix="",
        token="",
        chunksize="auto",
        virtualcache=False,
        **kwargs
    ):
        """
        Load data from FITS file.
        Parameters
        ----------
        overwrite_cache: bool
            Overwrite cache file if it exists.
        fileprefix: str
            Prefix of files to be loaded. If None, we take the first prefix.
        token: str
            Token to be used for dask arrays.
        chunksize: str
            Chunksize for dask arrays.
        virtualcache: bool
            Whether to use virtual caching.
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------

        """
        log.warning("FITS file support is work-in-progress.")
        self.location = self.path
        from astropy.io import fits

        ext = 1
        with fits.open(self.location, memmap=True, mode="denywrite") as hdulist:
            arr = hdulist[ext].data
        darrs = fitsrecords_to_daskarrays(arr)

        derivedfields_kwargs = kwargs.get("derivedfields_kwargs", {})
        withunits = kwargs.get("withunits", False)
        rootcontainer = FieldContainer(
            fieldrecipes_kwargs=derivedfields_kwargs, withunits=withunits
        )

        for name, darr in darrs.items():
            rootcontainer[name] = darr

        metadata = {}
        return rootcontainer, metadata

    def load_metadata(self, **kwargs):
        """
        Take a quick glance at the metadata, only attributes.
        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of attributes.
        """
        from astropy.io import fits

        ext = 0
        with fits.open(self.location, memmap=True, mode="denywrite") as hdulist:
            return hdulist[ext].header

    def load_metadata_all(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of attributes.
        """
        from astropy.io import fits

        ext = 1
        with fits.open(self.location, memmap=True, mode="denywrite") as hdulist:
            arr = hdulist[ext].data
        return arr.header


class HDF5Loader(Loader):
    """
    HDF5 file loader.
    """

    def __init__(self, path):
        """
        Initialize loader by passing target path.
        Parameters
        ----------
        path: str
            path to resource
        """
        super().__init__(path)
        self.location = ""

    def load(
        self,
        overwrite_overwrite=False,
        fileprefix="",
        token="",
        chunksize="auto",
        virtualcache=False,
        **kwargs
    ):
        """
        Load data from HDF5 file.
        Parameters
        ----------
        overwrite_overwrite: bool
            Overwrite cache file if it exists.
        fileprefix: str
            Prefix of files to be loaded. If None, we take the first prefix.
        token: str
            Token to be used for dask arrays.
        chunksize: str
            Chunksize for dask arrays.
        virtualcache: bool
            Whether to use virtual caching.
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of data and metadata.

        """
        self.location = self.path
        tree = {}
        walk_hdf5file(self.location, tree=tree)
        file = h5py.File(self.location, "r")
        data, metadata = load_datadict_old(
            self.path, file, token=token, chunksize=chunksize, **kwargs
        )
        self.file = file
        return data, metadata

    def load_metadata(self, **kwargs):
        """
        Take a quick glance at the metadata.
        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of attributes.

        """
        tree = {}
        walk_hdf5file(self.path, tree)
        return tree["attrs"]

    def load_metadata_all(self, **kwargs):
        """
        Take a quick glance at the metadata.
        Parameters
        ----------
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of attributes.
        """
        tree = {}
        walk_hdf5file(self.path, tree)
        return tree


class ZarrLoader(Loader):
    """
    Zarr file loader.
    """

    def __init__(self, path):
        """
        Initialize loader by passing target path.
        Parameters
        ----------
        path: str
            path to resource
        """
        super().__init__(path)
        self.location = ""

    def load(
        self,
        overwrite_cache=False,
        fileprefix="",
        token="",
        chunksize="auto",
        virtualcache=False,
        **kwargs
    ):
        """
        Load data from Zarr file.

        Parameters
        ----------
        overwrite_cache: bool
            Overwrite cache file if it exists.
        fileprefix: str
            Prefix of files to be loaded. If None, we take the first prefix.
        token: str
            Token to be used for dask arrays.
        chunksize: str
            Chunksize for dask arrays.
        virtualcache: bool
            Whether to use virtual caching.
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of data and metadata.
        """
        self.location = self.path
        tree = {}
        walk_zarrfile(self.location, tree=tree)
        self.file = zarr.open(self.location)
        data, metadata = load_datadict_old(
            self.path,
            self.file,
            token=token,
            chunksize=chunksize,
            filetype="zarr",
            **kwargs
        )
        return data, metadata

    def load_metadata(self, **kwargs):
        """
        Take a quick glance at the metadata.
        Parameters
        ----------
        kwargs: dict

        Returns
        -------
        dict
            Dictionary of attributes.

        """
        tree = {}
        walk_zarrfile(self.path, tree)
        return tree["attrs"]


class ChunkedHDF5Loader(Loader):
    """
    Chunked HDF5 file loader.
    """

    def __init__(self, path):
        """
        Initialize loader by passing target path.

        Parameters
        ----------
        path: str
            path to resource
        """
        super().__init__(path)
        self.tempfile = ""
        self.location = ""

    def load_metadata(self, fileprefix="", use_cachefile=True, **kwargs):
        """
        Take a quick glance at the metadata.

        Parameters
        ----------
        use_cachefile: bool
            Whether to use cache file (if present).
        fileprefix: str
            Prefix of files to be loaded. If None, we take the first prefix.
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of attributes.
        """
        cachefp = return_hdf5cachepath(self.path, fileprefix=fileprefix)
        if cachefp is not None and os.path.isfile(cachefp) and use_cachefile:
            path = cachefp
        else:
            # get data from first file in list
            paths = self.get_chunkedfiles(
                fileprefix, choose_prefix=kwargs.get("choose_prefix", False)
            )
            if len(paths) == 0:
                raise ValueError("No files for prefix '%s' found." % fileprefix)
            path = paths[0]
        tree = {}
        walk_hdf5file(path, tree)
        return tree["attrs"]

    def load(
        self,
        overwrite_cache=False,
        fileprefix="",
        choose_prefix=False,
        token="",
        chunksize="auto",
        virtualcache=False,
        **kwargs
    ):
        """
        Load data from chunked HDF5 file.

        Parameters
        ----------
        overwrite_cache: bool
            Overwrite cache file if it exists.
        fileprefix: str
            Prefix of files to be loaded. If None, we take the first prefix.
        choose_prefix: bool
            Whether to choose the prefix if multiple are available.
        token: str
            Token to be used for dask arrays.
        chunksize: str
            Chunksize for dask arrays.
        virtualcache: bool
            Whether to use virtual caching.
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of data and metadata.

        """
        cachefp = return_hdf5cachepath(self.path, fileprefix=fileprefix)
        # three cases we need to cache:
        create = False
        if cachefp is None:
            # 1. no cachepath given
            create = True
        elif not os.path.isfile(cachefp):
            # 2. no cachefile exists
            create = True
        elif overwrite_cache:
            # 3. cachefile exists, but overwrite=True
            create = True

        if create:
            print_cachefile_creation = kwargs.get("print_cachefile_creation", True)
            self.create_cachefile(
                fileprefix=fileprefix,
                virtualcache=virtualcache,
                verbose=print_cachefile_creation,
                choose_prefix=choose_prefix,
            )

        try:
            data, metadata = self.load_cachefile(
                cachefp, token=token, chunksize=chunksize, **kwargs
            )
        except InvalidCacheError:
            # if we get an error, we try to create a new cache file (once)
            log.info("Invalid cache file, attempting to create new one.")
            os.remove(cachefp)
            self.create_cachefile(fileprefix=fileprefix, virtualcache=virtualcache)
            data, metadata = self.load_cachefile(
                cachefp, token=token, chunksize=chunksize, **kwargs
            )
        return data, metadata

    def get_chunkedfiles(
        self, fileprefix: Optional[str] = "", choose_prefix=False
    ) -> list:
        """
        Get all files in directory with given prefix.
        Parameters
        ----------
        choose_prefix
        fileprefix: Optional[str]
            Prefix of files to be loaded. If None, we take the first prefix.

        Returns
        -------

        """
        return _get_chunkedfiles(
            self.path, fileprefix=fileprefix, choose_prefix=choose_prefix
        )

    def create_cachefile(
        self, fileprefix="", virtualcache=False, verbose=None, choose_prefix=False
    ):
        """
        Create a cache file from the chunked HDF5 files.
        Parameters
        ----------
        fileprefix: str
            Prefix of files to be loaded. If None, we take the first prefix.
        virtualcache: bool
            Whether to use virtual caching.
        verbose: bool
            Whether to print messages.
        choose_prefix: bool
            Whether to choose the prefix if multiple are available.

        Returns
        -------
        None
        """
        config = get_config()
        print_msg = verbose
        if verbose is None:
            print_msg = config.get("print_cachefile_creation", True)
        if print_msg:
            print("Creating cache file, this may take a while...")
        cachefp = return_hdf5cachepath(self.path, fileprefix=fileprefix)
        files = self.get_chunkedfiles(fileprefix, choose_prefix=choose_prefix)

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
            if os.path.exists(cachefp):
                os.remove(cachefp)  # remove failed attempt at merging file
            raise ex

        with h5py.File(cachefp, "r+") as hf:
            hf.attrs["_cachingcomplete"] = True  # mark that caching complete

    def load_cachefile(self, location, token="", chunksize="auto", **kwargs):
        """
        Load data from cache file.

        Parameters
        ----------
        location: str
            Location of cache file.
        token: str
            Token to be used for dask arrays.
        chunksize: str
            Chunksize for dask arrays.
        kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            Tuple of data and metadata.
        """
        try:
            self.file = h5py.File(location, "r")
        except (
            OSError
        ) as e:  # file potentially corrupted, raise InvalidCacheError and potentially recover
            raise InvalidCacheError(
                """Cache file '%s' might be invalid. An OSError '%s' was raised.
                Delete file and try again."""
                % (location, str(e))
            )
        hf = self.file
        cache_valid = False
        if "_cachingcomplete" in hf.attrs:
            cache_valid = hf.attrs["_cachingcomplete"]
        if not cache_valid:
            raise InvalidCacheError(
                "Cache file '%s' is not valid. Delete file and try again." % location
            )

        datadict = load_datadict_old(
            location, self.file, token=token, chunksize=chunksize, **kwargs
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
    withunits=False,
):
    """
    Load data from HDF5/Zarr resource into a dictionary of dask arrays.
    Parameters
    ----------
    location: str
        Location of resource.
    file: Union[h5py.File, zarr.hierarchy.Group]
        File handle.
    groups_load: list
        List of groups to load.
    token: str
        Token to be used for dask arrays.
    chunksize: str
        Chunksize for dask arrays.
    derivedfields_kwargs: dict
        Keyword arguments for derived fields.
    lazy: bool
        Whether to load data lazily.
    filetype: str
        Filetype of resource.
    withunits: bool
        Whether to load units.

    Returns
    -------
    dict
        Dictionary of dask arrays.

    """
    # TODO: Refactor and rename function
    inline_array = False  # inline arrays in dask; intended to improve dask scheduling (?). However, doesnt work with h5py (need h5pickle wrapper or zarr).
    if isinstance(file, h5py.File):
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
    rootcontainer = FieldContainer(
        fieldrecipes_kwargs=derivedfields_kwargs, withunits=withunits
    )

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
        # ignore if scalar
        if len(dataset[1]) == 0:
            continue
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

        # we do not support HDF5 vlen dtype
        if filetype == "hdf5":
            hds = file[dataset[0]]
            dt = hds.dtype
            if h5py.check_vlen_dtype(dt):
                log.warning(
                    "HDF5 vlen dtypes not supported. Skip loading field '%s'"
                    % fieldname
                )
                continue

        lz = lazy and i > 0  # need one non-lazy field (?)
        fieldpath = dataset[0]
        _add_hdf5arr_to_fieldcontainer(
            file,
            container,
            fieldpath,
            fieldname,
            name,
            chunksize,
            inline_array=inline_array,
            lazy=lz,
        )

    data = rootcontainer

    dtsdict = {k[0]: k[1:] for k in tree["datasets"]}

    # create uids fields for all containers
    def create_uids(container: FieldContainer, path: str):
        """
        helper function to create unique ids for each container
        """
        nparts = -1
        keys = container.keys(withgroups=False, withrecipes=True)
        for k in keys:
            dpath = path + "/" + k
            shape = dtsdict[dpath][0]
            nparts = shape[0]
        if nparts > -1:
            # TODO: as we do not load any fields any more we do not have a reference for the dask chunking.
            container["uid"] = da.arange(nparts)
        else:
            print("no uid created for %s" % container.name)

    walk_container(data, handler_group=create_uids)

    # attrs
    metadata = tree["attrs"]
    return data, metadata


def determine_loader(path, **kwargs):
    """
    Determine loader from path.
    Parameters
    ----------
    path: str
        Path to resource.
    kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    Loader
    """
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, ".zgroup")):
            # object is a zarr object
            loader = ZarrLoader(path)
            return loader
        newpath = _cachefile_available_in_path(path, **kwargs)
        if newpath:  # check whether a virtual merged HDF5 file is present in folder
            loader = HDF5Loader(newpath)
        else:
            # otherwise expect this is a chunked HDF5 file
            loader = ChunkedHDF5Loader(path)
    elif not os.path.exists(path):
        raise ValueError("Path '%s' does not exist" % path)
    elif str(path).endswith(".hdf5"):
        # we are directly given a target file
        loader = HDF5Loader(path)
    elif str(path).endswith(".fits"):
        loader = FITSLoader(path)
    else:
        raise ValueError("Unknown filetype of path '%s'" % path)
    return loader


def load_metadata(path, **kwargs):
    """
    Take a quick glance at the metadata.
    Parameters
    ----------
    path: str
        Path to resource.
    kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    dict
        Dictionary of attributes.
    """
    loader = determine_loader(path, **kwargs)
    metadata = loader.load_metadata(**kwargs)
    return metadata


def load_metadata_all(path, **kwargs):
    """
    Take a quick glance at the metadata.
    Parameters
    ----------
    path: str
        Path to resource.
    kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    dict
        Dictionary of attributes.

    """
    loader = determine_loader(path, **kwargs)
    metadata = loader.load_metadata_all(**kwargs)
    return metadata


def load(path, **kwargs):
    """
    Load data from resource.
    Parameters
    ----------
    path: str
        Path to resource.
    kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    tuple
        Tuple of data and metadata.
    """
    loader = determine_loader(path, **kwargs)
    data, metadata = loader.load(**kwargs)
    file = loader.file
    tmpfile = loader.tempfile
    return data, metadata, file, tmpfile


def _cachefile_available_in_path(
    path: str, fileprefix: Optional[str] = "", **kwargs
) -> Union[bool, str]:
    """
    Check whether a virtual merged HDF5 file is present in folder.
    Parameters
    ----------
    path: str
    fileprefix: Optional[str]
    kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    Union[bool, str]
        False if no cache file is available, otherwise the path to the cache file.

    """
    try:
        chnkfiles = _get_chunkedfiles(path, fileprefix=fileprefix)
    except ValueError:
        return False
    if len(chnkfiles) == 0:
        return False
    fn = chnkfiles[0]
    p = pathlib.Path(fn)
    s = p.name.split(".")
    if len(s) != 3:  # format "PREFIX.NMBR.HDF5"
        return False
    name_new = s[0] + "." + s[2]
    pnew = p.parent / name_new
    if pnew.exists():
        return str(pnew)
    return False


def _add_hdf5arr_to_fieldcontainer(
    file,
    container,
    fieldpath,
    fieldname,
    name,
    chunksize,
    inline_array=False,
    lazy=True,
):
    """
    Add HDF5 array to field container.
    Parameters
    ----------
    file: h5py.File
        file handle
    container: FieldContainer
        container to add field to
    fieldpath: str
        path to field in file
    fieldname: str
        name of field
    name: str
        name of dask array
    chunksize: str
        chunksize for dask array
    inline_array: bool
        whether to inline array in dask
    lazy: bool
        whether to load array lazily

    Returns
    -------
    None
    """
    if not lazy:
        hds = file[fieldpath]
        ds = da.from_array(
            hds,
            chunks=chunksize,
            name=name,
            inline_array=inline_array,
        )
        container[fieldname] = ds
        return

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
        """
        helper function to load resource field into dask array
        """
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
        h5path=fieldpath,
        chunksize=chunksize,
        name=name,
        inline_array=inline_array,
        file=file,
    )
    container.register_field(
        name=fieldname, description=fieldname + ": lazy field from disk"
    )(fnc)


def _get_chunkedfiles(
    path, fileprefix: Optional[str] = "", choose_prefix=False
) -> list:
    """
    Get all files in directory with given prefix.
    Parameters
    ----------
    choose_prefix
    fileprefix: Optional[str]
        Prefix of files to be loaded. If None, we take the first prefix.

    Returns
    -------

    """

    fns = [f for f in os.listdir(path)]
    files = [join(path, f) for f in fns]
    files = [
        f
        for i, f in enumerate(files)
        if not any(fns[i].startswith(start) for start in [".", "bak"])
    ]  # ignore hidden files
    files = [f for f in files if os.path.isfile(f)]  # ignore subdirectories
    if fileprefix is not None:
        files = [f for f in files if f.split("/")[-1].startswith(fileprefix)]
    # ignore backup files
    files = [f for f in files if not f.endswith("~")]
    files = [f for f in files if not f.endswith(".bak")]
    files = [f for f in files if not f.endswith(".swp")]

    files = np.array(files)
    prfxs = sorted([f.split(".")[0] for f in files])
    if fileprefix is None:
        prfx = prfxs[0]
        prfxs = [prfx]
        files = np.array([f for f in files if f.startswith(prfx)])
    if len(set(prfxs)) > 1:
        if choose_prefix:
            # choose prefix that occurs more often
            prfx = max(set(prfxs), key=prfxs.count)
            files = np.array([f for f in files if f.startswith(prfx)])
        else:
            # print("Available prefixes:", set(prfxs))
            msg = (
                "More than one file prefix in directory '%s', specify 'fileprefix'."
                % path
            )
            raise ValueError(msg)
    # determine numbers where possible
    numbers = []
    files_numbered = []
    for f in files:
        try:
            numbers.append(int(f.split(".")[-2]))
            files_numbered.append(f)
        except ValueError:
            pass  # do not add file
    sortidx = np.argsort(numbers)
    files_numbered = np.array(files_numbered)
    files = files_numbered[sortidx]
    return files
