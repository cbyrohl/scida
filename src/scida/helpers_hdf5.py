"""
Helper functions for hdf5 and zarr file processing.
"""

import logging
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import zarr

from scida.config import get_config

log = logging.getLogger(__name__)


def get_dtype(obj):
    """
    Get the data type of given h5py.Dataset or zarr.Array object

    Parameters
    ----------
    obj: h5py.Dataset or zarr.Array
        object to get dtype from

    Returns
    -------
    dtype: numpy.dtype
        dtype of the object
    """
    if isinstance(obj, h5py.Dataset):
        try:
            dtype = obj.dtype
        except TypeError as e:
            msg = "data type '<u6' not understood"
            if msg == e.__str__():
                # MTNG defines 6 byte unsigned integers, which are not supported by h5py
                # could not figure out how to query type in h5py other than the reporting error.
                # (any call to .dtype will try to resolve "<u6" to a numpy dtype, which fails)
                # we just handle this as 64 bit unsigned integer
                dtype = np.uint64
            else:
                raise e
        return dtype
    elif isinstance(obj, zarr.Array):
        return obj.dtype
    else:
        return None


def walk_group(obj, tree, get_attrs=False, scalar_to_attr=True):
    """
    Walks through a h5py.Group or zarr.Group object and fills the tree dictionary with
    information about the datasets and groups.

    Parameters
    ----------
    obj: h5py.Group or zarr.Group
        object to walk through
    tree: dict
        dictionary to fill recursively
    get_attrs: bool
        whether to get attributes of each object
    scalar_to_attr: bool
        whether to convert scalar datasets to attributes

    Returns
    -------

    """
    if len(tree) == 0:
        tree.update(**dict(attrs={}, groups=[], datasets=[]))
    if get_attrs and len(obj.attrs) > 0:
        tree["attrs"][obj.name] = dict(obj.attrs)
    if isinstance(obj, (h5py.Dataset, zarr.Array)):
        dtype = get_dtype(obj)
        tree["datasets"].append([obj.name, obj.shape, dtype])
        if scalar_to_attr and len(obj.shape) == 0:
            tree["attrs"][obj.name] = obj[()]
    elif isinstance(obj, (h5py.Group, zarr.Group)):
        tree["groups"].append(obj.name)  # continue the walk
        for o in obj.values():
            walk_group(o, tree, get_attrs=get_attrs)


def walk_zarrfile(fn, tree, get_attrs=True):
    """
    Walks through a zarr file and fills the tree dictionary with
    information about the datasets and groups.

    Parameters
    ----------
    fn: str
        file path to zarr file to walk through
    tree: dict
        dictionary to fill recursively
    get_attrs: bool
        whether to get attributes of each object

    Returns
    -------
    tree: dict
        filled dictionary
    """
    with zarr.open(fn) as z:
        walk_group(z, tree, get_attrs=get_attrs)
    return tree


def walk_hdf5file(fn, tree, get_attrs=True):
    """
    Walks through a hdf5 file and fills the tree dictionary with
    information about the datasets and groups.

    Parameters
    ----------
    fn: str
        file path to hdf5 file to walk through
    tree: dict
        dictionary to fill recursively
    get_attrs: bool
        whether to get attributes of each object

    Returns
    -------
    tree: dict
        filled dictionary
    """
    with h5py.File(fn, "r") as hf:
        walk_group(hf, tree, get_attrs=get_attrs)
    return tree


def create_mergedhdf5file(
    fn, files, max_workers=None, virtual=True, groupwise_shape=False
):
    """
    Creates a virtual hdf5 file from list of given files. Virtual by default.

    Parameters
    ----------
    fn: str
        file to write to
    files: list
        files to merge
    max_workers: int
        parallel workers to process files
    virtual: bool
        whether to create linked ("virtual") dataset on disk (otherwise copy)
    groupwise_shape: bool
        whether to require shapes to be the same within a group

    Returns
    -------
    None
    """
    if max_workers is None:
        # read from config
        config = get_config()
        max_workers = config.get("nthreads", 16)
    # first obtain all datasets and groups
    trees = [{} for i in range(len(files))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        result = executor.map(walk_hdf5file, files, trees)
    result = list(result)

    groups = set([item for r in result for item in r["groups"]])
    datasets = set([item[0] for r in result for item in r["datasets"]])

    def todct(lst):
        """helper func"""
        return {item[0]: (item[1], item[2]) for item in lst["datasets"]}

    dcts = [todct(lst) for lst in result]
    shapes = OrderedDict((d, {}) for d in datasets)

    def shps(i, k, s):
        """helper func"""
        return shapes[k].update({i: s[0]}) if s is not None else None

    [shps(i, k, dct.get(k)) for k in datasets for i, dct in enumerate(dcts)]
    dtypes = {}

    def dtps(k, s):
        """helper func"""
        return dtypes.update({k: s[1]}) if s is not None else None

    [dtps(k, dct.get(k)) for k in datasets for i, dct in enumerate(dcts)]

    # get shapes of the respective chunks
    chunks = {}
    for field in sorted(shapes.keys()):
        chunks[field] = [[k, shapes[field][k][0]] for k in shapes[field]]
    groupchunks = {}

    # assert that all datasets in a given group have the same chunks.
    for group in sorted(groups):
        if group == "/":
            group = ""  # this is needed to have consistent levels for 0th level
        groupfields = [f for f in shapes.keys() if f.startswith(group)]
        groupfields = [f for f in groupfields if f.count("/") - 1 == group.count("/")]
        groupfields = sorted(groupfields)
        if len(groupfields) == 0:
            continue
        arr0 = chunks[groupfields[0]]
        for field in groupfields[1:]:
            arr = np.array(chunks[field])
            if groupwise_shape and not np.array_equal(arr0, arr):
                raise ValueError("Requiring same shape (see 'groupwise_shape' flag)")
            # then save the chunking information for this group
            groupchunks[field] = arr0

    # next fill merger file
    with h5py.File(fn, "w", libver="latest") as hf:
        # create groups
        for group in sorted(groups):
            if group == "/":
                group = ""
            else:
                hf.create_group(group)
            groupfields = [
                field
                for field in shapes.keys()
                if field.startswith(group) and field.count("/") - 1 == group.count("/")
            ]
            if len(groupfields) == 0:
                continue

            # fill fields
            if virtual:
                # for virtual datasets, iterate over all fields and concat each file to virtual dataset
                for field in groupfields:
                    totentries = np.array([k[1] for k in chunks[field]]).sum()
                    newshape = (totentries,) + shapes[field][next(iter(shapes[field]))][
                        1:
                    ]

                    # create virtual sources
                    vsources = []
                    for k in shapes[field]:
                        vsources.append(
                            h5py.VirtualSource(
                                files[k],
                                name=field,
                                shape=shapes[field][k],
                                dtype=dtypes[field],
                            )
                        )
                    layout = h5py.VirtualLayout(
                        shape=tuple(newshape), dtype=dtypes[field]
                    )

                    # fill virtual dataset
                    offset = 0
                    for vsource in vsources:
                        length = vsource.shape[0]
                        layout[offset : offset + length] = vsource
                        offset += length
                    assert (
                        newshape[0] == offset
                    )  # make sure we filled the array up fully.
                    hf.create_virtual_dataset(field, layout)
            else:  # copied dataset. For performance, we iterate differently: Loop over each file's fields
                for field in groupfields:
                    totentries = np.array([k[1] for k in chunks[field]]).sum()
                    extrashapes = shapes[field][next(iter(shapes[field]))][1:]
                    newshape = (totentries,) + extrashapes
                    hf.create_dataset(field, shape=newshape, dtype=dtypes[field])
                counters = {field: 0 for field in groupfields}
                for k, fl in enumerate(files):
                    with h5py.File(fl) as hf_load:
                        for field in groupfields:
                            n = shapes[field].get(k, [0, 0])[0]
                            if n == 0:
                                continue
                            offset = counters[field]
                            hf[field][offset : offset + n] = hf_load[field]
                            counters[field] = offset + n

        # save information regarding chunks
        grp = hf.create_group("_chunks")
        for k, v in groupchunks.items():
            grp.attrs[k] = v

        # write the attributes
        # find attributes that change across data sets
        attrs_key_lists = [
            list(v["attrs"].keys()) for v in result
        ]  # attribute paths for each file
        attrspaths_all = set().union(*attrs_key_lists)
        attrspaths_intersec = set(attrspaths_all).intersection(*attrs_key_lists)
        attrspath_diff = attrspaths_all.difference(attrspaths_intersec)
        if attrspaths_all != attrspaths_intersec:
            # if difference only stems from missing datasets (and their assoc. attrs); thats fine
            if not attrspath_diff.issubset(datasets):
                raise NotImplementedError(
                    "Some attribute paths not present in each partial data file."
                )
        # check for common key+values across all files
        attrs_same = {}
        attrs_differ = {}

        nfiles = len(files)

        for apath in sorted(attrspaths_all):
            attrs_same[apath] = {}
            attrs_differ[apath] = {}
            attrsnames = set().union(
                *[
                    result[i]["attrs"][apath]
                    for i in range(nfiles)
                    if apath in result[i]["attrs"]
                ]
            )
            for k in attrsnames:
                # we ignore apaths and k existing in some files.
                attrvallist = [
                    result[i]["attrs"][apath][k]
                    for i in range(nfiles)
                    if apath in result[i]["attrs"] and k in result[i]["attrs"][apath]
                ]
                attrval0 = attrvallist[0]
                if isinstance(attrval0, np.ndarray):
                    if not (np.all([np.array_equal(attrval0, v) for v in attrvallist])):
                        log.debug("%s: %s has different values." % (apath, k))
                        attrs_differ[apath][k] = np.stack(attrvallist)
                        continue
                else:
                    same = len(set(attrvallist)) == 1
                    if isinstance(attrval0, np.floating):
                        # for floats we do not require binary equality
                        # (we had some incident...)
                        same = np.allclose(attrval0, attrvallist)
                    if not same:
                        log.debug("%s: %s has different values." % (apath, k))
                        attrs_differ[apath][k] = np.array(attrvallist)
                        continue
                attrs_same[apath][k] = attrval0
        for apath in attrspaths_all:
            for k, v in attrs_same.get(apath, {}).items():
                hf[apath].attrs[k] = v
            for k, v in attrs_differ.get(apath, {}).items():
                hf[apath].attrs[k] = v
