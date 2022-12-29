import logging
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import zarr


def walk_group(obj, tree, get_attrs=False):
    if len(tree) == 0:
        tree.update(**dict(attrs={}, groups=[], datasets=[]))
    if get_attrs and len(obj.attrs) > 0:
        tree["attrs"][obj.name] = dict(obj.attrs)
    if isinstance(obj, (h5py.Dataset, zarr.Array)):
        tree["datasets"].append([obj.name, obj.shape, obj.dtype])
    elif isinstance(obj, (h5py.Group, zarr.Group)):
        tree["groups"].append(obj.name)  # continue the walk
        for k, o in obj.items():
            walk_group(o, tree, get_attrs=get_attrs)


# def walk_group(obj, tree, get_attrs=False):
#    if len(tree) == 0:
#        tree.update(**dict(attrs={}, groups=[], datasets=[]))
#        if get_attrs and len(obj.attrs) > 0:
#            tree["attrs"][obj.name] = dict(obj.attrs)
#    for k, o in obj.items():
#        if isinstance(o, (h5py.Group, zarr.Group)):
#            tree["groups"].append(o.name)  # continue the walk
#            walk_group(o, tree, get_attrs=get_attrs)
#        elif isinstance(o, (h5py.Dataset, zarr.Array)):
#            tree["datasets"].append([o.name, o.shape, o.dtype])
#        else:
#            raise TypeError("Unknown type '%s'" % type(o))


def walk_zarrfile(fn, tree, get_attrs=True):
    with zarr.open(fn) as z:
        walk_group(z, tree, get_attrs=get_attrs)
    return tree


def walk_hdf5file(fn, tree, get_attrs=True):
    with h5py.File(fn, "r") as hf:
        walk_group(hf, tree, get_attrs=get_attrs)
    return tree


def create_mergedhdf5file(fn, files, max_workers=16, virtual=True):
    """Creates a virtual hdf5 file from list of given files. Virtual by default."""
    # first obtain all datasets and groups
    trees = [{} for i in range(len(files))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        result = executor.map(walk_hdf5file, files, trees)
    result = list(result)

    groups = set([item for r in result for item in r["groups"]])
    datasets = groups = set([item for r in result for item in r["groups"]])
    datasets = set([item[0] for r in result for item in r["datasets"]])

    def todct(lst):
        return {item[0]: (item[1], item[2]) for item in lst["datasets"]}

    dcts = [todct(lst) for lst in result]
    shapes = OrderedDict((d, {}) for d in datasets)

    def shps(i, k, s):
        return shapes[k].update({i: s[0]}) if s is not None else None

    [shps(i, k, dct.get(k)) for k in datasets for i, dct in enumerate(dcts)]
    dtypes = {}

    def dtps(k, s):
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
        groupfields = [f for f in groupfields if f.count("/") == group.count("/")]
        groupfields = sorted(groupfields)
        if len(groupfields) == 0:
            continue
        arr0 = chunks[groupfields[0]]
        for field in groupfields[1:]:
            arr = np.array(chunks[field])
            for a0, a in zip(arr0, arr):
                print(groupfields[0], field, a0, a)
            assert np.array_equal(arr0, arr)
        # then save the chunking information for this group
        groupchunks[field] = arr0

    # next fill merger file
    with h5py.File(fn, "w", libver="latest") as hf:
        # create groups
        for group in sorted(groups):
            if group == "/":
                continue  # nothing to do.
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
                for k in range(len(files)):
                    with h5py.File(files[k]) as hf_load:
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
        attrs0paths = attrs_key_lists[0]
        if len(attrs0paths) != len(set(attrs0paths).intersection(*attrs_key_lists)):
            raise NotImplementedError(
                "Some attribute paths not present in each partial data file."
            )
        # check for common key+values across all files
        attrs_same = {}
        attrs_differ = {}

        for apath in attrs0paths:
            attrs_same[apath] = {}
            attrs_differ[apath] = {}
            for k in result[0]["attrs"][apath]:
                attrvallist = [result[i]["attrs"][apath][k] for i in range(len(files))]
                attrval0 = attrvallist[0]
                if isinstance(attrval0, np.ndarray):
                    if not (np.all([np.array_equal(attrval0, v) for v in attrvallist])):
                        logging.info(apath, k, "has different values.")
                        print(apath, k)
                        attrs_differ[apath][k] = np.stack(attrvallist)
                        continue
                else:
                    # assert len(set(vals)) == 1, k + " has different values."  # TODO
                    if not len(set(attrvallist)) == 1:
                        logging.info(apath, k, "has different values.")
                        attrs_differ[apath][k] = np.array(attrval0)
                        continue
                attrs_same[apath][k] = attrval0
            for apath in attrs0paths:
                for k, v in attrs_same.get(apath, {}).items():
                    hf[apath].attrs[k] = v
                for k, v in attrs_differ.get(apath, {}).items():
                    hf[apath].attrs[k] = v
