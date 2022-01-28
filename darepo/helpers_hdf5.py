from collections import OrderedDict
import numpy as np
import h5py
import logging
import zarr

from concurrent.futures import ProcessPoolExecutor

def walk_group(grp, tree, get_attrs=False):
    if len(tree) == 0:
        tree.update(**dict(attrs={}, groups=[], datasets=[]))
    if len(grp.attrs) > 0:
        tree["attrs"][grp.name] = dict(grp.attrs)
    for k,o in grp.items():
        if isinstance(o, h5py._hl.group.Group) or isinstance(o, zarr.hierarchy.Group):
            tree["groups"].append(o.name)
            walk_group(o, tree, get_attrs=get_attrs)
        elif isinstance(o,h5py._hl.dataset.Dataset) or isinstance(o,zarr.core.Array):  # it's a dataset (?)
            tree["datasets"].append([o.name, o.shape, o.dtype])
        else:
            raise TypeError("Unknown type '%s'"%type(o))


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

    todct = lambda lst: {item[0]: (item[1], item[2]) for item in lst["datasets"]}
    dcts = [todct(lst) for lst in result]
    shapes = OrderedDict((d, {}) for d in datasets)
    shps = lambda i, k, s: shapes[k].update({i: s[0]}) if s is not None else None
    [shps(i, k, dct.get(k)) for k in datasets for i, dct in enumerate(dcts)]
    dtypes = {}
    dtps = lambda k, s: dtypes.update({k: s[1]}) if s is not None else None
    [dtps(k, dct.get(k)) for k in datasets for i, dct in enumerate(dcts)]

    # get shapes of the respective chunks
    chunks = {}
    for field in sorted(shapes.keys()):
        chunks[field] = [[k, shapes[field][k][0]] for k in shapes[field]]
    groupchunks = {}

    # assert that all datasets in a given group have the same chunks.
    for group in groups:
        groupfields = [field for field in shapes.keys() if
                       field.startswith(group) and field.count("/") - 1 == group.count("/")]
        if len(groupfields) == 0:
            continue
        arr0 = chunks[groupfields[0]]
        for field in groupfields[1:]:
            arr = np.array(chunks[field])
            assert np.array_equal(arr0, arr)
        # then save the chunking information for this group
        groupchunks[field] = arr0

    # next fill merger file
    with h5py.File(fn, "w") as hf:
        # create groups
        for group in groups:
            hf.create_group(group)
            groupfields = [field for field in shapes.keys() if
                           field.startswith(group) and field.count("/") - 1 == group.count("/")]

            # fill fields
            if virtual: # for virtual datasets, iterate over all fields and concat each file to virtual dataset
                for field in groupfields:
                    totentries = np.array([k[1] for k in chunks[field]]).sum()
                    newshape = (totentries,) + shapes[field][next(iter(shapes[field]))][1:]

                    # create virtual sources
                    vsources = []
                    for k in shapes[field]:
                        print(k)
                        vsources.append(
                            h5py.VirtualSource(files[k], name=field, shape=shapes[field][k], dtype=dtypes[field]))
                    layout = h5py.VirtualLayout(shape=tuple(newshape), dtype=dtypes[field])

                    # fill virtual dataset
                    offset = 0
                    for vsource in vsources:
                        length = vsource.shape[0]
                        layout[offset: offset + length] = vsource
                        offset += length
                    assert newshape[0] == offset  # make sure we filled the array up fully.
                    hf.create_virtual_dataset(field, layout)
            else: # copied dataset. For performance, we iterate differently: Loop over each file's fields
                for field in groupfields:
                    totentries = np.array([k[1] for k in chunks[field]]).sum()
                    newshape = (totentries,) + shapes[field][next(iter(shapes[field]))][1:]
                    dset = hf.create_dataset(field, shape=newshape)
                counters = {field: 0 for field in groupfields}
                for i in range(len(files)):
                    with h5py.File(files[i]) as hf_load:
                        for field in groupfields:
                            n = chunks[field][i][1]
                            hf[field][counters[field]:counters[field]+n] = hf_load[field]
                            counters[field] = counters[field] + n

        # save information regarding chunks
        grp = hf.create_group("_chunks")
        for k, v in groupchunks.items():
            grp.attrs[k] = v

        # write the attributes
        # find attributes that change across data sets
        attrsdiffer = {}
        attrssame = {}
        for group in groups:
            attrsdiffer[group] = {}
            attrssame[group] = {}
            for i in range(len(files)):
                keys = []
                if group in result[i]["attrs"]:
                    keys += result[i]["attrs"][group].keys()
                keys = set(keys)
            for k in keys:
                vals = [result[i]["attrs"][group][k] for i in range(len(files))]
                if isinstance(vals[0], np.ndarray):
                    if not (np.all([np.array_equal(vals[0], v) for v in vals])):
                        logging.info(group, k, "has different values.")
                        attrsdiffer[group][k] = np.stack(vals)
                        continue
                else:
                    #assert len(set(vals)) == 1, k + " has different values."  # TODO
                    if not len(set(vals)) == 1:
                        logging.info(group, k, "has different values.")
                        attrsdiffer[group][k] = np.array(vals)
                        continue
                attrssame[group][k] = vals[0]
        for group in result[0]["attrs"].keys():
            for k, v in attrssame[group].items():
                hf[group].attrs[k] = v
            for k, v in attrsdiffer[group].items():
                hf[group].attrs[k] = v

