from collections import OrderedDict
import numpy as np
import h5py

from concurrent.futures import ProcessPoolExecutor

def walk_hdf5group(grp, tree, get_attrs=False):
    if len(tree) == 0:
        tree.update(**dict(attrs={}, groups=[], datasets=[]))
    if len(grp.attrs) > 0:
        tree["attrs"][grp.name] = dict(grp.attrs)
    for k in grp.keys():
        if isinstance(grp[k], h5py._hl.group.Group):
            tree["groups"].append(grp[k].name)
            walk_hdf5group(grp[k], tree, get_attrs=get_attrs)
        else:  # it's a dataset (?)
            tree["datasets"].append([grp[k].name, grp[k].shape, grp[k].dtype])


def walk_hdf5file(fn, tree, get_attrs=True):
    with h5py.File(fn, "r") as hf:
        walk_hdf5group(hf, tree, get_attrs=get_attrs)
    return tree


def create_virtualfile(fn, files, max_workers=16):
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

    # next fill virtual file
    with h5py.File(fn, "w") as hf:
        # create groups
        for group in groups:
            hf.create_group(group)
            groupfields = [field for field in shapes.keys() if
                           field.startswith(group) and field.count("/") - 1 == group.count("/")]
            # fill fields
            for field in groupfields:
                totentries = np.array([k[1] for k in chunks[field]]).sum()
                virtshape = (totentries,) + shapes[field][next(iter(shapes[field]))][1:]

                # create virtual sources
                vsources = []
                for k in shapes[field]:
                    vsources.append(
                        h5py.VirtualSource(files[k], name=field, shape=shapes[field][k], dtype=dtypes[field]))
                layout = h5py.VirtualLayout(shape=tuple(virtshape), dtype=dtypes[field])

                # fill virtual dataset
                offset = 0
                for vsource in vsources:
                    length = vsource.shape[0]
                    layout[offset: offset + length] = vsource
                    offset += length
                assert virtshape[0] == offset  # make sure we filled the array up fully.
                hf.create_virtual_dataset(field, layout)

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
                        if k not in ["NumPart_ThisFile"]: # expected
                            print(group, k, "has different values.")
                        attrsdiffer[group][k] = np.stack(vals)
                        continue
                else:
                    #assert len(set(vals)) == 1, k + " has different values."  # TODO
                    if not len(set(vals)) == 1:
                        print(group, k, "has different values.")
                        attrsdiffer[group][k] = np.array(vals)
                        continue
                attrssame[group][k] = vals[0]
        for group in result[0]["attrs"].keys():
            for k, v in attrssame[group].items():
                hf[group].attrs[k] = v
            for k, v in attrsdiffer[group].items():
                hf[group].attrs[k] = v

