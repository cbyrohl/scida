# some script allowing us to copy HDF5 test files, but truncating fields to length 1
import os.path
import pathlib

import h5py
import numpy as np
import typer

from astrodask import load
from astrodask.io import walk_hdf5file

app = typer.Typer()


@app.command()
def create_hdf5_testfile(src: str, dst: str):
    if os.path.isdir(src):
        # we support multi-file hdf5 files via our load() function
        ds = load(src)
        src = ds.file.filename
    tree = {}
    walk_hdf5file(src, tree)
    grps = tree["groups"]
    datasets = tree["datasets"]
    attrs = tree["attrs"]
    with h5py.File(dst, "w") as f:
        for k in grps:
            if k == "/":
                continue
            f.create_group(k)
        with h5py.File(src, "r") as g:
            for path, shape, dtype in datasets:
                shape_new = (1,)
                tmpdata = None
                # vlen HDF5 dataset
                dt = g[path].dtype
                print(path, dt, h5py.check_vlen_dtype(dt))
                if h5py.check_vlen_dtype(dt):
                    # do not know how to treat, so skip for now
                    dtype = np.int32
                    shape_new = (1,)
                    tmpdata = np.array([0], dtype=dtype)
                else:
                    if len(shape) > 1:
                        shape_new = (1,) + shape[1:]
                        tmpdata = g[path][0]
                    elif len(shape) == 1:
                        shape_new = (1,)
                        tmpdata = g[path][0]
                    if len(shape) == 0:
                        # this is a scalar, no slicing
                        shape_new = ()
                        tmpdata = g[path]
                print(path, shape, shape_new, tmpdata.shape)
                f.create_dataset(path, shape=shape_new, dtype=dtype, data=tmpdata)
        for path, dct in attrs.items():
            for k in dct:
                f[path].attrs[k] = dct[k]


@app.command()
def minimize_series_hdf5(src: str, dst: str):
    """We minimize the series data by copying the folder structure,
    copying&finding all HDF5 files and truncating their record lengths."""
    pathlib.Path(dst).mkdir(exist_ok=True)
    for root, dirs, files in os.walk(src):
        print("=== %s ===" % root)
        for d in dirs:
            pathlib.Path(os.path.join(dst, root[len(src) + 1 :], d)).mkdir(
                exist_ok=True
            )
        hdf5files = [f for f in files if f.endswith(".hdf5")]
        for fn in hdf5files:
            srcfn = os.path.join(root, fn)
            dstfn = os.path.join(dst, srcfn[len(src) + 1 :])
            print(srcfn, "->", dstfn)
            create_hdf5_testfile(srcfn, dstfn)


if __name__ == "__main__":
    app()
