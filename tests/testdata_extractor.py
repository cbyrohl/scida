# some script allowing us to copy HDF5 test files, but truncating fields to length 1
import os.path
import pathlib
import re
import shutil
from typing import List, Optional

import h5py
import numpy as np
import typer
from typing_extensions import Annotated

from scida import load
from scida.io import walk_hdf5file

app = typer.Typer()


@app.command()
def create_hdf5_testfile(src: str, dst: str, length: int = 1, verbose: bool = False):
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
                shape_new = (length,)
                tmpdata = None
                # vlen HDF5 dataset
                dt = g[path].dtype
                if h5py.check_vlen_dtype(dt):
                    # do not know how to treat, so skip for now
                    dtype = np.int32
                    shape_new = (length,)
                    tmpdata = np.array([0], dtype=dtype)
                else:
                    if len(shape) > 1:
                        shape_new = (length,) + shape[1:]
                        tmpdata = g[path][:length]
                        if length > 1:
                            np.random.seed(42)
                            idx = np.random.choice(
                                g[path].shape[0], length, replace=False
                            )
                            idx = np.sort(idx)
                            tmpdata = g[path][idx]
                    elif len(shape) == 1:
                        shape_new = (length,)
                        if len(g[path]) > 0:
                            tmpdata = g[path][0]
                            if length > 1:
                                np.random.seed(42)
                                idx = np.random.choice(
                                    g[path].shape[0], length, replace=False
                                )
                                idx = np.sort(idx)
                                tmpdata = g[path][idx]
                        else:
                            tmpdata = np.zeros((0,))
                    if len(shape) == 0:
                        # this is a scalar, no slicing
                        shape_new = ()
                        tmpdata = g[path]
                    elif shape[0] == 0:
                        shape_new = (0, *shape[1:])
                if verbose:
                    print(path, shape, shape_new, tmpdata.shape)
                f.create_dataset(path, shape=shape_new, dtype=dtype, data=tmpdata)
        for path, dct in attrs.items():
            for k in dct:
                f[path].attrs[k] = dct[k]


@app.command()
def minimize_series_hdf5(
    src: str,
    dst: str,
    exclude: Annotated[Optional[List[str]], typer.Option()] = None,
    max_snaps: Optional[int] = None,
    verbose: bool = False,
):
    """We minimize the series data by copying the folder structure,
    copying&finding all HDF5 files and truncating their record lengths."""
    pathlib.Path(dst).mkdir(exist_ok=True)
    if exclude is None:
        exclude = []

    if max_snaps is None:
        max_snaps = 9999999
    ps = PState(max_snaps)

    for root, dirs, files in os.walk(src):
        if any(["/" + k + "/" in root for k in exclude]):
            continue
        if not ps.resume_operation(root):
            continue

        print("=== %s ===" % root)
        for d in dirs:
            if not ps.resume_operation(d):
                continue
            pathlib.Path(os.path.join(dst, root[len(src) + 1 :], d)).mkdir(
                exist_ok=True
            )

        # hdf5 files
        hdf5files = [f for f in files if f.endswith(".hdf5")]
        for fn in hdf5files:
            if not ps.resume_operation(fn):
                continue
            srcfn = os.path.join(root, fn)
            dstfn = os.path.join(dst, srcfn[len(src) + 1 :])
            print(srcfn, "->", dstfn)
            create_hdf5_testfile(srcfn, dstfn, verbose=verbose)

        # txt files
        txtfiles = [f for f in files if f.endswith(".txt")]
        for fn in txtfiles:
            srcfn = os.path.join(root, fn)
            dstfn = os.path.join(dst, srcfn[len(src) + 1 :])
            print(srcfn, "->", dstfn)
            # only copy files below 100kib
            if os.path.getsize(srcfn) < 100 * 1024:
                shutil.copyfile(srcfn, dstfn)


def get_snapid(root):
    pattern = "_(\\d{4})"
    result = re.search(pattern, root)
    if result:
        result = result.group(1)
        return result
    # try again with 3 digits
    pattern = "_(\\d{3})"
    result = re.search(pattern, root)
    if result:
        result = result.group(1)
    return result


class PState:
    def __init__(self, max_snaps):
        self.snap_ids = set()
        self.max_snaps = max_snaps

    def resume_operation(self, path) -> bool:
        sid = get_snapid(path)
        if sid is None:
            return True  # does nto seem to be a chunk, ignore
        sid = int(sid)
        if sid not in self.snap_ids:
            if len(self.snap_ids) >= self.max_snaps:
                return False
            self.snap_ids.add(sid)
        return True


if __name__ == "__main__":
    app()
