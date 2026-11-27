import os
import pathlib
import re

import dask
import h5py
import numpy as np
import pytest

from scida.config import get_config
from scida.helpers_hdf5 import (
    SOURCE_SHARDS_PATH,
    VIRTUAL_SOURCES_PATH,
    create_mergedhdf5file,
)
from scida.io import load
from scida.misc import return_hdf5cachepath


@pytest.fixture
def snapshot_shards(tmp_path, monkeypatch):
    monkeypatch.setitem(get_config(), "nthreads", 1)
    directory = tmp_path / "snapshot"
    directory.mkdir()
    first = directory / "snap_000.0.hdf5"
    second = directory / "snap_000.1.hdf5"
    for path, values in ((first, [1, 2, 3]), (second, [11, 12, 13])):
        with h5py.File(path, "w") as shard:
            shard["Masses"] = values
    return directory, first, second


@pytest.mark.integration
def test_ioload_virtual_cache_checks_missing_shards(snapshot_shards, monkeypatch):
    directory, first, missing_shard = snapshot_shards
    fields, _, cache_file, _ = load(directory, virtualcache=True)
    with cache_file:
        cache_path = pathlib.Path(cache_file.filename)
        assert cache_file["Masses"].is_virtual
        assert set(cache_file[VIRTUAL_SOURCES_PATH][:]) == {
            os.fsencode(first),
            os.fsencode(missing_shard),
        }
        np.testing.assert_array_equal(fields["Masses"].compute(), [1, 2, 3, 11, 12, 13])

    scans = []
    for method in ("listdir", "scandir"):
        scan = getattr(os, method)

        def count_scan(path=".", *, scan=scan):
            if not isinstance(path, int) and pathlib.Path(path) == directory:
                scans.append(path)
            return scan(path)

        monkeypatch.setattr(os, method, count_scan)

    fields, _, cache_file, _ = load(directory, virtualcache=True)
    with cache_file:
        np.testing.assert_array_equal(fields["Masses"].compute(), [1, 2, 3, 11, 12, 13])
    assert len(scans) == 1

    missing_shard.unlink()
    with pytest.raises(FileNotFoundError, match=re.escape(missing_shard.name)):
        load(directory, virtualcache=True)
    assert cache_path.is_file()


@pytest.mark.integration
@pytest.mark.parametrize("load_directory", [False, True], ids=["direct", "adjacent"])
def test_ioload_legacy_virtual_cache_checks_all_shards(snapshot_shards, load_directory):
    directory, first, second = snapshot_shards
    cache_path = directory / "snap_000.hdf5"
    with h5py.File(cache_path, "w", libver="latest") as cache:
        cache.create_group("_chunks")
        for shard, field in ((first, "Masses"), (second, "ParticleIDs")):
            layout = h5py.VirtualLayout(shape=(3,), dtype=np.int64)
            layout[:] = h5py.VirtualSource(shard.name, "Masses", shape=(3,))
            cache.create_virtual_dataset(field, layout)

    input_path = directory if load_directory else cache_path
    fields, _, cache_file, _ = load(input_path)
    with cache_file:
        np.testing.assert_array_equal(fields["ParticleIDs"].compute(), [11, 12, 13])

    second.unlink()
    with pytest.raises(FileNotFoundError, match=re.escape(second.name)):
        load(input_path)
    if load_directory:
        first.unlink()
        with pytest.raises(FileNotFoundError, match=re.escape(first.name)):
            load(input_path)


@pytest.mark.integration
def test_ioload_materialized_cache_does_not_need_original_shards(snapshot_shards):
    directory, first, second = snapshot_shards
    _, _, cache_file, _ = load(directory, virtualcache=False)
    with cache_file:
        assert not cache_file["Masses"].is_virtual
    first.unlink()
    second.unlink()

    fields, _, cache_file, _ = load(directory, virtualcache=False)
    with cache_file:
        np.testing.assert_array_equal(fields["Masses"].compute(), [1, 2, 3, 11, 12, 13])


@pytest.mark.integration
def test_ioload_refreshes_cache_for_new_shards_of_same_snapshot(snapshot_shards):
    directory, first, second = snapshot_shards
    # A shard with no virtual fields is still part of the original snapshot.
    with h5py.File(second, "a") as shard:
        del shard["Masses"]
        shard["ParticleIDs"] = [21, 22, 23]
    _, _, cache_file, _ = load(
        directory, virtualcache=True, nonvirtual_datasets=["ParticleIDs"]
    )
    with cache_file:
        cache_path = pathlib.Path(cache_file.filename)
        assert set(cache_file[SOURCE_SHARDS_PATH][:]) == {
            os.fsencode(path) for path in (first, second)
        }
        original_inode = cache_path.stat().st_ino

    for name in ("snap_999.0.hdf5", "snap_000.4.hdf5.bak"):
        with h5py.File(directory / name, "w") as shard:
            shard["Masses"] = [99]
    _, _, cache_file, _ = load(directory)
    with cache_file:
        assert pathlib.Path(cache_file.filename).stat().st_ino == original_inode

    third = directory / "snap_000.2.hdf5"
    with h5py.File(third, "w") as shard:
        shard["Masses"] = [31, 32, 33]
        shard["ParticleIDs"] = [41, 42, 43]
    fields, _, cache_file, _ = load(directory)
    with cache_file:
        assert cache_file["Masses"].is_virtual
        assert not cache_file["ParticleIDs"].is_virtual
        np.testing.assert_array_equal(fields["Masses"].compute(), [1, 2, 3, 31, 32, 33])
        np.testing.assert_array_equal(
            fields["ParticleIDs"].compute(), [21, 22, 23, 41, 42, 43]
        )


@pytest.mark.integration
def test_ioload_failed_refresh_keeps_previous_virtual_cache(
    snapshot_shards, monkeypatch
):
    directory, _, _ = snapshot_shards
    _, _, cache_file, _ = load(directory, virtualcache=True)
    with cache_file:
        cache_path = pathlib.Path(cache_file.filename)
    with h5py.File(directory / "snap_000.2.hdf5", "w") as shard:
        shard["Masses"] = [31, 32, 33]

    def fail_merge(path, *args, **kwargs):
        with h5py.File(path, "w") as incomplete:
            incomplete["Partial"] = [1]
        raise RuntimeError("interrupted cache build")

    monkeypatch.setattr("scida.io._base.create_mergedhdf5file", fail_merge)
    with pytest.raises(RuntimeError, match="interrupted cache build"):
        load(directory)
    assert list(cache_path.parent.iterdir()) == [cache_path]
    with h5py.File(cache_path, "r") as original:
        np.testing.assert_array_equal(original["Masses"][:], [1, 2, 3, 11, 12, 13])


@pytest.mark.integration
def test_ioload_refresh_keeps_open_readers_independent(snapshot_shards):
    directory, _, _ = snapshot_shards
    old_fields, _, old_file, _ = load(directory, virtualcache=True)
    with old_file:
        with h5py.File(directory / "snap_000.2.hdf5", "w") as shard:
            shard["Masses"] = [31, 32, 33]
        new_fields, _, new_file, _ = load(directory)
        with new_file:
            old, new = dask.compute(old_fields["Masses"], new_fields["Masses"])
    np.testing.assert_array_equal(old, [1, 2, 3, 11, 12, 13])
    np.testing.assert_array_equal(new, [1, 2, 3, 11, 12, 13, 31, 32, 33])


@pytest.mark.integration
@pytest.mark.parametrize("remove_index", [1, 2], ids=["middle", "last"])
def test_ioload_rejects_missing_shards_declared_in_header(
    snapshot_shards, remove_index
):
    directory, first, second = snapshot_shards
    third = directory / "snap_000.2.hdf5"
    with h5py.File(third, "w") as shard:
        shard["Masses"] = [31, 32, 33]
    files = (first, second, third)
    for path in files:
        with h5py.File(path, "a") as shard:
            shard.create_group("Header").attrs["NumFilesPerSnapshot"] = 3
    files[remove_index].unlink()

    with pytest.raises(FileNotFoundError, match="header declares 3 shards.*found 2"):
        load(directory, virtualcache=True)


@pytest.mark.integration
def test_ioload_refreshes_legacy_cache_after_declared_shards_arrive(snapshot_shards):
    directory, first, second = snapshot_shards
    with h5py.File(first, "a") as shard:
        shard.create_group("Header").attrs["NumFilesPerSnapshot"] = 2
    with h5py.File(second, "a") as shard:
        shard.create_group("Header").attrs["NumFilesPerSnapshot"] = 2
    cache_path = return_hdf5cachepath(directory, fileprefix="")
    create_mergedhdf5file(cache_path, [first], max_workers=1)
    with h5py.File(cache_path, "a") as cache:
        cache.attrs["_cachingcomplete"] = True
        del cache[SOURCE_SHARDS_PATH]
        del cache[VIRTUAL_SOURCES_PATH]

    fields, _, cache_file, _ = load(directory)
    with cache_file:
        np.testing.assert_array_equal(fields["Masses"].compute(), [1, 2, 3, 11, 12, 13])
