"""Tests for cache versioning and dataset-local pre-cache lookup."""

import json
import os
import pathlib
import shutil

import h5py
import numpy as np
import pytest

from scida.helpers_misc import hash_path
from scida.interface import Dataset
from scida.io._base import ChunkedHDF5Loader, InvalidCacheError
from scida.misc import CACHE_FORMAT_VERSION, find_precached_file, return_hdf5cachepath
from tests.helpers import DummyGadgetSnapshotFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunked_snapshot(directory, nchunks=2):
    """Create a multi-chunk Gadget snapshot directory and return the dir path."""
    d = pathlib.Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    dummy = DummyGadgetSnapshotFile()
    dummy.header["NumFilesPerSnapshot"] = nchunks
    for i in range(nchunks):
        dummy.write(str(d / ("snap_%03d.%d.hdf5" % (0, i))))
    return str(d)


def _create_precache_hdf5(start_path, hsh, depth=1):
    """Place a pre-cache HDF5 file at the expected ancestor location.

    Returns the path to the created file.
    """
    p = pathlib.Path(start_path).resolve()
    for _ in range(depth):
        p = p.parent
    dest = p / "postprocessing" / "scida"
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest / ("%s.hdf5" % hsh)
    return str(fp)


def _create_precache_json(start_path, hsh, content, depth=1):
    """Place a pre-cache JSON file at the expected ancestor location.

    Returns the path to the created file.
    """
    p = pathlib.Path(start_path).resolve()
    for _ in range(depth):
        p = p.parent
    dest = p / "postprocessing" / "scida"
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest / ("%s.json" % hsh)
    with open(str(fp), "w") as f:
        json.dump(content, f)
    return str(fp)


# ===========================================================================
# Part A: Cache Versioning
# ===========================================================================


class TestCacheVersionHDF5:
    """HDF5 cache versioning tests."""

    def test_cache_version_written_to_hdf5(self, tmp_path, cachedir):
        """Verify _cache_format_version attribute is written when creating a cache."""
        snap_dir = _make_chunked_snapshot(tmp_path / "snap")
        loader = ChunkedHDF5Loader(snap_dir)
        loader.load(fileprefix="snap_000", chunksize="auto")

        cachefp = return_hdf5cachepath(snap_dir, fileprefix="snap_000")
        with h5py.File(cachefp, "r") as hf:
            assert "_cache_format_version" in hf.attrs
            assert hf.attrs["_cache_format_version"] == CACHE_FORMAT_VERSION

    def test_cache_version_mismatch_rejected(self, tmp_path, cachedir):
        """A cache with the wrong version must raise InvalidCacheError."""
        snap_dir = _make_chunked_snapshot(tmp_path / "snap")
        loader = ChunkedHDF5Loader(snap_dir)
        loader.load(fileprefix="snap_000", chunksize="auto")
        # Close the file handle from the first load
        if loader.file is not None:
            loader.file.close()

        cachefp = return_hdf5cachepath(snap_dir, fileprefix="snap_000")
        # Tamper with the version
        with h5py.File(cachefp, "r+") as hf:
            hf.attrs["_cache_format_version"] = CACHE_FORMAT_VERSION + 99

        loader2 = ChunkedHDF5Loader(snap_dir)
        with pytest.raises(InvalidCacheError, match="version mismatch"):
            loader2.load_cachefile(cachefp)

    def test_cache_missing_version_rejected(self, tmp_path, cachedir):
        """A cache without _cache_format_version must raise InvalidCacheError."""
        snap_dir = _make_chunked_snapshot(tmp_path / "snap")
        loader = ChunkedHDF5Loader(snap_dir)
        loader.load(fileprefix="snap_000", chunksize="auto")
        # Close the file handle from the first load
        if loader.file is not None:
            loader.file.close()

        cachefp = return_hdf5cachepath(snap_dir, fileprefix="snap_000")
        # Remove the version attribute
        with h5py.File(cachefp, "r+") as hf:
            del hf.attrs["_cache_format_version"]

        loader2 = ChunkedHDF5Loader(snap_dir)
        with pytest.raises(InvalidCacheError, match="version mismatch"):
            loader2.load_cachefile(cachefp)


class TestCacheVersionJSON:
    """JSON (series) cache versioning tests."""

    def test_series_cache_version_in_json(self, tmp_path, cachedir):
        """Verify _cache_format_version is written into series JSON cache."""
        from scida.series import DatasetSeries

        snap_dirs = []
        for i in range(2):
            d = tmp_path / ("snap_%d" % i)
            _make_chunked_snapshot(d, nchunks=1)
            snap_dirs.append(d)
        # Use the snapshot HDF5 files directly (single-file, not chunked)
        paths = [list(d.glob("*.hdf5"))[0] for d in snap_dirs]

        series = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        # metadata was created
        fp = series._metadatafile
        assert fp is not None
        assert os.path.exists(fp)
        with open(fp, "r") as f:
            raw = json.load(f)
        assert "_cache_format_version" in raw
        assert raw["_cache_format_version"] == CACHE_FORMAT_VERSION

    def test_series_cache_version_mismatch(self, tmp_path, cachedir):
        """Stale JSON cache with wrong version triggers re-creation."""
        from scida.series import DatasetSeries

        snap_dirs = []
        for i in range(2):
            d = tmp_path / ("snap_%d" % i)
            _make_chunked_snapshot(d, nchunks=1)
            snap_dirs.append(d)
        paths = [list(d.glob("*.hdf5"))[0] for d in snap_dirs]

        series = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        fp = series._metadatafile

        # Tamper with version
        with open(fp, "r") as f:
            raw = json.load(f)
        raw["_cache_format_version"] = CACHE_FORMAT_VERSION + 99
        with open(fp, "w") as f:
            json.dump(raw, f)

        # Re-load: the getter should return None (triggering re-creation)
        series2 = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        # The metadata was re-created (old file overwritten or new one)
        assert series2.metadata is not None
        with open(series2._metadatafile, "r") as f:
            raw2 = json.load(f)
        assert raw2["_cache_format_version"] == CACHE_FORMAT_VERSION


# ===========================================================================
# Part B: Pre-Cache Lookup
# ===========================================================================


class TestFindPrecachedFile:
    """Unit tests for find_precached_file()."""

    def test_found(self, tmp_path):
        """Pre-cache file found at parent/postprocessing/scida/."""
        start = tmp_path / "sim" / "output" / "snapdir_000"
        start.mkdir(parents=True)
        hsh = "abc123"
        dest = tmp_path / "sim" / "postprocessing" / "scida"
        dest.mkdir(parents=True)
        fp = dest / ("%s.hdf5" % hsh)
        fp.touch()

        result = find_precached_file(hsh, "hdf5", str(start))
        assert result == str(fp)

    def test_not_found(self, tmp_path):
        """Returns None when no pre-cache exists."""
        start = tmp_path / "sim" / "output" / "snapdir_000"
        start.mkdir(parents=True)
        result = find_precached_file("nonexistent", "hdf5", str(start))
        assert result is None

    def test_depth_limit(self, tmp_path):
        """Pre-cache beyond max_depth is not found."""
        deep = tmp_path
        for i in range(8):
            deep = deep / ("level%d" % i)
        deep.mkdir(parents=True)

        dest = tmp_path / "postprocessing" / "scida"
        dest.mkdir(parents=True)
        fp = dest / "somehash.hdf5"
        fp.touch()

        # default max_depth=5 should NOT reach it (8 levels deep)
        result = find_precached_file("somehash", "hdf5", str(deep), max_depth=5)
        assert result is None

        # with enough depth it should find it
        result = find_precached_file("somehash", "hdf5", str(deep), max_depth=10)
        assert result == str(fp)


class TestChunkedLoaderPrecache:
    """Integration tests for ChunkedHDF5Loader pre-cache behavior."""

    def test_chunked_loader_uses_precache(self, tmp_path, cachedir):
        """When user cache is absent, loader should use a valid pre-cache."""
        snap_dir = tmp_path / "sim" / "output" / "snapdir_000"
        _make_chunked_snapshot(snap_dir, nchunks=2)

        # First, create a valid cache to use as the pre-cache source
        loader = ChunkedHDF5Loader(str(snap_dir))
        loader.load(fileprefix="snap_000", chunksize="auto")
        user_cachefp = return_hdf5cachepath(str(snap_dir), fileprefix="snap_000")

        # Place it as a pre-cache
        hsh = hash_path(os.path.join(str(snap_dir), "snap_000"))
        precache_fp = _create_precache_hdf5(str(snap_dir), hsh, depth=1)
        shutil.copy2(user_cachefp, precache_fp)

        # Remove the user cache
        os.remove(user_cachefp)

        # Now load again — should use pre-cache
        loader2 = ChunkedHDF5Loader(str(snap_dir))
        data, metadata = loader2.load(fileprefix="snap_000", chunksize="auto")
        assert data is not None
        # The user cache should NOT have been recreated (pre-cache was used)
        # Actually, the current flow creates user cache when cachefp doesn't exist
        # and pre-cache fails. Since pre-cache succeeds, it returns early.
        assert not os.path.isfile(user_cachefp)

    def test_chunked_loader_prefers_user_cache(self, tmp_path, cachedir):
        """When user cache exists, pre-cache is ignored."""
        snap_dir = tmp_path / "sim" / "output" / "snapdir_000"
        _make_chunked_snapshot(snap_dir, nchunks=2)

        loader = ChunkedHDF5Loader(str(snap_dir))
        loader.load(fileprefix="snap_000", chunksize="auto")
        user_cachefp = return_hdf5cachepath(str(snap_dir), fileprefix="snap_000")
        assert os.path.isfile(user_cachefp)

        # Place a pre-cache (it won't be used since user cache exists)
        hsh = hash_path(os.path.join(str(snap_dir), "snap_000"))
        precache_fp = _create_precache_hdf5(str(snap_dir), hsh, depth=1)
        # Write an INVALID file as pre-cache to prove it's not read
        with h5py.File(precache_fp, "w") as hf:
            hf.attrs["_cachingcomplete"] = False

        # Should succeed (uses user cache, not pre-cache)
        loader2 = ChunkedHDF5Loader(str(snap_dir))
        data, metadata = loader2.load(fileprefix="snap_000", chunksize="auto")
        assert data is not None

    def test_chunked_loader_fallthrough_on_invalid(self, tmp_path, cachedir):
        """Invalid pre-cache causes fallback to creating user cache."""
        snap_dir = tmp_path / "sim" / "output" / "snapdir_000"
        _make_chunked_snapshot(snap_dir, nchunks=2)

        # Place an invalid pre-cache
        hsh = hash_path(os.path.join(str(snap_dir), "snap_000"))
        precache_fp = _create_precache_hdf5(str(snap_dir), hsh, depth=1)
        with h5py.File(precache_fp, "w") as hf:
            hf.attrs["_cachingcomplete"] = True
            hf.attrs["_cache_format_version"] = CACHE_FORMAT_VERSION + 99

        # Load — should fall through to creating user cache
        loader = ChunkedHDF5Loader(str(snap_dir))
        data, metadata = loader.load(fileprefix="snap_000", chunksize="auto")
        assert data is not None
        # User cache should have been created
        user_cachefp = return_hdf5cachepath(str(snap_dir), fileprefix="snap_000")
        assert os.path.isfile(user_cachefp)

    def test_overwrite_cache_skips_precache(self, tmp_path, cachedir):
        """overwrite_cache=True should ignore pre-cache and create fresh user cache."""
        snap_dir = tmp_path / "sim" / "output" / "snapdir_000"
        _make_chunked_snapshot(snap_dir, nchunks=2)

        # Create a valid pre-cache
        loader = ChunkedHDF5Loader(str(snap_dir))
        loader.load(fileprefix="snap_000", chunksize="auto")
        user_cachefp = return_hdf5cachepath(str(snap_dir), fileprefix="snap_000")
        hsh = hash_path(os.path.join(str(snap_dir), "snap_000"))
        precache_fp = _create_precache_hdf5(str(snap_dir), hsh, depth=1)
        shutil.copy2(user_cachefp, precache_fp)
        os.remove(user_cachefp)

        # Load with overwrite_cache — should NOT use pre-cache
        loader2 = ChunkedHDF5Loader(str(snap_dir))
        data, metadata = loader2.load(
            fileprefix="snap_000", chunksize="auto", overwrite_cache=True
        )
        assert data is not None
        # User cache was freshly created
        assert os.path.isfile(user_cachefp)


class TestSeriesPrecache:
    """Pre-cache tests for DatasetSeries."""

    def test_series_uses_precached_json(self, tmp_path, cachedir):
        """Series should use pre-cached JSON when user cache is absent."""
        from scida.series import DatasetSeries

        snap_dirs = []
        for i in range(2):
            d = tmp_path / "sim" / "output" / ("snap_%d" % i)
            _make_chunked_snapshot(d, nchunks=1)
            snap_dirs.append(d)
        paths = [list(d.glob("*.hdf5"))[0] for d in snap_dirs]

        # Create a series to generate the cache
        series = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        user_fp = series._metadatafile
        assert os.path.exists(user_fp)

        # Read the cache content and place it as pre-cache
        with open(user_fp, "r") as f:
            cache_content = json.load(f)

        hsh = hash_path("".join([str(p) for p in paths]))
        precache_fp = _create_precache_json(
            str(paths[0]), hsh, cache_content, depth=3
        )

        # Remove user cache
        os.remove(user_fp)
        # Also remove parent dir if empty
        user_dir = os.path.dirname(user_fp)
        if os.path.isdir(user_dir) and not os.listdir(user_dir):
            os.rmdir(user_dir)

        # Re-create series — should pick up pre-cache
        series2 = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        assert series2.metadata is not None
        assert series2._metadatafile == precache_fp

    def test_series_precache_never_deleted_on_version_mismatch(
        self, tmp_path, cachedir
    ):
        """A pre-cache with wrong version must NOT be deleted."""
        from scida.series import DatasetSeries

        snap_dirs = []
        for i in range(2):
            d = tmp_path / "sim" / "output" / ("snap_%d" % i)
            _make_chunked_snapshot(d, nchunks=1)
            snap_dirs.append(d)
        paths = [list(d.glob("*.hdf5"))[0] for d in snap_dirs]

        # Create a series to generate valid cache content
        series = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        user_fp = series._metadatafile
        with open(user_fp, "r") as f:
            cache_content = json.load(f)

        # Tamper version and place as pre-cache
        cache_content["_cache_format_version"] = CACHE_FORMAT_VERSION + 99
        hsh = hash_path("".join([str(p) for p in paths]))
        precache_fp = _create_precache_json(
            str(paths[0]), hsh, cache_content, depth=3
        )

        # Remove user cache so __init__ picks up the pre-cache
        os.remove(user_fp)
        user_dir = os.path.dirname(user_fp)
        if os.path.isdir(user_dir) and not os.listdir(user_dir):
            os.rmdir(user_dir)

        # Re-create series — version mismatch on pre-cache
        series2 = DatasetSeries(paths, datasetclass=Dataset, lazy=True)
        assert series2.metadata is not None
        # The pre-cache must still exist (never deleted)
        assert os.path.isfile(precache_fp)
        # The new metadata was written to the user cache, not the pre-cache
        assert series2._metadatafile != precache_fp
