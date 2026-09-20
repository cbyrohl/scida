"""Integration tests for catalog ID assignment (GroupID, SubhaloID)."""

import os

import dask
import h5py
import numpy as np
import pytest

from scida import load
from scida.config import get_config
from scida.customs.arepo.dataset import ArepoCatalog, ArepoSnapshot
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot
from scida.discovertypes import CandidateStatus


def _create_arepo_snapshot_and_catalog(tmp_path, parttype=0):
    """Create minimal Arepo-style snapshot + catalog files for testing."""
    npart = 100
    counts = np.zeros(6, dtype=np.int64)
    counts[parttype] = npart
    ngroups = 3
    nsubs = 4

    # Group structure: groups own [40, 30, 20] particles, 10 unbound
    group_lentype = np.zeros((ngroups, 6), dtype=np.int64)
    group_lentype[:, parttype] = [40, 30, 20]

    # Subhalo structure:
    # Group 0: subs 0,1 with 20 particles each
    # Group 1: sub 2 with 30 particles
    # Group 2: sub 3 with 20 particles
    sub_lentype = np.zeros((nsubs, 6), dtype=np.int64)
    sub_lentype[:, parttype] = [20, 20, 30, 20]
    sub_grnr = np.array([0, 0, 1, 2], dtype=np.int64)
    group_firstsub = np.array([0, 2, 3], dtype=np.int64)
    group_nsubs = np.array([2, 1, 1], dtype=np.int64)

    # Write snapshot
    snap_path = tmp_path / "snap_000.hdf5"
    with h5py.File(snap_path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["NumPart_ThisFile"] = counts
        hdr.attrs["NumPart_Total"] = counts
        hdr.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        hdr.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hdr.attrs["Time"] = 1.0
        hdr.attrs["Redshift"] = 0.0
        hdr.attrs["BoxSize"] = 100.0
        hdr.attrs["NumFilesPerSnapshot"] = 1
        hdr.attrs["Omega0"] = 0.3
        hdr.attrs["OmegaBaryon"] = 0.04
        hdr.attrs["OmegaLambda"] = 0.7
        hdr.attrs["HubbleParam"] = 0.7
        hdr.attrs["Flag_Sfr"] = 0
        hdr.attrs["Flag_Cooling"] = 0
        hdr.attrs["Flag_StellarAge"] = 0
        hdr.attrs["Flag_Metals"] = 0
        hdr.attrs["Flag_Feedback"] = 0
        hdr.attrs["Flag_DoublePrecision"] = 0
        hdr.attrs["Flag_IC_Info"] = 0
        hdr.attrs["Flag_LptInitCond"] = 0
        hdr.attrs["Git_commit"] = b"dummy"
        pt0 = f.create_group(f"PartType{parttype}")
        pt0.create_dataset("Coordinates", data=np.zeros((npart, 3), dtype=np.float64))
        pt0.create_dataset("ParticleIDs", data=np.arange(npart, dtype=np.int64))
        pt0.create_dataset("Velocities", data=np.zeros((npart, 3), dtype=np.float64))
        pt0.create_dataset("Masses", data=np.ones(npart, dtype=np.float64))

    # Write catalog
    cat_path = tmp_path / "groups_000.hdf5"
    with h5py.File(cat_path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["Ngroups_ThisFile"] = ngroups
        hdr.attrs["Ngroups_Total"] = ngroups
        hdr.attrs["Nsubgroups_ThisFile"] = nsubs
        hdr.attrs["Nsubgroups_Total"] = nsubs
        hdr.attrs["Time"] = 1.0
        hdr.attrs["Redshift"] = 0.0
        hdr.attrs["BoxSize"] = 100.0
        hdr.attrs["NumFilesPerSnapshot"] = 1
        hdr.attrs["Omega0"] = 0.3
        hdr.attrs["OmegaBaryon"] = 0.04
        hdr.attrs["OmegaLambda"] = 0.7
        hdr.attrs["HubbleParam"] = 0.7
        hdr.attrs["Git_commit"] = b"dummy"
        grp = f.create_group("Group")
        grp.create_dataset("GroupLenType", data=group_lentype)
        grp.create_dataset("GroupLen", data=group_lentype.sum(axis=1))
        grp.create_dataset("GroupFirstSub", data=group_firstsub)
        grp.create_dataset("GroupNsubs", data=group_nsubs)
        grp.create_dataset("GroupPos", data=np.zeros((ngroups, 3), dtype=np.float64))
        grp.create_dataset("GroupMass", data=np.zeros(ngroups, dtype=np.float64))
        sh = f.create_group("Subhalo")
        sh.create_dataset("SubhaloLenType", data=sub_lentype)
        sh.create_dataset("SubhaloLen", data=sub_lentype.sum(axis=1))
        sh.create_dataset("SubhaloGrNr", data=sub_grnr)
        sh.create_dataset("SubhaloPos", data=np.zeros((nsubs, 3), dtype=np.float64))
        sh.create_dataset("SubhaloMass", data=np.zeros(nsubs, dtype=np.float64))

    return snap_path, cat_path


@pytest.mark.integration
def test_subhaloid_small_chunksize(tmp_path):
    """Test that SubhaloID computes correctly with small dask chunk sizes (issue #57)."""
    snap_path, cat_path = _create_arepo_snapshot_and_catalog(tmp_path)

    # Use a tiny chunk size so the per-group arrays (e.g. GroupFirstSub,
    # 3 int64 = 24 bytes) get split into multiple chunks, triggering the
    # chunk mismatch with per-particle arrays in map_blocks.
    with dask.config.set({"array.chunk-size": "16B"}):
        snap = load(snap_path, catalog=cat_path, units=False)

    # This would raise "ValueError: Shapes do not align" before the fix
    subhalo_ids = snap.data["PartType0"]["SubhaloID"].compute()
    group_ids = snap.data["PartType0"]["GroupID"].compute()

    maxint = np.iinfo(np.int64).max

    # Verify SubhaloID values
    assert np.all(subhalo_ids[:20] == 0)
    assert np.all(subhalo_ids[20:40] == 1)
    assert np.all(subhalo_ids[40:70] == 2)
    assert np.all(subhalo_ids[70:90] == 3)
    assert np.all(subhalo_ids[90:] == maxint)  # unbound

    # Verify GroupID values
    assert np.all(group_ids[:40] == 0)
    assert np.all(group_ids[40:70] == 1)
    assert np.all(group_ids[70:90] == 2)
    assert np.all(group_ids[90:] == maxint)  # unbound


@pytest.mark.integration
@pytest.mark.parametrize("first_entry", ["snapshot", "hidden", "backup", "directory"])
def test_catalog_ids_ignore_directory_listing_order(tmp_path, monkeypatch, first_entry):
    """Temporary entries must not suppress Arepo catalog attachment. [AI-Codex]"""
    monkeypatch.setitem(get_config(), "nthreads", 1)
    snap_path, cat_path = _create_arepo_snapshot_and_catalog(tmp_path, parttype=4)
    snapdir = tmp_path / "snapdir_000"
    catdir = tmp_path / "groups_000"
    snapdir.mkdir()
    catdir.mkdir()
    names = {
        "snapshot": "snap_000.0.hdf5",
        "hidden": ".snap_000.0.hdf5.PARTIAL",
        "backup": "snap_000.0.hdf5.bak",
        "directory": "notes",
    }
    snap_path.rename(snapdir / names["snapshot"])
    cat_path.rename(catdir / "fof_subhalo_tab_000.0.hdf5")
    # Use multipart headers, including an empty trailing chunk, like real outputs.
    for directory, prefix, count_keys in (
        (snapdir, "snap_000", ("NumPart_ThisFile",)),
        (catdir, "fof_subhalo_tab_000", ("Ngroups_ThisFile", "Nsubgroups_ThisFile")),
    ):
        with h5py.File(directory / f"{prefix}.0.hdf5", "r+") as first:
            first["Header"].attrs["NumFilesPerSnapshot"] = 2
            with h5py.File(directory / f"{prefix}.1.hdf5", "w") as second:
                header = second.create_group("Header")
                header.attrs.update(first["Header"].attrs)
                for key in count_keys:
                    header.attrs[key] = np.zeros_like(header.attrs[key])
    (snapdir / names["hidden"]).write_text("incomplete rsync transfer")
    (snapdir / names["backup"]).write_text("backup")
    (snapdir / names["directory"]).mkdir()
    listdir = os.listdir

    def reordered_listdir(path):
        entries = listdir(path)
        if os.fspath(path) == str(snapdir):
            first = names[first_entry]
            return [first] + [entry for entry in entries if entry != first]
        return entries

    monkeypatch.setattr(os, "listdir", reordered_listdir)
    # Match the production catalogue-first loading order.
    catalog = load(catdir, fileprefix="fof_subhalo_tab_000", units=False)
    assert isinstance(catalog, ArepoCatalog)
    snap = load(snapdir, catalog=catdir, fileprefix="snap_000", units=False)
    assert isinstance(snap, ArepoSnapshot)
    assert isinstance(snap.catalog, ArepoCatalog)
    stars = snap.data["PartType4"]
    np.testing.assert_array_equal(stars["ParticleIDs"].compute(), np.arange(100))
    expected = np.repeat([0, 1, 2, np.iinfo(np.int64).max], [40, 30, 20, 10])
    np.testing.assert_array_equal(stars["GroupID"].compute(), expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "entry", [None, ".snap_000.0.hdf5", "snap_000.0.hdf5.bak", "directory"]
)
def test_directory_without_snapshot_chunks_is_rejected(tmp_path, entry):
    """Empty or ignored-only directories are not snapshots. [AI-Codex]"""
    if entry == "directory":
        (tmp_path / "snap_000.0.hdf5").mkdir()
    elif entry is not None:
        (tmp_path / entry).touch()
    assert GadgetStyleSnapshot.validate_path(tmp_path) == CandidateStatus.NO


@pytest.mark.integration
@pytest.mark.parametrize("artifact_chunk", [0, 1])
@pytest.mark.parametrize("fileprefix", ["", "snap_000"])
def test_directory_with_non_hdf5_chunk_is_rejected(
    tmp_path, artifact_chunk, fileprefix
):
    """Reject numbered transfer artifacts before or after valid chunks. [AI-Codex]"""
    snap_path, _ = _create_arepo_snapshot_and_catalog(tmp_path)
    snapdir = tmp_path / "snapdir_000"
    snapdir.mkdir()
    snap_path.rename(snapdir / f"snap_000.{1 - artifact_chunk}.hdf5")
    assert (
        GadgetStyleSnapshot.validate_path(snapdir, fileprefix=fileprefix)
        == CandidateStatus.MAYBE
    )

    (snapdir / f"snap_000.{artifact_chunk}.PARTIAL").write_text("incomplete transfer")
    assert (
        GadgetStyleSnapshot.validate_path(snapdir, fileprefix=fileprefix)
        == CandidateStatus.NO
    )
