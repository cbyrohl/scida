"""Integration tests for catalog ID assignment (GroupID, SubhaloID)."""

import dask
import h5py
import numpy as np
import pytest

from scida import load


def _create_arepo_snapshot_and_catalog(tmp_path):
    """Create minimal Arepo-style snapshot + catalog files for testing."""
    npart = 100  # particles in PartType0
    ngroups = 3
    nsubs = 4

    # Group structure: groups own [40, 30, 20] particles, 10 unbound
    group_lentype = np.zeros((ngroups, 6), dtype=np.int64)
    group_lentype[:, 0] = [40, 30, 20]

    # Subhalo structure:
    # Group 0: subs 0,1 with 20 particles each
    # Group 1: sub 2 with 30 particles
    # Group 2: sub 3 with 20 particles
    sub_lentype = np.zeros((nsubs, 6), dtype=np.int64)
    sub_lentype[:, 0] = [20, 20, 30, 20]
    sub_grnr = np.array([0, 0, 1, 2], dtype=np.int64)
    group_firstsub = np.array([0, 2, 3], dtype=np.int64)
    group_nsubs = np.array([2, 1, 1], dtype=np.int64)

    # Write snapshot
    snap_path = tmp_path / "snap_000.hdf5"
    with h5py.File(snap_path, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["NumPart_ThisFile"] = [npart, 0, 0, 0, 0, 0]
        hdr.attrs["NumPart_Total"] = [npart, 0, 0, 0, 0, 0]
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
        pt0 = f.create_group("PartType0")
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
