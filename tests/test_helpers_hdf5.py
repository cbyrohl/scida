import h5py
import numpy as np

from scida.helpers_hdf5 import create_mergedhdf5file


def test_create_mergedhdf5file_handles_datasets_missing_from_some_files(tmp_path):
    first = tmp_path / "snapshot.0.hdf5"
    second = tmp_path / "snapshot.1.hdf5"
    merged = tmp_path / "snapshot.hdf5"

    with h5py.File(first, "w") as h5f:
        parttype0 = h5f.create_group("PartType0")
        parttype0.create_dataset(
            "Coordinates",
            data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        )
        parttype0.create_dataset("Masses", data=np.array([10, 20], dtype=np.int64))

    with h5py.File(second, "w") as h5f:
        parttype0 = h5f.create_group("PartType0")
        parttype0.create_dataset(
            "Coordinates",
            data=np.array([[7, 8, 9]], dtype=np.float64),
        )
        parttype1 = h5f.create_group("PartType1")
        parttype1.create_dataset(
            "ParticleIDs",
            data=np.array([101, 102, 103], dtype=np.uint64),
        )

    create_mergedhdf5file(
        merged,
        [first, second],
        max_workers=1,
        virtual=False,
    )

    with h5py.File(merged, "r") as h5f:
        np.testing.assert_array_equal(
            h5f["PartType0/Coordinates"][:],
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            h5f["PartType0/Masses"][:],
            np.array([10, 20], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            h5f["PartType1/ParticleIDs"][:],
            np.array([101, 102, 103], dtype=np.uint64),
        )
