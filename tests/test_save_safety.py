import numpy as np
import pytest
import zarr

from scida import load
from scida.interface import BaseDataset


@pytest.mark.unit
class TestSaveSafetyCheck:
    """Test that save() refuses to overwrite non-zarr directories."""

    def _make_dataset(self):
        """Create a minimal BaseDataset with empty data for testing save()."""
        from scida.fields import FieldContainer

        ds = BaseDataset.__new__(BaseDataset)
        ds.data = FieldContainer()
        ds.header = {}
        ds.config = {}
        ds.parameters = {}
        return ds

    def test_refuses_nonempty_nonzarr_dir(self, tmp_path):
        notzarr = tmp_path / "mydir"
        notzarr.mkdir()
        (notzarr / "important.txt").write_text("data")

        ds = self._make_dataset()
        with pytest.raises(ValueError, match="is not a zarr group"):
            ds.save(str(notzarr))

    def test_allows_empty_dir(self, tmp_path):
        emptydir = tmp_path / "emptydir"
        emptydir.mkdir()

        ds = self._make_dataset()
        ds.save(str(emptydir))
        # Should succeed and create a zarr group marker
        assert (emptydir / ".zgroup").exists()

    def test_allows_existing_zarr_group(self, tmp_path):
        zarrdir = tmp_path / "zarrdir"
        zarrdir.mkdir()
        (zarrdir / ".zgroup").write_text("{}")

        ds = self._make_dataset()
        ds.save(str(zarrdir))
        assert (zarrdir / ".zgroup").exists()

    def test_allows_existing_zarr_array(self, tmp_path):
        zarrdir = tmp_path / "zarrarray"
        zarrdir.mkdir()
        (zarrdir / ".zarray").write_text("{}")

        ds = self._make_dataset()
        ds.save(str(zarrdir))

    def test_overwrite_false_skips_check(self, tmp_path):
        notzarr = tmp_path / "mydir"
        notzarr.mkdir()
        (notzarr / "important.txt").write_text("data")

        ds = self._make_dataset()
        # overwrite=False should not trigger the safety check
        # (it may fail for other zarr-related reasons, but not our safety check)
        try:
            ds.save(str(notzarr), overwrite=False)
        except ValueError as e:
            if "is not a zarr group" in str(e):
                pytest.fail("Safety check incorrectly triggered when overwrite=False")

    def test_nonexistent_path_proceeds(self, tmp_path):
        newdir = tmp_path / "newzarr"

        ds = self._make_dataset()
        ds.save(str(newdir))
        assert (newdir / ".zgroup").exists()

    def test_preserves_nonempty_nonzarr_dir(self, tmp_path):
        """Verify the directory contents are NOT deleted when save is rejected."""
        notzarr = tmp_path / "mydir"
        notzarr.mkdir()
        (notzarr / "important.txt").write_text("data")

        ds = self._make_dataset()
        with pytest.raises(ValueError, match="is not a zarr group"):
            ds.save(str(notzarr))

        # The original file must still exist
        assert (notzarr / "important.txt").exists()
        assert (notzarr / "important.txt").read_text() == "data"


@pytest.mark.integration
@pytest.mark.parametrize("nested_source", [False, True])
@pytest.mark.parametrize("use_symlink", [False, True])
def test_refuses_to_save_over_source_zarr_store(tmp_path, use_symlink, nested_source):
    store = tmp_path / "source.zarr"
    root = zarr.open_group(str(store), mode="w")
    source = root.create_group("nested") if nested_source else root
    source["a/b"] = [1, 2, 3]
    source["a/c"] = [4, 5, 6]
    dataset = load(store / source.path, units=False)

    destination = store
    if use_symlink:
        destination = tmp_path / "link.zarr"
        destination.symlink_to(store, target_is_directory=True)

    with pytest.raises(ValueError, match="source Zarr store"):
        dataset.save(destination)

    np.testing.assert_array_equal(source["a/b"][:], [1, 2, 3])
    np.testing.assert_array_equal(source["a/c"][:], [4, 5, 6])


@pytest.mark.integration
@pytest.mark.parametrize(
    ("source_name", "destination_name", "overwrite"),
    [("nested", "", False), ("data-extra", "data", True)],
    ids=["ancestor-without-overwrite", "sibling-with-shared-prefix"],
)
def test_save_preserves_source_when_destination_can_be_written(
    tmp_path, source_name, destination_name, overwrite
):
    store = tmp_path / "source.zarr"
    root = zarr.open_group(str(store), mode="w")
    source = root.create_group(source_name)
    source["a/c"] = [4, 5, 6]
    dataset = load(store / source_name, units=False)
    destination = store / destination_name

    dataset.save(destination, overwrite=overwrite)

    saved = zarr.open_group(str(destination), mode="r")
    np.testing.assert_array_equal(saved["a/c"][:], [4, 5, 6])
    np.testing.assert_array_equal(source["a/c"][:], [4, 5, 6])
