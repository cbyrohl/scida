import pytest

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
        # Should succeed and create a zarr group marker (v3 uses zarr.json)
        assert (emptydir / "zarr.json").exists() or (emptydir / ".zgroup").exists()

    def test_allows_existing_zarr_group(self, tmp_path):
        zarrdir = tmp_path / "zarrdir"
        zarrdir.mkdir()
        (zarrdir / ".zgroup").write_text("{}")

        ds = self._make_dataset()
        ds.save(str(zarrdir))
        assert (zarrdir / "zarr.json").exists() or (zarrdir / ".zgroup").exists()

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
        assert (newdir / "zarr.json").exists() or (newdir / ".zgroup").exists()

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
