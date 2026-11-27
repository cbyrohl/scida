import io
import tarfile

import pytest

from scida.convenience import download_and_extract


@pytest.mark.integration
@pytest.mark.parametrize("entry_type", ["regular", "unsafe-path", "link"])
def test_download_archive_extracts_only_safe_entries(tmp_path, monkeypatch, entry_type):
    payload = b"sample data"

    def download(url, path, **kwargs):
        with tarfile.open(path, "w:gz") as archive:
            directory = tarfile.TarInfo("dataset")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)

            name = "../outside.txt" if entry_type == "unsafe-path" else "dataset/data"
            member = tarfile.TarInfo(name)
            if entry_type == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "other"
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

    monkeypatch.setattr("scida.convenience._download", download)
    destination = tmp_path / "download"
    destination.mkdir()
    downloaded = destination / "archive.tar.gz"
    url = "https://example.invalid/archive.tar.gz"

    if entry_type == "regular":
        assert download_and_extract(url, downloaded) == str(destination / "dataset")
        assert (destination / "dataset/data").read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="archive (path|entry)"):
            download_and_extract(url, downloaded)
        assert not (destination / "dataset").exists()
        assert not (tmp_path / "outside.txt").exists()

    assert not downloaded.exists()
