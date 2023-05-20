import os

from astrodask.interfaces.arepo import ArepoSnapshot


class IllustrisSnapshot(ArepoSnapshot):
    _fileprefix_catalog = "groups"

    def __init__(self, path, chunksize="auto", catalog=None, **kwargs):
        super().__init__(
            path,
            chunksize=chunksize,
            catalog=catalog,
            catalog_kwargs=dict(fileprefix=self._fileprefix_catalog),
            **kwargs
        )

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        kwargs["fileprefix_catalog"] = cls._fileprefix_catalog
        valid = super().validate_path(path, *args, **kwargs)
        if not valid:
            return False
        if os.path.isdir(path):
            files = os.listdir(path)
            files = [f for f in files if f.startswith(cls._fileprefix_catalog)]
            if len(files) == 0:
                return False
        return True
