import os

from astrodask.interfaces.arepo import ArepoSnapshot


class IllustrisSnapshot(ArepoSnapshot):
    def __init__(self, path, chunksize="auto", catalog=None, **kwargs):
        prfx = self._get_fileprefix(path)
        prfx = kwargs.pop("fileprefix", prfx)
        super().__init__(
            path,
            chunksize=chunksize,
            catalog=catalog,
            fileprefix=prfx,
            catalog_kwargs=dict(fileprefix="groups"),
            **kwargs
        )

    @classmethod
    def _get_fileprefix(cls, path) -> str:
        # order matters: groups will be taken before fof_subhalo, requires py>3.7 for dict order
        prfxs_prfx_sim = dict.fromkeys(["groups", "fof_subhalo", "snap"])
        files = os.listdir(path)
        prfxs = [f.split(".")[0] for f in files]
        prfxs = [p for s in prfxs_prfx_sim for p in prfxs if p.startswith(s)]
        prfxs = dict.fromkeys(prfxs)
        if len(prfxs) > 1:
            print(
                "Note: We have more than one prefix avail:", prfxs
            )  # TODO: Remove verbosity eventually.
        prfx = list(prfxs.keys())[0]
        return prfx

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        if path.endswith(".hdf5") or path.endswith(".zarr"):
            return True
        if os.path.isdir(path):
            files = os.listdir(path)
            cls._get_fileprefix(path)
            sufxs = [f.split(".")[-1] for f in files]
            if len(set(sufxs)) > 1:
                return False
            if sufxs[0] in ["hdf5", "zarr"]:
                return True
        return False
