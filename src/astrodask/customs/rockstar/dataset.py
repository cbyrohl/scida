import os
from typing import Union

from astrodask.interface import Dataset
from astrodask.io import load_metadata_all


class RockstarCatalog(Dataset):
    def __init__(self, path, **kwargs) -> None:
        super().__init__(path, **kwargs)

    @classmethod
    def validate_path(cls, path: Union[str, os.PathLike], *args, **kwargs) -> bool:
        """
        Check if path is valid for this interface.
        Parameters
        ----------
        path: str, os.PathLike
            path to check
        args
        kwargs

        Returns
        -------
        bool
        """
        path = str(path)
        possibly_valid = False
        iszarr = path.rstrip("/").endswith(".zarr")
        if path.endswith(".hdf5") or iszarr:
            possibly_valid = True

        if possibly_valid:
            metadata_raw_all = load_metadata_all(path, **kwargs)

            attrs = metadata_raw_all["attrs"]

            # lets use the "feature" that rockstar adds no attributes (?) but uses scalar datasets
            if len(attrs) != 0:
                return False

            # usually rockstar has some cosmology parameters attached
            datasets = metadata_raw_all["datasets"]
            dsnames = [d[0] for d in datasets]
            if "/cosmology:omega_dm" not in dsnames:
                return False

            return True
        return False
