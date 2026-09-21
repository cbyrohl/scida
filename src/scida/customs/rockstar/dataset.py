"""
Defines the RockstarCatalog class.
"""

from __future__ import annotations

import os

from scida.interface import Dataset
from scida.interfaces.mixins import CosmologyMixin
from scida.io import load_metadata_all


class RockstarCatalog(CosmologyMixin, Dataset):
    """Rockstar catalog dataset."""

    _unitfile = "units/rockstar.yaml"

    def __init__(self, path, **kwargs) -> None:
        """
        Initialize a RockstarCatalog object.

        Parameters
        ----------
        path: str
            Path to the catalog.
        kwargs: dict
            Additional keyword arguments.
        """
        super().__init__(path, **kwargs)

    @classmethod
    def validate_path(cls, path: str | os.PathLike, *args, **kwargs) -> bool:
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

            # attrs = metadata_raw_all["attrs"]

            # usually rockstar has some cosmology parameters attached
            datasets = metadata_raw_all["datasets"]
            dsnames = [d[0] for d in datasets]
            if "/cosmology:omega_dm" not in dsnames:
                return False

            return True
        return False
