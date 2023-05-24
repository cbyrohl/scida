import logging
import os
import re
from typing import Union

import numpy as np

from astrodask.interface import Dataset
from astrodask.io import load_metadata

log = logging.getLogger(__name__)


class GadgetStyleSnapshot(Dataset):
    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        """We define gadget-style snapshots as nbody/hydrodynamical simulation snapshots that follow
        the common /PartType0, /PartType1 grouping scheme."""
        self.boxsize = np.full(3, np.nan)
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

        defaultattributes = ["config", "header", "parameters"]
        for k in self._metadata_raw:
            name = k.strip("/").lower()
            if name in defaultattributes:
                self.__dict__[name] = self._metadata_raw[k]
                if "BoxSize" in self.__dict__[name]:
                    self.boxsize = self.__dict__[name]["BoxSize"]
                elif "Boxsize" in self.__dict__[name]:
                    self.boxsize = self.__dict__[name]["Boxsize"]

    @classmethod
    def _get_fileprefix(cls, path: Union[str, os.PathLike], **kwargs) -> str:
        """
        Get the fileprefix used to identify files belonging to given dataset.
        Parameters
        ----------
        path: str, os.PathLike
            path to check
        kwargs

        Returns
        -------
        str
        """
        if os.path.isfile(path):
            return ""  # nothing to do, we have a single file, not a directory
        # order matters: groups will be taken before fof_subhalo, requires py>3.7 for dict order
        prfxs = ["groups", "fof_subhalo", "snap"]
        prfxs_prfx_sim = dict.fromkeys(prfxs)
        files = sorted(os.listdir(path))
        prfxs_lst = []
        for fn in files:
            s = re.search(r"^(\w*)_(\d*)", fn)
            if s is not None:
                prfxs_lst.append(s.group(1))
        prfxs_lst = [p for s in prfxs_prfx_sim for p in prfxs_lst if p.startswith(s)]
        prfxs = dict.fromkeys(prfxs_lst)
        prfxs = list(prfxs.keys())
        if len(prfxs) > 1:
            log.debug("We have more than one prefix avail: %s" % prfxs)
        elif len(prfxs) == 0:
            return ""
        if set(prfxs) == {"groups", "fof_subhalo_tab"}:
            return "groups"  # "groups" over "fof_subhalo_tab"
        return prfxs[0]

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, expect_grp=False, **kwargs
    ) -> bool:
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
        if os.path.isdir(path):
            files = os.listdir(path)
            sufxs = [f.split(".")[-1] for f in files]
            if not iszarr and len(set(sufxs)) > 1:
                possibly_valid = False
            if sufxs[0] == "hdf5":
                possibly_valid = True
        if possibly_valid:
            metadata_raw = load_metadata(path, **kwargs)
            # need some silly combination of attributes to be sure
            if all([k in metadata_raw for k in ["/Header"]]):
                # identifying snapshot or group catalog
                is_snap = all(
                    [
                        k in metadata_raw["/Header"]
                        for k in ["NumPart_ThisFile", "NumPart_Total"]
                    ]
                )
                is_grp = all(
                    [
                        k in metadata_raw["/Header"]
                        for k in ["Ngroups_ThisFile", "Ngroups_Total"]
                    ]
                )
                if is_grp:
                    return True
                if is_snap and not expect_grp:
                    return True
        return False

    def register_field(self, parttype, name=None, description=""):
        res = self.data.register_field(parttype, name=name, description=description)
        return res


class SwiftSnapshot(GadgetStyleSnapshot):
    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

    @classmethod
    def validate_path(cls, path: Union[str, os.PathLike], *args, **kwargs) -> bool:
        valid = super().validate_path(path, *args, **kwargs)
        if not valid:
            return False
        metadata_raw = load_metadata(path, **kwargs)
        comparestr = metadata_raw.get("/Code", {}).get("Code", b"").decode()
        valid = "SWIFT" in comparestr
        return valid


class GizmoSnapshot(GadgetStyleSnapshot):
    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

    @classmethod
    def validate_path(cls, path: Union[str, os.PathLike], *args, **kwargs) -> bool:
        valid = super().validate_path(path, *args, **kwargs)
        if not valid:
            return False
        # there does not seem to be identifying metadata for Gizmo (? see SIMBA snap),
        # so we just return False for now (i.e. falling back onto GadgetStyleSnapshot on auto-detection)
        # TODO: split routine into "validate" and "identify"? One accessing whether we *could* read the data,
        # the other checking whether we *should* primarily read the data this way?
        return False


# right now ArepoSnapshot is defined in separate file
