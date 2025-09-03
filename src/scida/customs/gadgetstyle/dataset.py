"""
Defines the GadgetStyleSnapshot class, mostly used for deriving subclasses for related codes/simulations.
"""

import logging
import os
import re
from typing import Optional, Union

import numpy as np

from scida.discovertypes import CandidateStatus
from scida.interface import Dataset
from scida.interfaces.mixins import CosmologyMixin
from scida.io import load_metadata
from scida.misc import get_scalar

log = logging.getLogger(__name__)


class GadgetStyleSnapshot(Dataset):
    """A dataset representing a Gadget-style snapshot."""

    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        """We define gadget-style snapshots as nbody/hydrodynamical simulation snapshots that follow
        the common /PartType0, /PartType1 grouping scheme."""
        self.boxsize = np.nan
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

        sanity_check = kwargs.get("sanity_check", False)
        key_nparts = "NumPart_Total"
        key_nparts_hw = "NumPart_Total_HighWord"
        if sanity_check and key_nparts in self.header and key_nparts_hw in self.header:
            nparts = self.header[key_nparts_hw] * 2**32 + self.header[key_nparts]
            for i, n in enumerate(nparts):
                pkey = "PartType%i" % i
                if pkey in self.data:
                    pdata = self.data[pkey]
                    fkey = next(iter(pdata.keys()))
                    nparts_loaded = pdata[fkey].shape[0]
                    if nparts_loaded != n:
                        raise ValueError(
                            "Number of particles in header (%i) does not match number of particles loaded (%i) "
                            "for particle type %i" % (n, nparts_loaded, i)
                        )

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
    def validate_path(cls, path: Union[str, os.PathLike], *args, expect_grp=False, **kwargs) -> CandidateStatus:
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
        possibly_valid = CandidateStatus.NO
        iszarr = path.rstrip("/").endswith(".zarr")
        if path.endswith(".hdf5") or iszarr:
            possibly_valid = CandidateStatus.MAYBE
        if os.path.isdir(path):
            files = os.listdir(path)
            sufxs = [f.split(".")[-1] for f in files]
            if not iszarr and len(set(sufxs)) > 1:
                possibly_valid = CandidateStatus.NO
            if sufxs[0] == "hdf5":
                possibly_valid = CandidateStatus.MAYBE
        if possibly_valid != CandidateStatus.NO:
            metadata_raw = load_metadata(path, **kwargs)
            # need some silly combination of attributes to be sure
            if all([k in metadata_raw for k in ["/Header"]]):
                # tng-project.org api snapshots do not have "NumPart_Total" in header
                is_zoom = "CutoutID" in metadata_raw["/Header"]
                headerattrs = ["NumPart_ThisFile"]
                if not is_zoom:
                    headerattrs.append("NumPart_Total")

                # identifying snapshot or group catalog
                is_snap = all([k in metadata_raw["/Header"] for k in headerattrs])
                is_grp = all([k in metadata_raw["/Header"] for k in ["Ngroups_ThisFile", "Ngroups_Total"]])
                if is_grp:
                    return CandidateStatus.MAYBE
                if is_snap and not expect_grp:
                    return CandidateStatus.MAYBE
        return CandidateStatus.NO

    def register_field(self, parttype, name=None, description=""):
        """
        Register a field for a given particle type by returning through decorator.

        Parameters
        ----------
        parttype: Optional[Union[str, List[str]]]
            Particle type name to register with. If None, register for the base field container.
        name: Optional[str]
            Name of the field to register.
        description: Optional[str]
            Description of the field to register.

        Returns
        -------
        callable

        """
        res = self.data.register_field(parttype, name=name, description=description)
        return res

    def merge_data(self, secondobj, fieldname_suffix="", root_group: Optional[str] = None):
        """
        Merge data from other snapshot into self.data.

        Parameters
        ----------
        secondobj: GadgetStyleSnapshot
        fieldname_suffix: str
        root_group: Optional[str]

        Returns
        -------
        None

        """
        data = self.data
        if root_group is not None:
            if root_group not in data._containers:
                data.add_container(root_group)
            data = self.data[root_group]
        for k in secondobj.data:
            key = k + fieldname_suffix
            if key not in data:
                data[key] = secondobj.data[k]
            else:
                log.debug("Not overwriting field '%s' during merge_data." % key)
            secondobj.data.fieldrecipes_kwargs["snap"] = self

    def merge_hints(self, secondobj):
        """
        Merge hints from other snapshot into self.hints.

        Parameters
        ----------
        secondobj: GadgetStyleSnapshot
            Other snapshot to merge hints from.

        Returns
        -------
        None
        """
        # merge hints from snap and catalog
        for h in secondobj.hints:
            if h not in self.hints:
                self.hints[h] = secondobj.hints[h]
            elif isinstance(self.hints[h], dict):
                # merge dicts
                for k in secondobj.hints[h]:
                    if k not in self.hints[h]:
                        self.hints[h][k] = secondobj.hints[h][k]
            else:
                pass  # nothing to do; we do not overwrite with catalog props

    @classmethod
    def _clean_metadata_from_raw(cls, rawmetadata):
        """
        Set metadata from raw metadata.
        """
        # check whether we inherit from CosmologyMixin
        cosmological = issubclass(cls, CosmologyMixin)
        metadata = dict()
        if "/Header" in rawmetadata:
            header = rawmetadata["/Header"]
            if cosmological and "Redshift" in header:
                metadata["redshift"] = get_scalar(header["Redshift"])
                metadata["z"] = metadata["redshift"]
            if "BoxSize" in header:
                # can be scalar or array
                metadata["boxsize"] = header["BoxSize"]
            if "Time" in header:
                metadata["time"] = get_scalar(header["Time"])
                metadata["t"] = metadata["time"]
        return metadata

    def _set_metadata(self):
        """
        Set metadata from header and config.
        """
        md = self._clean_metadata_from_raw(self._metadata_raw)
        self.metadata = md
