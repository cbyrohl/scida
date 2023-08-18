import os
from typing import Union

from scida import GadgetStyleSnapshot
from scida.convenience import load
from scida.discovertypes import CandidateStatus
from scida.interfaces.mixins import SpatialCartesian3DMixin
from scida.io import load_metadata


class GizmoSnapshot(SpatialCartesian3DMixin, GadgetStyleSnapshot):
    _fileprefix_catalog = "groups"

    def __init__(self, path, chunksize="auto", catalog=None, **kwargs) -> None:
        self.iscatalog = kwargs.pop("iscatalog", False)
        self.header = {}
        self.config = {}
        self.parameters = {}
        self._grouplengths = {}
        prfx = kwargs.pop("fileprefix", None)
        if prfx is None:
            prfx = self._get_fileprefix(path)
        super().__init__(path, chunksize=chunksize, fileprefix=prfx, **kwargs)
        ureg = None
        if hasattr(self, "ureg"):
            ureg = self.ureg
        rhp = kwargs.get("catalog_rockstar_halo", None)
        if rhp is not None:
            rh = load(rhp, ureg=ureg, **kwargs)
            self.merge_data(rh, root_group="rockstar_halo")
        rhs = kwargs.get("catalog_rockstar_star", None)
        if rhs is not None:
            rs = load(rhs, ureg=ureg, **kwargs)
            self.merge_data(rs, root_group="rockstar_star")

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, **kwargs
    ) -> CandidateStatus:
        valid = super().validate_path(path, *args, **kwargs)
        if valid == CandidateStatus.NO:
            return valid
        valid = CandidateStatus.NO
        metadata_raw = load_metadata(path, **kwargs)

        if "GIZMO_version" in metadata_raw["/Header"]:
            return CandidateStatus.YES  # safe to assume this is a gizmo snapshot

        matchingattrs = True
        matchingattrs &= "Flag_IC_Info" in metadata_raw["/Header"]
        # Following not present in gizmo
        matchingattrs &= "Git_commit" not in metadata_raw["/Header"]  # only in arepo?
        matchingattrs &= "/Config" not in metadata_raw
        matchingattrs &= "/Parameters" not in metadata_raw

        if matchingattrs:
            return CandidateStatus.YES

        return valid
