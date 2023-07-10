import os
from typing import Union

from scida import ArepoSnapshot
from scida.discovertypes import CandidateStatus
from scida.io import load_metadata


class TNGClusterSnapshot(ArepoSnapshot):
    _fileprefix_catalog = "fof_subhalo_tab_"
    _fileprefix = "snap_"

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, **kwargs
    ) -> CandidateStatus:
        tkwargs = dict(
            fileprefix=cls._fileprefix, fileprefix_catalog=cls._fileprefix_catalog
        )
        tkwargs.update(**kwargs)
        valid = super().validate_path(path, *args, **tkwargs)
        if valid == CandidateStatus.NO:
            return valid
        metadata_raw = load_metadata(path, **tkwargs)

        matchingattrs = True

        parameters = metadata_raw["/Parameters"]
        if "InitCondFile" in parameters:
            matchingattrs &= parameters["InitCondFile"] == "various"
        else:
            return CandidateStatus.NO
        header = metadata_raw["/Header"]
        matchingattrs &= header["BoxSize"] == 680000.0
        matchingattrs &= header["NumPart_Total"][1] == 1944529344
        matchingattrs &= header["NumPart_Total"][2] == 586952200

        if matchingattrs:
            valid = CandidateStatus.YES
        else:
            valid = CandidateStatus.NO
        return valid
