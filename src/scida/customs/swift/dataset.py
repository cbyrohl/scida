"""
Defines the SwiftSnapshot class for the SWIFT code, see https://gitlab.cosma.dur.ac.uk/swift/swiftsim
"""

from __future__ import annotations

import os

from scida import GadgetStyleSnapshot
from scida.discovertypes import CandidateStatus
from scida.io import load_metadata


class SwiftSnapshot(GadgetStyleSnapshot):
    """SWIFT snapshot dataset."""

    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        """
        Initialize a SwiftSnapshot object.

        Parameters
        ----------
        path: str
        chunksize: int
        virtualcache: bool
        kwargs: dict
        """

        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

    @classmethod
    def validate_path(cls, path: str | os.PathLike, *args, **kwargs) -> CandidateStatus:
        """
        Check if path is valid for this dataset class.

        Parameters
        ----------
        path: str, os.PathLike
            path to check
        args: list
        kwargs: dict

        Returns
        -------
        CandidateStatus
        """
        valid = super().validate_path(path, *args, **kwargs)
        if valid == CandidateStatus.NO:
            return CandidateStatus.NO
        metadata_raw = load_metadata(path, **kwargs)
        comparestr = metadata_raw.get("/Code", {}).get("Code", b"").decode()
        if "SWIFT" not in comparestr:
            return CandidateStatus.NO
        return CandidateStatus.MAYBE
