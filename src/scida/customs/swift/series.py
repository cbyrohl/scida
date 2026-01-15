"""
Defines a series representing a SWIFT simulation.
"""

from __future__ import annotations

import pathlib

from scida.customs.gadgetstyle.series import GadgetStyleSimulation
from scida.discovertypes import CandidateStatus


class SwiftSimulation(GadgetStyleSimulation):
    """
    A dataseries representing a SWIFT simulation.
    """

    def __init__(self, path, lazy=True, async_caching=False, **interface_kwargs):
        """
        Initialize a SWIFT simulation series.

        Parameters
        ----------
        path: str
            Path to the simulation.
        lazy: bool
            Whether to load the data lazily.
        interface_kwargs: dict
            Additional keyword arguments to pass to the dataset initializations.
        """
        subpath_dict = dict(paths="snapshots")
        super().__init__(path, subpath_dict=subpath_dict, lazy=lazy, **interface_kwargs)

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        """
        Validate a path as a candidate for this simulation class.

        Parameters
        ----------
        path: str
        args: list
        kwargs: dict

        Returns
        -------
        CandidateStatus
            Whether the path is a candidate for this simulation class.
        """
        p = pathlib.Path(path)
        if not p.is_dir():
            return CandidateStatus.NO
        pcode = p / "Code" / "swiftsim"
        if pcode.exists():
            return CandidateStatus.YES
        # TODO: check individual snapshot for swift header
        return CandidateStatus.NO
