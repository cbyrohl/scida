"""
Contains Series class for Arepo simulations.
"""

import os
import pathlib
from os.path import join

from scida.customs.gadgetstyle.series import GadgetStyleSimulation
from scida.discovertypes import CandidateStatus


class ArepoSimulation(GadgetStyleSimulation):
    """A series representing an Arepo simulation."""

    def __init__(self, path, lazy=True, async_caching=False, **interface_kwargs):
        """
        Initialize an ArepoSimulation object.

        Parameters
        ----------
        path: str
            Path to the simulation folder, should contain "output" folder.
        lazy: bool
            Whether to load data files lazily.
        interface_kwargs: dict
            Additional keyword arguments passed to the interface.
        """
        # choose parent folder as path if we are passed "output" dir
        p = pathlib.Path(path)
        if p.name == "output":
            path = str(p.parent)
        prefix_dict = dict(paths="snapdir", gpaths="group")
        arg_dict = dict(gpaths="catalog")
        super().__init__(
            path,
            prefix_dict=prefix_dict,
            arg_dict=arg_dict,
            lazy=lazy,
            **interface_kwargs
        )

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        """
        Validate a path as a candidate for this simulation class.

        Parameters
        ----------
        path: str
            Path to validate.
        args: list
            Additional positional arguments.
        kwargs:
            Additional keyword arguments.

        Returns
        -------
        CandidateStatus
            Whether the path is a candidate for this simulation class.
        """
        valid = CandidateStatus.NO
        if not os.path.isdir(path):
            return CandidateStatus.NO
        fns = os.listdir(path)
        if "gizmo_parameters.txt" in fns:
            return CandidateStatus.NO
        sprefixs = ["snapdir", "snapshot"]
        opath = path
        if "output" in fns:
            opath = join(path, "output")
        folders = os.listdir(opath)
        folders = [f for f in folders if os.path.isdir(join(opath, f))]
        if any([f.startswith(k) for f in folders for k in sprefixs]):
            valid = CandidateStatus.MAYBE
        return valid
