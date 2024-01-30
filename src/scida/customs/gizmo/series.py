"""
Defines a series representing a Gizmo simulation.
"""
import os

from scida.customs.gadgetstyle.series import GadgetStyleSimulation
from scida.discovertypes import CandidateStatus


class GizmoSimulation(GadgetStyleSimulation):
    """A dataseries representing a Gizmo simulation."""

    def __init__(self, path, lazy=True, async_caching=False, **interface_kwargs):
        """A series representing a gizmo simulation."""
        prefix_dict = dict(paths="snapdir", rh_paths="halo", rs_paths="star")
        subpath_dict = dict(
            paths="output",
            rh_paths="halo/rockstar_dm/catalog_hdf5",
            rs_paths="halo/rockstar_dm/catalog_hdf5",
        )
        arg_dict = dict(
            rh_paths="catalog_rockstar_halo", rs_paths="catalog_rockstar_star"
        )
        super().__init__(
            path,
            prefix_dict=prefix_dict,
            subpath_dict=subpath_dict,
            arg_dict=arg_dict,
            lazy=lazy,
            async_caching=async_caching,
            **interface_kwargs
        )

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
        if not os.path.isdir(path):
            return CandidateStatus.NO
        if "gizmo_parameters.txt" in os.listdir(path):
            return CandidateStatus.YES
        return CandidateStatus.NO
