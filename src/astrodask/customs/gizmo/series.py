import os

from astrodask.discovertypes import CandidateStatus
from astrodask.series import DatasetSeries


class GizmoSimulation(DatasetSeries):
    """A series representing a gizmo simulation."""

    pass

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        print("validating path", path)
        print("isdir", os.path.isdir(path))
        if not os.path.isdir(path):
            return CandidateStatus.NO
        if "gizmo_parameters.txt" in os.listdir(path):
            return CandidateStatus.YES
        return CandidateStatus.NO
