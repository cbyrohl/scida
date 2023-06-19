import os

from scida.discovertypes import CandidateStatus
from scida.series import DatasetSeries


class GizmoSimulation(DatasetSeries):
    """A series representing a gizmo simulation."""

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        if not os.path.isdir(path):
            return CandidateStatus.NO
        if "gizmo_parameters.txt" in os.listdir(path):
            return CandidateStatus.YES
        return CandidateStatus.NO
