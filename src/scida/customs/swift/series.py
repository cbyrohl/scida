import pathlib

from scida.customs.gadgetstyle.series import GadgetStyleSimulation
from scida.discovertypes import CandidateStatus


class SwiftSimulation(GadgetStyleSimulation):
    def __init__(self, path, lazy=True, async_caching=False, **interface_kwargs):
        """A dataseries representing a SWIFT simulation."""
        subpath_dict = dict(paths="snapshots")
        super().__init__(
            path,
            subpath_dict=subpath_dict,
            lazy=lazy,
            async_caching=async_caching,
            **interface_kwargs
        )

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        p = pathlib.Path(path)
        if not p.is_dir():
            return CandidateStatus.NO
        pcode = p / "Code" / "swiftsim"
        if pcode.exists():
            return CandidateStatus.YES
        # TODO: check individual snapshot for swift header
        return CandidateStatus.NO
