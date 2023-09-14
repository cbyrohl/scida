from typing import TYPE_CHECKING

from scida.customs.arepo.helpers import grp_type_str
from scida.interface import Selector

if TYPE_CHECKING:
    from scida.customs.arepo.snapshot import ArepoSnapshot


class ArepoSelector(Selector):
    def __init__(self) -> None:
        super().__init__()
        self.keys = ["haloID", "subhaloID", "unbound"]

    def prepare(self, *args, **kwargs) -> None:
        if all([kwargs.get(k, None) is None for k in self.keys]):
            return  # no specific selection, thus just return
        snap: ArepoSnapshot = args[0]
        halo_id = kwargs.get("haloID", None)
        subhalo_id = kwargs.get("subhaloID", None)
        unbound = kwargs.get("unbound", None)

        if halo_id is not None and subhalo_id is not None:
            raise ValueError("Cannot select for haloID and subhaloID at the same time.")

        if unbound is True and (halo_id is not None or subhalo_id is not None):
            raise ValueError(
                "Cannot select haloID/subhaloID and unbound particles at the same time."
            )

        if snap.catalog is None:
            raise ValueError("Cannot select for haloID without catalog loaded.")

        # select for halo
        idx = subhalo_id if subhalo_id is not None else halo_id
        objtype = "subhalo" if subhalo_id is not None else "halo"
        if idx is not None:
            self.select_group(snap, idx, objtype=objtype)
        elif unbound is True:
            self.select_unbound(snap)

    def select_unbound(self, snap):
        lengths = self.data_backup["Group"]["GroupLenType"][-1, :].compute()
        offsets = self.data_backup["Group"]["GroupOffsetsType"][-1, :].compute()
        # for unbound gas, we start after the last halo particles
        offsets = offsets + lengths
        for p in self.data_backup:
            splt = p.split("PartType")
            if len(splt) == 1:
                for k, v in self.data_backup[p].items():
                    self.data[p][k] = v
            else:
                pnum = int(splt[1])
                offset = offsets[pnum]
                if hasattr(offset, "magnitude"):  # hack for issue 59
                    offset = offset.magnitude
                for k, v in self.data_backup[p].items():
                    self.data[p][k] = v[offset:-1]
        snap.data = self.data

    def select_group(self, snap, idx, objtype="Group"):
        objtype = grp_type_str(objtype)
        if objtype == "halo":
            lengths = self.data_backup["Group"]["GroupLenType"][idx, :].compute()
            offsets = self.data_backup["Group"]["GroupOffsetsType"][idx, :].compute()
        elif objtype == "subhalo":
            lengths = {i: snap.get_subhalolengths(i)[idx] for i in range(6)}
            offsets = {i: snap.get_subhalooffsets(i)[idx] for i in range(6)}
        else:
            raise ValueError("Unknown object type: %s" % objtype)

        for p in self.data_backup:
            splt = p.split("PartType")
            if len(splt) == 1:
                for k, v in self.data_backup[p].items():
                    self.data[p][k] = v
            else:
                pnum = int(splt[1])
                offset = offsets[pnum]
                length = lengths[pnum]
                if hasattr(offset, "magnitude"):  # hack for issue 59
                    offset = offset.magnitude
                if hasattr(length, "magnitude"):
                    length = length.magnitude
                for k, v in self.data_backup[p].items():
                    self.data[p][k] = v[offset : offset + length]
        snap.data = self.data
