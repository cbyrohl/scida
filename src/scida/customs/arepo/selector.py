"""
Selector for ArepoSnapshot
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scida.customs.arepo.helpers import grp_type_str
from scida.interface import Selector

if TYPE_CHECKING:
    from scida.customs.arepo.snapshot import ArepoSnapshot


class ArepoSelector(Selector):
    """Selector for ArepoSnapshot.
    Can select for haloID, subhaloID, and unbound particles."""

    def __init__(self) -> None:
        """
        Initialize the selector.
        """
        super().__init__()
        self.keys = ["haloID", "subhaloID", "localSubhaloID", "unbound"]

    def prepare(self, *args, **kwargs) -> None:
        if all([kwargs.get(k, None) is None for k in self.keys]):
            return  # no specific selection, thus just return
        snap: ArepoSnapshot = args[0]
        halo_id = kwargs.get("haloID", None)
        subhalo_id = kwargs.get("subhaloID", None)
        subhalo_id_local: int | None = kwargs.get("localSubhaloID", None)
        unbound = kwargs.get("unbound", None)

        if halo_id is not None and subhalo_id is not None:
            raise ValueError("Cannot select for haloID and subhaloID at the same time.")

        if unbound is True and (halo_id is not None or subhalo_id is not None):
            raise ValueError(
                "Cannot select haloID/subhaloID and unbound particles at the same time."
            )

        if subhalo_id_local is not None and subhalo_id is not None:
            raise ValueError(
                "Cannot select for localSubhaloID and subhaloID at the same time."
            )

        if snap.catalog is None:
            raise ValueError("Cannot select for haloID without catalog loaded.")

        # select for halo
        if subhalo_id_local is not None:
            if halo_id is None:
                raise ValueError("Cannot select for localSubhaloID without haloID.")
            # compute subhalo_id from subhalo_id_local
            shid_of_first_sh = snap.data["Group"]["GroupFirstSub"]
            nshs = int(snap.data["Group"]["GroupNsubs"][halo_id].compute())
            if subhalo_id_local >= nshs:
                raise ValueError("localSubhaloID exceeds number of subhalos in halo.")
            subhalo_id = shid_of_first_sh[halo_id] + subhalo_id_local

        idx = subhalo_id if subhalo_id is not None else halo_id
        objtype = "subhalo" if subhalo_id is not None else "halo"
        if idx is not None:
            self.select_group(snap, idx, objtype=objtype)
        elif unbound is True:
            self.select_unbound(snap)

    def select_unbound(self, snap):
        """
        Select unbound particles.

        Parameters
        ----------
        snap: ArepoSnapshot

        Returns
        -------
        None
        """
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
        """
        Select particles for given group/subhalo index.

        Parameters
        ----------
        snap: ArepoSnapshot
        idx: int
        objtype: str

        Returns
        -------
        None
        """
        # TODO: test whether works for multiple groups via idx list
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
