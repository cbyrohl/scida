import os
from typing import Union

import numpy as np

from scida import ArepoSnapshot
from scida.discovertypes import CandidateStatus
from scida.interface import Selector
from scida.io import load_metadata


class TNGClusterSelector(Selector):
    def __init__(self) -> None:
        super().__init__()
        self.keys = ["zoomID", "withfuzz", "onylfuzz"]

    def prepare(self, *args, **kwargs) -> None:
        snap: TNGClusterSnapshot = args[0]
        zoom_id = kwargs.get("zoomID", None)
        fuzz = kwargs.get("withfuzz", None)
        onlyfuzz = kwargs.get("onlylfuzz", None)
        if zoom_id is None:
            return
        if zoom_id < 0 or zoom_id > (snap.ntargets - 1):
            raise ValueError("zoomID must be in range 0-%i" % (snap.ntargets - 1))

        for p in self.data_backup:
            if p.startswith("PartType"):
                key = "particles"
            elif p == "Group":
                key = "groups"
            elif p == "Subhalo":
                key = "subgroups"
            else:
                continue
            lengths = snap.lengths_zoom[key][zoom_id]
            offsets = snap.offsets_zoom[key][zoom_id]
            lengths_fuzz = None
            offsets_fuzz = None
            if fuzz and key == "particles":  # fuzz files only for particles
                lengths_fuzz = snap.lengths_zoom[key][zoom_id + snap.ntargets]
                offsets_fuzz = snap.offsets_zoom[key][zoom_id + snap.ntargets]

            if key == "particles":
                splt = p.split("PartType")
                pnum = int(splt[1])
                offset = offsets[pnum]
                length = lengths[pnum]
            else:
                offset = offsets
                length = lengths
            for k, v in self.data_backup[p].items():
                arr = v[offset : offset + length]
                if fuzz and key == "particles":
                    offset_fuzz = offsets_fuzz[pnum]
                    length_fuzz = lengths_fuzz[pnum]
                    arr_fuzz = v[offset_fuzz : offset_fuzz + length_fuzz]
                    if onlyfuzz:
                        arr = arr_fuzz
                    else:
                        arr = np.concatenate([arr, arr_fuzz])
                self.data[p][k] = arr
        snap.data = self.data


class TNGClusterSnapshot(ArepoSnapshot):
    _fileprefix_catalog = "fof_subhalo_tab_"
    _fileprefix = "snap_"
    ntargets = 352

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # we can get the offsets from the header as our load routine concats the various values from different files
        # these offsets can be used to select a given halo.
        # see: https://tng-project.org/w/index.php/TNG-Cluster
        # each zoom-target has two entries i and i+N, where N=352 is the number of zoom targets.
        # the first file contains the particles that were contained in the original low-res run
        # the second file contains all other remaining particles in a given zoom target
        def len_to_offsets(lengths):
            shp = len(lengths.shape)
            n = lengths.shape[-1]
            if shp == 1:
                res = np.concatenate([np.zeros(1), np.cumsum(lengths)])[:-1]
            else:
                res = np.cumsum(np.vstack([np.zeros(n), lengths]), axis=0)[:-1]
            return res

        self.lengths_zoom = dict(particles=self.header["NumPart_ThisFile"])
        self.offsets_zoom = dict(
            particles=len_to_offsets(self.lengths_zoom["particles"])
        )

        if hasattr(self, "catalog") and self.catalog is not None:
            self.lengths_zoom["groups"] = self.catalog.header["Ngroups_ThisFile"]
            self.offsets_zoom["groups"] = len_to_offsets(self.lengths_zoom["groups"])
            self.lengths_zoom["subgroups"] = self.catalog.header["Nsubgroups_ThisFile"]
            self.offsets_zoom["subgroups"] = len_to_offsets(
                self.lengths_zoom["subgroups"]
            )

    @TNGClusterSelector()
    def return_data(self):
        return super().return_data()

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
