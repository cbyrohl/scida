import os
from typing import Union

import numpy as np

from scida import ArepoSnapshot
from scida.discovertypes import CandidateStatus
from scida.fields import DerivedFieldRecipe, FieldContainer, FieldRecipe
from scida.interface import Selector
from scida.io import load_metadata


class TNGClusterSelector(Selector):
    """
    Selector for TNGClusterSnapshot.  Can select for zoomID, which selects a given zoom target.
    Can specify withfuzz=True to include the "fuzz" particles for a given zoom target.
    Can specify onlyfuzz=True to only return the "fuzz" particles for a given zoom target.
    """

    def __init__(self) -> None:
        """
        Initialize the selector.
        """
        super().__init__()
        self.keys = ["zoomID", "withfuzz", "onlyfuzz"]

    def prepare(self, *args, **kwargs) -> None:
        """
        Prepare the selector.

        Parameters
        ----------
        args: list
        kwargs: dict

        Returns
        -------
        None
        """
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
            length_fuzz = None
            offset_fuzz = None

            if fuzz and key == "particles":  # fuzz files only for particles
                lengths_fuzz = snap.lengths_zoom[key][zoom_id + snap.ntargets]
                offsets_fuzz = snap.offsets_zoom[key][zoom_id + snap.ntargets]

                splt = p.split("PartType")
                pnum = int(splt[1])
                offset_fuzz = offsets_fuzz[pnum]
                length_fuzz = lengths_fuzz[pnum]

            if key == "particles":
                splt = p.split("PartType")
                pnum = int(splt[1])
                offset = offsets[pnum]
                length = lengths[pnum]
            else:
                offset = offsets
                length = lengths

            def get_slicedarr(v, offset, length, offset_fuzz, length_fuzz, key, fuzz=False):
                """
                Get a sliced dask array for a given (length, offset) and (length_fuzz, offset_fuzz).

                Parameters
                ----------
                v: da.Array
                    The array to slice.
                offset: int
                length: int
                offset_fuzz: int
                length_fuzz: int
                key: str
                    ?
                fuzz: bool

                Returns
                -------
                da.Array
                    The sliced array.
                """
                arr = v[offset : offset + length]
                if offset_fuzz is not None:
                    arr_fuzz = v[offset_fuzz : offset_fuzz + length_fuzz]
                    if onlyfuzz:
                        arr = arr_fuzz
                    else:
                        arr = np.concatenate([arr, arr_fuzz])
                return arr

            def get_slicedfunc(func, offset, length, offset_fuzz, length_fuzz, key, fuzz=False):
                """
                Slice a functions output for a given (length, offset) and (length_fuzz, offset_fuzz).

                Parameters
                ----------
                func: callable
                offset: int
                length: int
                offset_fuzz: int
                length_fuzz: int
                key: str
                fuzz: bool

                Returns
                -------
                callable
                    The sliced function.
                """

                def newfunc(arrs, o=offset, ln=length, of=offset_fuzz, lnf=length_fuzz, **kwargs):
                    arr_all = func(arrs, **kwargs)
                    arr = arr_all[o : o + ln]
                    if of is None:
                        return arr
                    arr_fuzz = arr_all[of : of + lnf]
                    if onlyfuzz:
                        return arr_fuzz
                    else:
                        return np.concatenate([arr, arr_fuzz])

                return newfunc

            # need to evaluate without recipes first
            for k, v in self.data_backup[p].items(withrecipes=False):
                self.data[p][k] = get_slicedarr(v, offset, length, offset_fuzz, length_fuzz, key, fuzz)

            for k, v in self.data_backup[p].items(withfields=False, withrecipes=True, evaluate=False):
                if not isinstance(v, FieldRecipe):
                    continue  # already evaluated, no need to port recipe (?)
                rcp: FieldRecipe = v
                func = get_slicedfunc(v.func, offset, length, offset_fuzz, length_fuzz, key, fuzz)
                newrcp = DerivedFieldRecipe(rcp.name, func)
                newrcp.type = rcp.type
                newrcp.description = rcp.description
                newrcp.units = rcp.units
                self.data[p][k] = newrcp
        snap.data = self.data


class TNGClusterSnapshot(ArepoSnapshot):
    """
    Dataset class for the TNG-Cluster simulation.
    """

    _fileprefix_catalog = "fof_subhalo_tab_"
    _fileprefix = "snap_"
    ntargets = 352

    def __init__(self, *args, **kwargs):
        """
        Initialize a TNGClusterSnapshot object.

        Parameters
        ----------
        args
        kwargs
        """
        super().__init__(*args, **kwargs)

        # we can get the offsets from the header as our load routine concats the various values from different files
        # these offsets can be used to select a given halo.
        # see: https://tng-project.org/w/index.php/TNG-Cluster
        # each zoom-target has two entries i and i+N, where N=352 is the number of zoom targets.
        # the first file contains the particles that were contained in the original low-res run
        # the second file contains all other remaining particles in a given zoom target
        def len_to_offsets(lengths):
            """
            From the particle count (field length), get the offset of all zoom targets.

            Parameters
            ----------
            lengths

            Returns
            -------
            np.ndarray
            """
            lengths = np.array(lengths)
            shp = len(lengths.shape)
            n = lengths.shape[-1]
            if shp == 1:
                res = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths.astype(np.int64))])[:-1]
            else:
                res = np.cumsum(
                    np.vstack([np.zeros(n, dtype=np.int64), lengths.astype(np.int64)]),
                    axis=0,
                )[:-1]
            return res

        self.lengths_zoom = dict(particles=self.header["NumPart_ThisFile"])
        self.offsets_zoom = dict(particles=len_to_offsets(self.lengths_zoom["particles"]))

        if hasattr(self, "catalog") and self.catalog is not None:
            self.lengths_zoom["groups"] = self.catalog.header["Ngroups_ThisFile"]
            self.offsets_zoom["groups"] = len_to_offsets(self.lengths_zoom["groups"])
            self.lengths_zoom["subgroups"] = self.catalog.header["Nsubgroups_ThisFile"]
            self.offsets_zoom["subgroups"] = len_to_offsets(self.lengths_zoom["subgroups"])

    @TNGClusterSelector()
    def return_data(self) -> FieldContainer:
        """
        Return the data object.

        Returns
        -------
        FieldContainer
            The data object.
        """
        return super().return_data()

    @classmethod
    def validate_path(cls, path: Union[str, os.PathLike], *args, **kwargs) -> CandidateStatus:
        """
        Validate a path as a candidate for TNG-Cluster snapshot class.

        Parameters
        ----------
        path: str
            Path to validate.
        args: list
        kwargs: dict

        Returns
        -------
        CandidateStatus
            Whether the path is a candidate for this simulation class.
        """

        tkwargs = dict(fileprefix=cls._fileprefix, fileprefix_catalog=cls._fileprefix_catalog)
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
