import copy
import logging
import os
import re
import warnings
from typing import Dict, List, Optional, Union

import dask
import dask.array as da
import numpy as np
from dask import delayed
from numba import jit, njit
from numpy.typing import NDArray

from astrodask.fields import FieldContainer
from astrodask.helpers_misc import (
    computedecorator,
    get_args,
    get_kwargs,
    parse_humansize,
)
from astrodask.interface import Dataset, Selector
from astrodask.interfaces.mixins import CosmologyMixin, SpatialCartesian3DMixin
from astrodask.io import load_metadata

log = logging.getLogger(__name__)


class ArepoSelector(Selector):
    def __init__(self) -> None:
        super().__init__()
        self.keys = ["haloID"]

    def prepare(self, *args, **kwargs) -> None:
        snap = args[0]
        if snap.catalog is None:
            raise ValueError("Cannot select for haloID without catalog loaded.")
        halo_id = kwargs.get("haloID", None)
        if halo_id is None:
            return
        lengths = self.data_backup["Group"]["GroupLenType"][halo_id, :].compute()
        offsets = self.data_backup["Group"]["GroupOffsetsType"][halo_id, :].compute()
        for p in self.data_backup:
            splt = p.split("PartType")
            if len(splt) == 1:
                for k, v in self.data_backup[p].items():
                    self.data[p][k] = v
            else:
                pnum = int(splt[1])
                offset = offsets[pnum]
                length = lengths[pnum]
                for k, v in self.data_backup[p].items():
                    self.data[p][k] = v[offset : offset + length]
        snap.data = self.data


class BaseSnapshot(Dataset):
    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        self.boxsize = np.full(3, np.nan)
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

        defaultattributes = ["config", "header", "parameters"]
        for k in self._metadata_raw:
            name = k.strip("/").lower()
            if name in defaultattributes:
                self.__dict__[name] = self._metadata_raw[k]
                if "BoxSize" in self.__dict__[name]:
                    self.boxsize = self.__dict__[name]["BoxSize"]
                elif "Boxsize" in self.__dict__[name]:
                    self.boxsize = self.__dict__[name]["Boxsize"]

    @classmethod
    def _get_fileprefix(cls, path: Union[str, os.PathLike], **kwargs) -> str:
        """
        Get the fileprefix used to identify files belonging to given dataset.
        Parameters
        ----------
        path: str, os.PathLike
            path to check
        kwargs

        Returns
        -------
        str
        """
        if os.path.isfile(path):
            return ""  # nothing to do, we have a single file, not a directory
        # order matters: groups will be taken before fof_subhalo, requires py>3.7 for dict order
        prfxs_prfx_sim = dict.fromkeys(["groups", "fof_subhalo", "snap"])
        files = sorted(os.listdir(path))
        prfxs_lst = []
        for fn in files:
            s = re.search(r"^(\w*)_(\d*)", fn)
            if s is not None:
                prfxs_lst.append(s.group(1))
        prfxs_lst = [p for s in prfxs_prfx_sim for p in prfxs_lst if p.startswith(s)]
        prfxs = dict.fromkeys(prfxs_lst)
        prfxs = list(prfxs.keys())
        if len(prfxs) > 1:
            log.debug("We have more than one prefix avail: %s" % prfxs)
        elif len(prfxs) == 0:
            return ""
        return prfxs[0]

    @classmethod
    def validate_path(cls, path: Union[str, os.PathLike], *args, **kwargs) -> bool:
        """
        Check if path is valid for this interface.
        Parameters
        ----------
        path: str, os.PathLike
            path to check
        args
        kwargs

        Returns
        -------
        bool
        """
        path = str(path)
        possibly_valid = False
        iszarr = path.rstrip("/").endswith(".zarr")
        if path.endswith(".hdf5") or iszarr:
            possibly_valid = True
        if os.path.isdir(path):
            files = os.listdir(path)
            sufxs = [f.split(".")[-1] for f in files]
            if not iszarr and len(set(sufxs)) > 1:
                possibly_valid = False
            if sufxs[0] == "hdf5":
                possibly_valid = True
        if possibly_valid:
            metadata_raw = load_metadata(path, **kwargs)
            # need some silly combination of attributes to be sure
            if all([k in metadata_raw for k in ["/Config", "/Header", "/Parameters"]]):
                if (
                    "NumPart_ThisFile" in metadata_raw["/Header"]
                    and "NumPart_Total" in metadata_raw["/Header"]
                ):
                    return True
        return False

    def register_field(self, parttype, name=None, description=""):
        res = self.data.register_field(parttype, name=name, description=description)
        return res


class ArepoSnapshot(SpatialCartesian3DMixin, BaseSnapshot):
    def __init__(self, path, chunksize="auto", catalog=None, **kwargs) -> None:
        self.iscatalog = kwargs.pop("iscatalog", False)
        self.header = {}
        self.config = {}
        self.parameters = {}
        self._grouplengths = {}
        prfx = kwargs.pop("fileprefix", None)
        if prfx is None:
            prfx = self._get_fileprefix(path)
        super().__init__(path, chunksize=chunksize, fileprefix=prfx, **kwargs)

        self.catalog = catalog
        if not self.iscatalog:
            if self.catalog is None:
                self.discover_catalog()
                # try to discover group catalog in parent directories.
            if self.catalog is not None:
                virtualcache = False  # copy catalog for better performance
                catalog_kwargs = kwargs.get("catalog_kwargs", {})
                catalog_kwargs["overwritecache"] = kwargs.get("overwritecache", False)
                fileprefix = catalog_kwargs.get("fileprefix", "")
                self.catalog = ArepoSnapshot(
                    self.catalog,
                    virtualcache=virtualcache,
                    fileprefix=fileprefix,
                    iscatalog=True,
                )
                if "Redshift" in self.catalog.header and "Redshift" in self.header:
                    z_catalog = self.catalog.header["Redshift"]
                    z_snap = self.header["Redshift"]
                    if not np.isclose(z_catalog, z_snap):
                        raise ValueError(
                            "Redshift mismatch between snapshot and catalog: "
                            f"{z_snap:.2f} vs {z_catalog:.2f}"
                        )
                for k in self.catalog.data:
                    if k not in self.data:
                        self.data[k] = self.catalog.data[k]
                    self.catalog.data.fieldrecipes_kwargs["snap"] = self
                print(
                    self.path,
                    self.catalog.data.keys(),
                    self.catalog.data["Group"].keys(),
                )
                if (
                    len(self.catalog.data["Group"].keys()) > 0
                ):  # starting snapshots often dont have groups
                    self.add_catalogIDs()

                # merge hints from snap and catalog
                for h in self.catalog.hints:
                    if h not in self.hints:
                        self.hints[h] = self.catalog.hints[h]
                    elif isinstance(self.hints[h], dict):
                        # merge dicts
                        for k in self.catalog.hints[h]:
                            if k not in self.hints[h]:
                                self.hints[h][k] = self.catalog.hints[h][k]
                    else:
                        pass  # nothing to do; we do not overwrite with catalog props

        # add aliases
        aliases = dict(
            PartType0=["gas", "baryons"],
            PartType1=["dm", "dark matter"],
            PartType2=["lowres", "lowres dm"],
            PartType3=["tracer", "tracers"],
            PartType4=["stars"],
            PartType5=["bh", "black holes"],
        )
        for k, lst in aliases.items():
            if k not in self.data:
                continue
            for v in lst:
                self.data.add_alias(v, k)

        # set metadata
        self._set_metadata()

        # add some default fields
        self.data.merge(fielddefs)

    def _set_metadata(self):
        """
        Set metadata from header and config.
        """
        if self.header is not None:
            if "Redshift" in self.header:
                self.metadata["redshift"] = self.header["Redshift"]
                self.metadata["z"] = self.metadata["redshift"]
            if "BoxSize" in self.header:
                self.metadata["boxsize"] = self.header["BoxSize"]
            if "Time" in self.header:
                self.metadata["time"] = self.header["Time"]
                self.metadata["t"] = self.metadata["time"]

    @ArepoSelector()
    def return_data(self):
        """
        Return data.
        Returns
        -------

        """
        return super().return_data()

    def discover_catalog(self):
        """
        Discover the group catalog given the current path
        Returns
        -------

        """
        # order of candidates matters. For Illustris "groups" must precede "fof_subhalo_tab"
        candidates = [
            self.path.replace("snapshot", "group"),
            self.path.replace("snapshot", "groups"),
            self.path.replace("snapdir", "groups").replace("snap", "groups"),
            self.path.replace("snapdir", "groups").replace("snap", "fof_subhalo_tab"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            if candidate == self.path:
                continue
            self.catalog = candidate
            break
        print(self.path)

    def register_field(self, parttype: str, name: str = None, construct: bool = True):
        """
        Register a field.
        Parameters
        ----------
        parttype: str
            name of particle type
        name: str
            name of field
        construct: bool
            construct field immediately

        Returns
        -------

        """
        num = partTypeNum(parttype)
        if construct:  # TODO: introduce (immediate) construct option later
            raise NotImplementedError
        if num == -1:  # TODO: all particle species
            key = "all"
            raise NotImplementedError
        elif isinstance(num, int):
            key = "PartType" + str(num)
        else:
            key = parttype
        return super().register_field(key, name=name)

    def add_catalogIDs(self) -> None:
        """
        Add field for halo and subgroup IDs for all particle types.
        Returns
        -------

        """
        # TODO: make these delayed objects and properly pass into (delayed?) numba functions:
        # https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-repeatedly-putting-large-inputs-into-delayed-calls

        # Group ID
        if "Group" not in self.data:  # can happen for empty catalogs
            for key in self.data:
                if not (key.startswith("PartType")):
                    continue
                maxint = np.iinfo(np.int64).max
                uid = self.data[key]["uid"]
                self.data[key]["GroupID"] = maxint * da.ones_like(uid, dtype=np.int64)
                self.data[key]["SubhaloID"] = -1 * da.ones_like(uid, dtype=np.int64)
            return

        glen = self.data["Group"]["GroupLenType"]
        da_halocelloffsets = (
            da.concatenate(  # TODO: Do not hardcode shape of 6 particle types!
                [
                    np.zeros((1, 6), dtype=np.int64),
                    da.cumsum(glen, axis=0, dtype=np.int64),
                ]
            )
        )
        # remove last entry to match shapematch shape
        self.data["Group"]["GroupOffsetsType"] = da_halocelloffsets[:-1].rechunk(
            glen.chunks
        )
        halocelloffsets = da_halocelloffsets.rechunk(-1)

        for key in self.data:
            if not (key.startswith("PartType")):
                continue
            num = int(key[-1])
            if "uid" not in self.data[key]:
                continue  # can happen for empty containers
            gidx = self.data[key]["uid"]
            hidx = compute_haloindex(gidx, halocelloffsets[:, num])
            self.data[key]["GroupID"] = hidx

        # Subhalo ID
        if "Subhalo" not in self.data:  # can happen for empty catalogs
            for key in self.data:
                if not (key.startswith("PartType")):
                    continue
                self.data[key]["SubhaloID"] = -1 * da.ones_like(
                    da[key]["uid"], dtype=np.int64
                )
            return

        subhalogrnr = self.data["Subhalo"]["SubhaloGrNr"]
        subhalocellcounts = self.data["Subhalo"]["SubhaloLenType"]

        for key in self.data:
            if not (key.startswith("PartType")):
                continue
            num = int(key[-1])
            if "uid" not in self.data[key]:
                continue  # can happen for empty containers
            gidx = self.data[key]["uid"]
            dlyd = delayed(get_shcounts_shcells)(
                subhalogrnr, halocelloffsets[:, num].shape[0]
            )
            sidx = compute_subhaloindex(
                gidx,
                halocelloffsets[:, num],
                dlyd[1],
                dlyd[0],
                subhalocellcounts[:, num],
            )
            self.data[key]["SubhaloID"] = sidx

    @computedecorator
    def map_halo_operation(
        self,
        func,
        chunksize=int(3e7),
        cpucost_halo=1e4,
        Nmin=None,
        chunksize_bytes=None,
    ):
        dfltkwargs = get_kwargs(func)
        fieldnames = dfltkwargs.get("fieldnames", None)
        if fieldnames is None:
            fieldnames = get_args(func)
        parttype = dfltkwargs.get("parttype", "PartType0")
        entry_nbytes_in = np.sum([self.data[parttype][f][0].nbytes for f in fieldnames])
        lengths = self.get_grouplengths(parttype=parttype)
        arrdict = self.data[parttype]
        return map_halo_operation(
            func,
            lengths,
            arrdict,
            chunksize=chunksize,
            cpucost_halo=cpucost_halo,
            Nmin=Nmin,
            chunksize_bytes=chunksize_bytes,
            entry_nbytes_in=entry_nbytes_in,
        )

    def add_groupquantity_to_particles(self, name, parttype="PartType0"):
        assert (
            name not in self.data[parttype]
        )  # we simply map the name from Group to Particle for now. Should work (?)
        glen = self.data["Group"]["GroupLenType"]
        da_halocelloffsets = da.concatenate(
            [np.zeros((1, 6), dtype=np.int64), da.cumsum(glen, axis=0)]
        )
        if "GroupOffsetsType" not in self.data["Group"]:
            self.data["Group"]["GroupOffsetsType"] = da_halocelloffsets[:-1].rechunk(
                glen.chunks
            )  # remove last entry to match shape
        halocelloffsets = da_halocelloffsets.compute()

        gidx = self.data[parttype]["uid"]
        num = int(parttype[-1])
        hquantity = compute_haloquantity(
            gidx, halocelloffsets[:, num], self.data["Group"][name]
        )
        self.data[parttype][name] = hquantity

    def get_grouplengths(self, parttype="PartType0"):
        # todo: write/use PartType func always using integer rather than string?
        if parttype not in self._grouplengths:
            partnum = int(parttype[-1])
            lengths = self.data["Group"]["GroupLenType"][:, partnum].compute()
            self._grouplengths[parttype] = lengths
        return self._grouplengths[parttype]

    def grouped(
        self,
        fields: Union[str, da.Array, List[str], Dict[str, da.Array]] = "",
        parttype="PartType0",
    ):
        inputfields = None
        if isinstance(fields, str):
            if fields == "":  # if nothing is specified, we pass all we have.
                arrdict = self.data[parttype]
            else:
                arrdict = dict(field=self.data[parttype][fields])
                inputfields = [fields]
        elif isinstance(fields, da.Array):
            arrdict = dict(daskarr=fields)
            inputfields = [fields.name]
        elif isinstance(fields, list):
            arrdict = {k: self.data[parttype][k] for k in fields}
            inputfields = fields
        elif isinstance(fields, dict):
            arrdict = {}
            arrdict.update(**fields)
            inputfields = list(arrdict.keys())
        else:
            raise ValueError("Unknown input type.")
        gop = GroupAwareOperation(
            self.get_grouplengths(parttype=parttype), arrdict, inputfields=inputfields
        )
        return gop


class CosmologicalArepoSnapshot(CosmologyMixin, ArepoSnapshot):
    _mixins = [CosmologyMixin]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GroupAwareOperation:
    opfuncs = dict(min=np.min, max=np.max, sum=np.sum, half=lambda x: x[::2])
    finalops = {"min", "max", "sum"}
    __slots__ = ("arrs", "ops", "lengths", "final", "inputfields", "opfuncs_custom")

    def __init__(
        self, lengths: NDArray, arrs: Dict[str, da.Array], ops=None, inputfields=None
    ):
        self.lengths = lengths
        self.arrs = arrs
        self.opfuncs_custom = {}
        self.final = False
        self.inputfields = inputfields
        if ops is None:
            self.ops = []
        else:
            self.ops = ops

    def chain(self, add_op=None, final=False):
        if self.final:
            raise ValueError("Cannot chain any additional operation.")
        c = copy.copy(self)
        c.final = final
        c.opfuncs_custom = self.opfuncs_custom
        if add_op is not None:
            if isinstance(add_op, str):
                c.ops.append(add_op)
            elif callable(add_op):
                name = "custom" + str(len(self.opfuncs_custom) + 1)
                c.opfuncs_custom[name] = add_op
                c.ops.append(name)
            else:
                raise ValueError("Unknown operation of type '%s'" % str(type(add_op)))
        return c

    def min(self, field=None):
        if field is not None:
            if self.inputfields is not None:
                raise ValueError("Cannot change input field anymore.")
            self.inputfields = [field]
        return self.chain(add_op="min", final=True)

    def max(self, field=None):
        if field is not None:
            if self.inputfields is not None:
                raise ValueError("Cannot change input field anymore.")
            self.inputfields = [field]
        return self.chain(add_op="max", final=True)

    def sum(self, field=None):
        if field is not None:
            if self.inputfields is not None:
                raise ValueError("Cannot change input field anymore.")
            self.inputfields = [field]
        return self.chain(add_op="sum", final=True)

    def half(self):  # dummy operation for testing halfing particle count.
        return self.chain(add_op="half", final=False)

    def apply(self, func, final=False):
        return self.chain(add_op=func, final=final)

    def __copy__(self):
        # overwrite method so that copy holds a new ops list.
        c = type(self)(
            self.lengths, self.arrs, ops=list(self.ops), inputfields=self.inputfields
        )
        return c

    def evaluate(self, compute=True):
        # final operations: those that can only be at end of chain
        # intermediate operations: those that can only be prior to end of chain
        funcdict = dict()
        funcdict.update(**self.opfuncs)
        funcdict.update(**self.opfuncs_custom)

        def chainops(*funcs):
            def chained_call(*args):
                cf = None
                for i, f in enumerate(funcs):
                    # first chain element can be multiple fields. treat separately
                    if i == 0:
                        cf = f(*args)
                    else:
                        cf = f(cf)
                return cf

            return chained_call

        func = chainops(*[funcdict[k] for k in self.ops])

        fieldnames = list(self.arrs.keys())
        if self.inputfields is None:
            opname = self.ops[0]
            if opname.startswith("custom"):
                dfltkwargs = get_kwargs(self.opfuncs_custom[opname])
                fieldnames = dfltkwargs.get("fieldnames", None)
                if isinstance(fieldnames, str):
                    fieldnames = [fieldnames]
                if fieldnames is None:
                    raise ValueError(
                        "Either pass fields to grouped(fields=...) "
                        "or specify fieldnames=... in applied func."
                    )
            else:
                raise ValueError(
                    "Specify field to operate on in operation or grouped()."
                )

        res = map_halo_operation(func, self.lengths, self.arrs, fieldnames=fieldnames)
        if compute:
            res = res.compute()
        return res


def wrap_func_scalar(
    func,
    halolengths_in_chunks,
    *arrs,
    block_info=None,
    block_id=None,
    func_output_shape=(1,),
    func_output_dtype="float64",
    func_output_default=0,
):
    lengths = halolengths_in_chunks[block_id[0]]

    offsets = np.cumsum([0] + list(lengths))
    res = []
    for i, o in enumerate(offsets[:-1]):
        if o == offsets[i + 1]:
            res.append(
                func_output_default
                * np.ones(func_output_shape, dtype=func_output_dtype)
            )
            if func_output_shape == (1,):
                res[-1] = res[-1].item()
            continue
        arrchunks = [arr[o : offsets[i + 1]] for arr in arrs]
        res.append(func(*arrchunks))
    return np.array(res)


# class ArepoSnapshotWithUnits(ArepoSnapshot):
#    def __init__(self, path, catalog=None):
#        super().__init__(path, catalog=catalog)
#        # unitdict = get_units_from_AREPOdocs(unitstr_arepo, self.header) # from AREPO public docs
#        unitdict = get_units_from_TNGdocs(self.header)  # from TNG public docs
#        for k in self.data:
#            if k.startswith("PartType"):
#                dct = unitdict["particles"]
#            elif k == "Group":
#                dct = unitdict["groups"]
#            elif k == "Subhalo":
#                dct = unitdict["subhalos"]
#            else:
#                dct = unitdict[k]
#            for name in self.data[k]:
#                if name in dct:
#                    self.data[k][name] = self.data[k][name] * dct[name]
#                else:
#                    logging.info("No units for '%s'" % name)


@njit
def get_hidx(gidx_start, gidx_count, celloffsets):
    """Get halo index of a given cell

    Parameters
    ----------
    gidx_start: integer
        The first unique integer ID for the first particle
    gidx_count: integer
        The amount of halo indices we are querying after "gidx_start"
    celloffsets : array
        An array holding the starting cell offset for each halo. Needs to include the
        offset after the last halo. The required shape is thus (Nhalo+1,).
    """
    dtype = np.int64
    index_unbound = np.iinfo(dtype).max
    res = index_unbound * np.ones(gidx_count, dtype=dtype)
    # find initial celloffset
    hidx_idx = np.searchsorted(celloffsets, gidx_start, side="right") - 1
    if hidx_idx + 1 >= celloffsets.shape[0]:
        # we are done. Already out of scope of lookup => all unbound gas.
        return res
    celloffset = celloffsets[hidx_idx + 1]
    endid = celloffset - gidx_start
    startid = 0

    # Now iterate through list.
    while startid < gidx_count:
        res[startid:endid] = hidx_idx
        hidx_idx += 1
        startid = endid
        if hidx_idx >= celloffsets.shape[0] - 1:
            break
        count = celloffsets[hidx_idx + 1] - celloffsets[hidx_idx]
        endid = startid + count
    return res


def get_hidx_daskwrap(gidx, halocelloffsets):
    gidx_start = gidx[0]
    gidx_count = gidx.shape[0]
    return get_hidx(gidx_start, gidx_count, halocelloffsets)


def get_haloquantity_daskwrap(gidx, halocelloffsets, valarr):
    hidx = get_hidx_daskwrap(gidx, halocelloffsets)
    dmax = np.iinfo(hidx.dtype).max
    mask = ~((hidx == -1) | (hidx == dmax))
    result = -1.0 * np.ones(hidx.shape, dtype=valarr.dtype)
    result[mask] = valarr[hidx[mask]]
    return result


def compute_haloindex(gidx, halocelloffsets, *args):
    """Computes the halo index for each particle with dask."""
    return da.map_blocks(
        get_hidx_daskwrap, gidx, halocelloffsets, meta=np.array((), dtype=np.int64)
    )


def compute_haloquantity(gidx, halocelloffsets, hvals, *args):
    """Computes a halo quantity for each particle with dask."""
    return da.map_blocks(
        get_haloquantity_daskwrap,
        gidx,
        halocelloffsets,
        hvals,
        meta=np.array((), dtype=hvals.dtype),
    )


@jit
def get_shidx(
    gidx_start: int,
    gidx_count: int,
    celloffsets: NDArray[np.int64],
    shnumber,
    shcounts,
    shcellcounts,
):
    res = -1 * np.ones(gidx_count, dtype=np.int32)  # fuzz has negative index.

    # find initial Group we are in
    hidx_start_idx = np.searchsorted(celloffsets, gidx_start, side="right") - 1
    if hidx_start_idx + 1 >= celloffsets.shape[0]:
        # we are done. Already out of scope of lookup => all unbound gas.
        return res
    celloffset = celloffsets[hidx_start_idx + 1]
    endid = celloffset - gidx_start
    startid = 0

    # find initial subhalo we are in
    hidx = hidx_start_idx
    shcumsum = np.zeros(shcounts[hidx] + 1, dtype=np.int64)
    shcumsum[1:] = np.cumsum(
        shcellcounts[shnumber[hidx] : shnumber[hidx] + shcounts[hidx]]
    )  # collect halo's subhalo offsets
    shcumsum += celloffsets[hidx_start_idx]
    sidx_start_idx: int = int(np.searchsorted(shcumsum, gidx_start, side="right") - 1)
    if sidx_start_idx < shcounts[hidx]:
        endid = shcumsum[sidx_start_idx + 1] - gidx_start

    # Now iterate through list.
    cont = True
    while cont and (startid < gidx_count):
        res[startid:endid] = (
            sidx_start_idx if sidx_start_idx + 1 < shcumsum.shape[0] else -1
        )
        sidx_start_idx += 1
        if sidx_start_idx < shcounts[hidx_start_idx]:
            # we prepare to fill the next available subhalo for current halo
            count = shcumsum[sidx_start_idx + 1] - shcumsum[sidx_start_idx]
            startid = endid
        else:
            # we need to find the next halo to start filling its subhalos
            dbgcount = 0
            while dbgcount < 100:  # find next halo with >0 subhalos
                hidx_start_idx += 1
                if hidx_start_idx >= shcounts.shape[0]:
                    cont = False
                    break
                if shcounts[hidx_start_idx] > 0:
                    break
                dbgcount += 1
            hidx = hidx_start_idx
            if hidx_start_idx >= celloffsets.shape[0] - 1:
                startid = gidx_count
            else:
                count = celloffsets[hidx_start_idx + 1] - celloffsets[hidx_start_idx]
                if hidx < shcounts.shape[0]:
                    shcumsum = np.zeros(shcounts[hidx] + 1, dtype=np.int64)
                    shcumsum[1:] = np.cumsum(
                        shcellcounts[shnumber[hidx] : shnumber[hidx] + shcounts[hidx]]
                    )
                    shcumsum += celloffsets[hidx_start_idx]
                    sidx_start_idx = 0
                    if sidx_start_idx < shcounts[hidx]:
                        count = shcumsum[sidx_start_idx + 1] - shcumsum[sidx_start_idx]
                    startid = celloffsets[hidx_start_idx] - gidx_start
        endid = startid + count
    return res


def get_shidx_daskwrap(
    gidx: NDArray[np.int64],
    halocelloffsets: NDArray[np.int64],
    shnumber,
    shcounts,
    shcellcounts,
) -> np.ndarray:
    gidx_start = gidx[0]
    gidx_count = gidx.shape[0]
    return get_shidx(
        gidx_start, gidx_count, halocelloffsets, shnumber, shcounts, shcellcounts
    )


def compute_subhaloindex(
    gidx, halocelloffsets, shnumber, shcounts, shcellcounts
) -> da.Array:
    return da.map_blocks(
        get_shidx_daskwrap,
        gidx,
        halocelloffsets,
        shnumber,
        shcounts,
        shcellcounts,
        meta=np.array((), dtype=np.int64),
    )


@njit
def get_shcounts_shcells(SubhaloGrNr, hlength):
    """Returns the number offset and count of subhalos per halo."""
    shcounts = np.zeros(hlength, dtype=np.int32)
    shnumber = np.zeros(hlength, dtype=np.int32)
    i = 0
    hid = 0
    hid_old = 0
    while i < SubhaloGrNr.shape[0]:
        hid = SubhaloGrNr[i]
        if hid == hid_old:
            shcounts[hid] += 1
        else:
            shnumber[hid] = i
            shcounts[hid] += 1
            hid_old = hid
        i += 1
    return shcounts, shnumber


def partTypeNum(partType):
    """Mapping between common names and numeric particle types."""
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ["gas", "cells"]:
        return 0
    if str(partType).lower() in ["dm", "darkmatter"]:
        return 1
    if str(partType).lower() in ["dmlowres"]:
        return 2  # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ["tracer", "tracers", "tracermc", "trmc"]:
        return 3
    if str(partType).lower() in ["star", "stars", "stellar"]:
        return 4  # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ["wind"]:
        return 4  # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ["bh", "bhs", "blackhole", "blackholes", "black"]:
        return 5
    if str(partType).lower() in ["all"]:
        return -1


def memorycost_limiter(cost_memory, cost_cpu, list_chunkedges, cost_memory_max):
    """If a chunk too memory expensive, split into equal cpu expense operations."""
    list_chunkedges_new = []
    for chunkedges in list_chunkedges:
        slc = slice(*chunkedges)
        totcost_mem = np.sum(cost_memory[slc])
        list_chunkedges_new.append(chunkedges)
        if totcost_mem > cost_memory_max:
            sumcost = cost_cpu[slc].cumsum()
            sumcost /= sumcost[-1]
            idx = slc.start + np.argmin(np.abs(sumcost - 0.5))
            if idx == chunkedges[0]:
                idx += 1
            elif idx == chunkedges[-1]:
                idx -= 1
            chunkedges1 = [chunkedges[0], idx]
            chunkedges2 = [idx, chunkedges[1]]
            if idx == chunkedges[0] or idx == chunkedges[1]:
                raise ValueError("This should not happen.")
            list_chunkedges_new.pop()
            list_chunkedges_new += memorycost_limiter(
                cost_memory, cost_cpu, [chunkedges1], cost_memory_max
            )
            list_chunkedges_new += memorycost_limiter(
                cost_memory, cost_cpu, [chunkedges2], cost_memory_max
            )
    return list_chunkedges_new


def map_halo_operation_get_chunkedges(
    lengths,
    entry_nbytes_in,
    entry_nbytes_out,
    cpucost_halo=1.0,
    Nmin=None,
    chunksize_bytes=None,
):
    cpucost_particle = 1.0  # we only care about ratio, so keep particle cost fixed.
    cost = cpucost_particle * lengths + cpucost_halo
    sumcost = cost.cumsum()

    # let's allow a maximal chunksize of 16 times the dask default setting for an individual array [here: multiple]
    if chunksize_bytes is None:
        chunksize_bytes = 16 * parse_humansize(dask.config.get("array.chunk-size"))
    cost_memory = entry_nbytes_in * lengths + entry_nbytes_out

    if not np.max(cost_memory) < chunksize_bytes:
        raise ValueError(
            "Some halo requires more memory than allowed (%i allowed, %i requested). Consider overriding chunksize_bytes."
            % (chunksize_bytes, np.max(cost_memory))
        )

    N = int(np.ceil(np.sum(cost_memory) / chunksize_bytes))
    N = int(np.ceil(1.3 * N))  # fudge factor
    if Nmin is not None:
        N = max(Nmin, N)
    targetcost = sumcost[-1] / N
    arr = np.diff(sumcost % targetcost)
    idx = [0] + list(np.where(arr < 0)[0] + 1)
    if len(idx) == N + 1:
        idx[-1] = sumcost.shape[0]
    elif len(idx) - N in [0, -1, -2]:
        idx.append(sumcost.shape[0])
    else:
        raise ValueError("Unexpected chunk indices.")
    list_chunkedges = []
    for i in range(len(idx) - 1):
        list_chunkedges.append([idx[i], idx[i + 1]])

    list_chunkedges = np.asarray(
        memorycost_limiter(cost_memory, cost, list_chunkedges, chunksize_bytes)
    )

    # make sure we did not lose any halos.
    assert np.all(
        ~(list_chunkedges.flatten()[2:-1:2] - list_chunkedges.flatten()[1:-1:2]).astype(
            bool
        )
    )
    return list_chunkedges


def map_halo_operation(
    func,
    lengths,
    arrdict,
    chunksize=int(3e7),
    cpucost_halo=1e4,
    Nmin: Optional[int] = None,
    chunksize_bytes: Optional[int] = None,
    entry_nbytes_in: Optional[int] = 4,
    fieldnames: Optional[List[str]] = None,
) -> da.Array:
    """
    Map a function to all halos in a halo catalog.
    Parameters
    ----------
    func
    lengths
    arrdict
    chunksize
    cpucost_halo
    Nmin
    chunksize_bytes
    entry_nbytes_in
    fieldnames

    Returns
    -------

    """
    if chunksize is not None:
        warnings.warn(
            '"chunksize" parameter is depreciated and has no effect. Specify Nmin for control.',
            DeprecationWarning,
        )
    dfltkwargs = get_kwargs(func)
    if fieldnames is None:
        fieldnames = dfltkwargs.get("fieldnames", None)
    if fieldnames is None:
        fieldnames = get_args(func)
    shape = dfltkwargs.get("shape", (1,))
    dtype = dfltkwargs.get("dtype", "float64")
    default = dfltkwargs.get("default", 0)

    offsets = np.concatenate([[0], np.cumsum(lengths)])

    # Determine chunkedges automatically
    # TODO: very messy and inefficient routine. improve some time.
    # TODO: Set entry_bytes_out
    nbytes_dtype_out = 4  # TODO: hardcode 4 byte output dtype as estimate for now
    entry_nbytes_out = nbytes_dtype_out * np.product(shape)
    list_chunkedges = map_halo_operation_get_chunkedges(
        lengths,
        entry_nbytes_in,
        entry_nbytes_out,
        cpucost_halo=cpucost_halo,
        Nmin=Nmin,
        chunksize_bytes=chunksize_bytes,
    )

    # TODO: Get rid of oindex; only here because have not adjusted code to map_halo_operation_get_chunkedges
    totlength = offsets[-1]
    oindex = np.array(list(list_chunkedges[:, 0]) + [list_chunkedges[-1, -1]])

    new_axis = None
    chunks = np.diff(oindex)
    # TODO: Where does the next line come from? Does not make sense
    # chunks[-1] += lengths.shape[0]
    chunks = [tuple(chunks.tolist())]
    if isinstance(shape, tuple) and shape != (1,):
        chunks += [(s,) for s in shape]
        new_axis = np.arange(1, len(shape) + 1).tolist()

    slcoffsets = offsets[oindex]
    slclengths = np.diff(slcoffsets)

    slcs = [slice(oindex[i], oindex[i + 1]) for i in range(len(oindex) - 1)]
    slcs[-1] = slice(oindex[-2], oindex[-1] + 2)  # hacky! why needed? see TODO above.

    halolengths_in_chunks = [lengths[slc] for slc in slcs]

    d_hic = delayed(halolengths_in_chunks)

    arrs = [arrdict[f][:totlength] for f in fieldnames]
    for i, arr in enumerate(arrs):
        arrchunks = (tuple(slclengths.tolist()),)
        if len(arr.shape) > 1:
            arrchunks = arrchunks + (arr.shape[1:],)
        arrs[i] = arr.rechunk(chunks=arrchunks)
    arrdims = np.array([len(arr.shape) for arr in arrs])
    assert np.all(arrdims == arrdims[0])  # Cannot handle different input dims for now

    drop_axis = []
    if arrdims[0] > 1:
        drop_axis = np.arange(1, arrdims[0])

    calc = da.map_blocks(
        wrap_func_scalar,
        func,
        d_hic,
        *arrs,
        dtype=dtype,
        chunks=chunks,
        new_axis=new_axis,
        drop_axis=drop_axis,
        func_output_shape=shape,
        func_output_dtype=dtype,
        func_output_default=default,
    )

    return calc


groupnames = [
    "PartType0",
    "PartType1",
    "PartType3",
    "PartType4",
    "PartType5",
    "Group",
    "Subhalo",
]
fielddefs = FieldContainer(containers=groupnames)


@fielddefs.register_field("PartType0")
def Temperature(arrs, **kwargs):
    """Compute gas temperature given (ElectronAbundance,InternalEnergy) in [K]."""
    xh = 0.76
    gamma = 5.0 / 3.0

    m_p = 1.672622e-24  # proton mass [g]
    k_B = 1.380650e-16  # boltzmann constant [erg/K]

    UnitEnergy_over_UnitMass = (
        1e10  # standard unit system (TODO: can obtain from snapshot)
    )

    xe = arrs["ElectronAbundance"]
    u_internal = arrs["InternalEnergy"]

    mu = 4 / (1 + 3 * xh + 4 * xh * xe) * m_p
    temp = (gamma - 1.0) * u_internal / k_B * UnitEnergy_over_UnitMass * mu

    return temp
