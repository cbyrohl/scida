import copy
import logging
import os
from typing import Dict, List, Optional, Union

import dask
import numpy as np
import pint
from dask import array as da
from dask import delayed
from numba import jit
from numpy.typing import NDArray

from scida.customs.arepo.extra_fields import fielddefs
from scida.customs.arepo.helpers import grp_type_str, part_type_num
from scida.customs.arepo.selector import ArepoSelector
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot
from scida.discovertypes import CandidateStatus, _determine_mixins
from scida.fields import FieldContainer
from scida.helpers_misc import (
    computedecorator,
    get_args,
    get_kwargs,
    map_blocks,
    parse_humansize,
)
from scida.interface import create_datasetclass_with_mixins
from scida.interfaces.mixins import CosmologyMixin, SpatialCartesian3DMixin, UnitMixin
from scida.io import load_metadata

log = logging.getLogger(__name__)


class ArepoSnapshot(SpatialCartesian3DMixin, GadgetStyleSnapshot):
    """
    Dataset class for Arepo snapshots.
    """

    _fileprefix_catalog = "groups"

    def __init__(self, path, chunksize="auto", catalog=None, **kwargs) -> None:
        """
        Initialize an ArepoSnapshot object.

        Parameters
        ----------
        path: str or pathlib.Path
            Path to snapshot, typically a directory containing multiple hdf5 files.
        chunksize:
            Chunksize to use for dask arrays. Can be "auto" to automatically determine chunksize.
        catalog:
            Path to group catalog. If None, the catalog is searched for in the parent directories.
        kwargs:
            Additional keyword arguments.
        """
        self.iscatalog = kwargs.pop("iscatalog", False)
        self.header = {}
        self.config = {}
        # check whether we have a cosmology mixin
        self._defaultunitfiles: List[str] = []
        if hasattr(self, "_mixins") and "cosmology" in self._mixins:
            self._defaultunitfiles = ["units/gadget_cosmological.yaml"]
        self.parameters = {}
        self._grouplengths = {}
        self._subhalolengths = {}
        # not needed for group catalogs as entries are back-to-back there, we will provide a property for this
        self._subhalooffsets = {}
        self.misc = {}  # for storing misc info
        prfx = kwargs.pop("fileprefix", None)
        if prfx is None:
            prfx = self._get_fileprefix(path)
        super().__init__(path, chunksize=chunksize, fileprefix=prfx, **kwargs)

        self.catalog = catalog
        if not self.iscatalog:
            if self.catalog is None:
                self.discover_catalog()
                # try to discover group catalog in parent directories.
            if self.catalog == "none":
                pass  # this string can be set to explicitly disable catalog
            elif self.catalog:
                catalog_cls = kwargs.get("catalog_cls", None)
                cosmological = False
                if hasattr(self, "_mixins") and "cosmology" in self._mixins:
                    cosmological = True
                self.load_catalog(
                    overwrite_cache=kwargs.get("overwrite_cache", False),
                    catalog_cls=catalog_cls,
                    units=self.withunits,
                    cosmological=cosmological,
                )

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
        overwrite_fields = False
        for kfield in fielddefs.fielddefs:
            parttype = fielddefs.fielddefs[kfield].parttype
            if parttype in self.data:
                if overwrite_fields or kfield not in self.data[parttype]:
                    func = fielddefs.fielddefs[kfield].func
                    # verify that we have the dependencies
                    dependencies = fielddefs.fielddefs[kfield].dependencies
                    deps_fulfilled = True
                    for dep in dependencies:
                        if dep not in self.data[parttype]:
                            deps_fulfilled = False
                            break
                    if deps_fulfilled:
                        self.data.register_field(parttype, name=kfield)(func)

    def load_catalog(
        self, overwrite_cache=False, units=False, cosmological=False, catalog_cls=None
    ):
        """
        Load the group catalog.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments passed to the catalog class.
        catalog_cls: type
            Class to use for the catalog. If None, the default catalog class is used.

        Returns
        -------
        None
        """
        virtualcache = False  # copy catalog for better performance
        # fileprefix = catalog_kwargs.get("fileprefix", self._fileprefix_catalog)
        prfx = self._get_fileprefix(self.catalog)

        # explicitly need to create unitaware class for catalog as needed
        # TODO: should just be determined from mixins of parent?
        if catalog_cls is None:
            cls = ArepoCatalog
        else:
            cls = catalog_cls
        withunits = units
        mixins = []
        if withunits:
            mixins += [UnitMixin]

        other_mixins = _determine_mixins(path=self.path)
        mixins += other_mixins
        if cosmological and CosmologyMixin not in mixins:
            mixins.append(CosmologyMixin)

        cls = create_datasetclass_with_mixins(cls, mixins)

        ureg = None
        if hasattr(self, "ureg"):
            ureg = self.ureg

        self.catalog = cls(
            self.catalog,
            overwrite_cache=overwrite_cache,
            virtualcache=virtualcache,
            fileprefix=prfx,
            units=self.withunits,
            ureg=ureg,
            hints=self.hints,
            metadata_raw_parent=self._metadata_raw,
        )
        if "Redshift" in self.catalog.header and "Redshift" in self.header:
            z_catalog = self.catalog.header["Redshift"]
            z_snap = self.header["Redshift"]
            if not np.isclose(z_catalog, z_snap):
                raise ValueError(
                    "Redshift mismatch between snapshot and catalog: "
                    f"{z_snap:.2f} vs {z_catalog:.2f}"
                )

        # merge data
        self.merge_data(self.catalog)

        # first snapshots often do not have groups
        if "Group" in self.catalog.data:
            ngkeys = self.catalog.data["Group"].keys()
            if len(ngkeys) > 0:
                self.add_catalogIDs()

        # merge hints from snap and catalog
        self.merge_hints(self.catalog)

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, **kwargs
    ) -> CandidateStatus:
        """
        Validate a path to use for instantiation of this class.

        Parameters
        ----------
        path: str or pathlib.Path
        args:
        kwargs:

        Returns
        -------
        CandidateStatus
        """
        valid = super().validate_path(path, *args, **kwargs)
        if valid.value > CandidateStatus.MAYBE.value:
            valid = CandidateStatus.MAYBE
        else:
            return valid
        # Arepo has no dedicated attribute to identify such runs.
        # lets just query a bunch of attributes that are present for arepo runs
        metadata_raw = load_metadata(path, **kwargs)
        matchingattrs = True
        matchingattrs &= "Git_commit" in metadata_raw["/Header"]
        # not existent for any arepo run?
        matchingattrs &= "Compactify_Version" not in metadata_raw["/Header"]

        if matchingattrs:
            valid = CandidateStatus.MAYBE

        return valid

    @ArepoSelector()
    def return_data(self) -> FieldContainer:
        """
        Return data object of this snapshot.

        Returns
        -------
        None
        """
        return super().return_data()

    def discover_catalog(self):
        """
        Discover the group catalog given the current path

        Returns
        -------
        None
        """
        p = str(self.path)
        # order of candidates matters. For Illustris "groups" must precede "fof_subhalo_tab"
        candidates = [
            p.replace("snapshot", "group"),
            p.replace("snapshot", "groups"),
            p.replace("snapdir", "groups").replace("snap", "groups"),
            p.replace("snapdir", "groups").replace("snap", "fof_subhalo_tab"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            if candidate == self.path:
                continue
            self.catalog = candidate
            break

    def register_field(self, parttype: str, name: str = None, construct: bool = False):
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
        None
        """
        num = part_type_num(parttype)
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
        None
        """
        # TODO: make these delayed objects and properly pass into (delayed?) numba functions:
        # https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-repeatedly-putting-large-inputs-into-delayed-calls

        maxint = np.iinfo(np.int64).max
        self.misc["unboundID"] = maxint

        # Group ID
        if "Group" not in self.data:  # can happen for empty catalogs
            for key in self.data:
                if not (key.startswith("PartType")):
                    continue
                uid = self.data[key]["uid"]
                self.data[key]["GroupID"] = self.misc["unboundID"] * da.ones_like(
                    uid, dtype=np.int64
                )
                self.data[key]["SubhaloID"] = self.misc["unboundID"] * da.ones_like(
                    uid, dtype=np.int64
                )
            return

        glen = self.data["Group"]["GroupLenType"]
        ngrp = glen.shape[0]
        da_halocelloffsets = da.concatenate(
            [
                np.zeros((1, 6), dtype=np.int64),
                da.cumsum(glen, axis=0, dtype=np.int64),
            ]
        )
        # remove last entry to match shapematch shape
        self.data["Group"]["GroupOffsetsType"] = da_halocelloffsets[:-1].rechunk(
            glen.chunks
        )
        halocelloffsets = da_halocelloffsets.rechunk(-1)

        index_unbound = self.misc["unboundID"]

        for key in self.data:
            if not (key.startswith("PartType")):
                continue
            num = int(key[-1])
            if "uid" not in self.data[key]:
                continue  # can happen for empty containers
            gidx = self.data[key]["uid"]
            hidx = compute_haloindex(
                gidx, halocelloffsets[:, num], index_unbound=index_unbound
            )
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

        shnr_attr = "SubhaloGrNr"
        if shnr_attr not in self.data["Subhalo"]:
            shnr_attr = "SubhaloGroupNr"  # what MTNG does
        if shnr_attr not in self.data["Subhalo"]:
            raise ValueError(
                f"Could not find 'SubhaloGrNr' or 'SubhaloGroupNr' in {self.catalog}"
            )

        subhalogrnr = self.data["Subhalo"][shnr_attr]
        subhalocellcounts = self.data["Subhalo"]["SubhaloLenType"]

        # remove "units" for numba funcs
        if hasattr(subhalogrnr, "magnitude"):
            subhalogrnr = subhalogrnr.magnitude
        if hasattr(subhalocellcounts, "magnitude"):
            subhalocellcounts = subhalocellcounts.magnitude

        grp = self.data["Group"]
        if "GroupFirstSub" not in grp or "GroupNsubs" not in grp:
            # if not provided, we calculate:
            # "GroupFirstSub": First subhalo index for each halo
            # "GroupNsubs": Number of subhalos for each halo
            dlyd = delayed(get_shcounts_shcells)(subhalogrnr, ngrp)
            grp["GroupFirstSub"] = dask.compute(dlyd[1])[0]
            grp["GroupNsubs"] = dask.compute(dlyd[0])[0]

        # remove "units" for numba funcs
        grpfirstsub = grp["GroupFirstSub"]
        if hasattr(grpfirstsub, "magnitude"):
            grpfirstsub = grpfirstsub.magnitude
        grpnsubs = grp["GroupNsubs"]
        if hasattr(grpnsubs, "magnitude"):
            grpnsubs = grpnsubs.magnitude

        for key in self.data:
            if not (key.startswith("PartType")):
                continue
            num = int(key[-1])
            pdata = self.data[key]
            if "uid" not in self.data[key]:
                continue  # can happen for empty containers
            gidx = pdata["uid"]

            # we need to make other dask arrays delayed,
            # map_block does not incorrectly infer output shape from these
            halocelloffsets_dlyd = delayed(halocelloffsets[:, num])
            grpfirstsub_dlyd = delayed(grpfirstsub)
            grpnsubs_dlyd = delayed(grpnsubs)
            subhalocellcounts_dlyd = delayed(subhalocellcounts[:, num])

            sidx = compute_localsubhaloindex(
                gidx,
                halocelloffsets_dlyd,
                grpfirstsub_dlyd,
                grpnsubs_dlyd,
                subhalocellcounts_dlyd,
                index_unbound=index_unbound,
            )

            pdata["LocalSubhaloID"] = sidx

            # reconstruct SubhaloID from Group's GroupFirstSub and LocalSubhaloID
            # should be easier to do it directly, but quicker to write down like this:

            # calculate first subhalo of each halo that a particle belongs to
            self.add_groupquantity_to_particles("GroupFirstSub", parttype=key)
            pdata["SubhaloID"] = pdata["GroupFirstSub"] + pdata["LocalSubhaloID"]
            pdata["SubhaloID"] = da.where(
                pdata["SubhaloID"] == index_unbound, index_unbound, pdata["SubhaloID"]
            )

        # add GroupID and SubhaloID to catalogs/groups themselves
        self.data["Group"]["GroupID"] = self.data["Group"]["uid"]
        self.data["Subhalo"]["SubhaloID"] = self.data["Subhalo"]["uid"]

    @computedecorator
    def map_group_operation(
        self,
        func,
        cpucost_halo=1e4,
        nchunks_min=None,
        chunksize_bytes=None,
        nmax=None,
        idxlist=None,
        objtype="halo",
    ):
        """
        Apply a function to each halo in the catalog.

        Parameters
        ----------
        objtype: str
            Type of object to process. Can be "halo" or "subhalo". Default: "halo"
        idxlist: Optional[np.ndarray]
            List of halo indices to process. If not provided, all halos are processed.
        func: function
            Function to apply to each halo. Must take a dictionary of arrays as input.
        cpucost_halo:
            "CPU cost" of processing a single halo. This is a relative value to the processing time per input particle
            used for calculating the dask chunks. Default: 1e4
        nchunks_min: Optional[int]
            Minimum number of particles in a halo to process it. Default: None
        chunksize_bytes: Optional[int]
        nmax: Optional[int]
            Only process the first nmax halos.

        Returns
        -------
        None
        """
        dfltkwargs = get_kwargs(func)
        fieldnames = dfltkwargs.get("fieldnames", None)
        if fieldnames is None:
            fieldnames = get_args(func)
        parttype = dfltkwargs.get("parttype", "PartType0")
        entry_nbytes_in = np.sum([self.data[parttype][f][0].nbytes for f in fieldnames])
        objtype = grp_type_str(objtype)
        if objtype == "halo":
            lengths = self.get_grouplengths(parttype=parttype)
            offsets = self.get_groupoffsets(parttype=parttype)
        elif objtype == "subhalo":
            lengths = self.get_subhalolengths(parttype=parttype)
            offsets = self.get_subhalooffsets(parttype=parttype)
        else:
            raise ValueError(f"objtype must be 'halo' or 'subhalo', not {objtype}")
        arrdict = self.data[parttype]
        return map_group_operation(
            func,
            offsets,
            lengths,
            arrdict,
            cpucost_halo=cpucost_halo,
            nchunks_min=nchunks_min,
            chunksize_bytes=chunksize_bytes,
            entry_nbytes_in=entry_nbytes_in,
            nmax=nmax,
            idxlist=idxlist,
        )

    def add_groupquantity_to_particles(self, name, parttype="PartType0"):
        """
        Map a quantity from the group catalog to the particles based on a particle's group index.

        Parameters
        ----------
        name: str
            Name of quantity to map
        parttype: str
            Name of particle type

        Returns
        -------
        None
        """
        pdata = self.data[parttype]
        assert (
            name not in pdata
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

        gidx = pdata["uid"]
        num = int(parttype[-1])
        hquantity = compute_haloquantity(
            gidx, halocelloffsets[:, num], self.data["Group"][name]
        )
        pdata[name] = hquantity

    def get_grouplengths(self, parttype="PartType0"):
        """
        Get the lengths, i.e. the total number of particles, of a given type in all halos.

        Parameters
        ----------
        parttype: str
            Name of particle type

        Returns
        -------
        np.ndarray
        """
        pnum = part_type_num(parttype)
        ptype = "PartType%i" % pnum
        if ptype not in self._grouplengths:
            lengths = self.data["Group"]["GroupLenType"][:, pnum].compute()
            if isinstance(lengths, pint.Quantity):
                lengths = lengths.magnitude
            self._grouplengths[ptype] = lengths
        return self._grouplengths[ptype]

    def get_groupoffsets(self, parttype="PartType0"):
        """
        Get the array index offset of the first particle of a given type in each halo.

        Parameters
        ----------
        parttype: str
            Name of particle type

        Returns
        -------
        np.ndarray
        """
        if parttype not in self._grouplengths:
            # need to calculate group lengths first
            self.get_grouplengths(parttype=parttype)
        return self._groupoffsets[parttype]

    @property
    def _groupoffsets(self):
        lengths = self._grouplengths
        offsets = {
            k: np.concatenate([[0], np.cumsum(v)[:-1]]) for k, v in lengths.items()
        }
        return offsets

    def get_subhalolengths(self, parttype="PartType0"):
        """
        Get the lengths, i.e. the total number of particles, of a given type in all subhalos.

        Parameters
        ----------
        parttype: str

        Returns
        -------
        np.ndarray
        """
        pnum = part_type_num(parttype)
        ptype = "PartType%i" % pnum
        if ptype in self._subhalolengths:
            return self._subhalolengths[ptype]
        lengths = self.data["Subhalo"]["SubhaloLenType"][:, pnum].compute()
        if isinstance(lengths, pint.Quantity):
            lengths = lengths.magnitude
        self._subhalolengths[ptype] = lengths
        return self._subhalolengths[ptype]

    def get_subhalooffsets(self, parttype="PartType0"):
        """
        Get the array index offset of the first particle of a given type in each subhalo.

        Parameters
        ----------
        parttype: str

        Returns
        -------
        np.ndarray
        """

        pnum = part_type_num(parttype)
        ptype = "PartType%i" % pnum
        if ptype in self._subhalooffsets:
            return self._subhalooffsets[ptype]  # use cached result
        goffsets = self.get_groupoffsets(ptype)
        shgrnr = self.data["Subhalo"]["SubhaloGrNr"]
        # calculate the index of the first particle for the central subhalo of each subhalos's parent halo
        shoffset_central = goffsets[shgrnr]

        grpfirstsub = self.data["Group"]["GroupFirstSub"]
        shlens = self.get_subhalolengths(ptype)
        shoffsets = np.concatenate([[0], np.cumsum(shlens)[:-1]])

        # particle offset for the first subhalo of each group that a subhalo belongs to
        shfirstshoffset = shoffsets[grpfirstsub[shgrnr]]

        # "LocalSubhaloOffset": particle offset of each subhalo in the parent group
        shoffset_local = shoffsets - shfirstshoffset

        # "SubhaloOffset": particle offset of each subhalo in the simulation
        offsets = shoffset_central + shoffset_local

        self._subhalooffsets[ptype] = offsets

        return offsets

    def grouped(
        self,
        fields: Union[str, da.Array, List[str], Dict[str, da.Array]] = "",
        parttype="PartType0",
        objtype="halo",
    ):
        """
        Create a GroupAwareOperation object for applying operations to groups.

        Parameters
        ----------
        fields: Union[str, da.Array, List[str], Dict[str, da.Array]]
            Fields to pass to the operation. Can be a string, a dask array, a list of strings or a dictionary of dask arrays.
        parttype: str
            Particle type to operate on.
        objtype: str
            Type of object to operate on. Can be "halo" or "subhalo". Default: "halo"

        Returns
        -------
        GroupAwareOperation
        """
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
        elif isinstance(fields, pint.Quantity):
            field_mag = fields.magnitude
            if isinstance(field_mag, da.Array):
                arrdict = dict(daskarr=fields)
                inputfields = [field_mag.name]
            elif isinstance(field_mag, np.ndarray):
                field_u = fields.units
                arrdict = dict(
                    daskarr=da.from_array(field_mag, chunks="auto") * field_u
                )
                inputfields = ["field"]
            else:
                raise ValueError(
                    "Unknown input type '%s' decorated with pint quantity."
                    % type(field_mag)
                )
        elif isinstance(fields, np.ndarray):
            arrdict = dict(daskarr=da.from_array(fields, chunks="auto"))
            inputfields = ["field"]
        elif isinstance(fields, list):
            arrdict = {k: self.data[parttype][k] for k in fields}
            inputfields = fields
        elif isinstance(fields, dict):
            arrdict = {}
            arrdict.update(**fields)
            inputfields = list(arrdict.keys())
        else:
            raise ValueError("Unknown input type '%s'." % type(fields))
        check_shape_plausability = True
        if check_shape_plausability and "Coordinates" in self.data[parttype]:
            shape0_ref = self.data[parttype]["Coordinates"].shape[0]
            for k, arr in arrdict.items():
                if arr.shape[0] != shape0_ref:
                    raise ValueError(
                        f"Shape mismatch: {k} has shape {arr.shape} while Coordinates has shape {shape0_ref}. "
                        f"Check parttype passed to grouped()?"
                    )
        objtype = grp_type_str(objtype)
        if objtype == "halo":
            offsets = self.get_groupoffsets(parttype=parttype)
            lengths = self.get_grouplengths(parttype=parttype)
        elif objtype == "subhalo":
            offsets = self.get_subhalooffsets(parttype=parttype)
            lengths = self.get_subhalolengths(parttype=parttype)
        else:
            raise ValueError("Unknown object type '%s'." % objtype)

        gop = GroupAwareOperation(
            offsets,
            lengths,
            arrdict,
            inputfields=inputfields,
        )
        return gop


class ArepoCatalog(ArepoSnapshot):
    """
    Dataset class for Arepo group catalogs.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an ArepoCatalog object.

        Parameters
        ----------
        args
        kwargs
        """
        kwargs["iscatalog"] = True
        if "fileprefix" not in kwargs:
            kwargs["fileprefix"] = "groups"
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, **kwargs
    ) -> CandidateStatus:
        """
        Validate a path to use for instantiation of this class.

        Parameters
        ----------
        path: str or pathlib.Path
        args:
        kwargs:

        Returns
        -------
        CandidateStatus
        """
        kwargs["fileprefix"] = cls._get_fileprefix(path)
        valid = super().validate_path(path, *args, expect_grp=True, **kwargs)
        return valid


class ChainOps:
    """
    Chain operations together.
    """

    def __init__(self, *funcs):
        """
        Initialize a ChainOps object.

        Parameters
        ----------
        funcs: List[function]
            Functions to chain together.
        """
        self.funcs = funcs
        self.kwargs = get_kwargs(
            funcs[-1]
        )  # so we can pass info from kwargs to map_halo_operation
        if self.kwargs.get("dtype") is None:
            self.kwargs["dtype"] = float

        def chained_call(*args):
            cf = None
            for i, f in enumerate(funcs):
                # first chain element can be multiple fields. treat separately
                if i == 0:
                    cf = f(*args)
                else:
                    cf = f(cf)
            return cf

        self.call = chained_call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class GroupAwareOperation:
    """
    Class for applying operations to groups.
    """

    opfuncs = dict(min=np.min, max=np.max, sum=np.sum, half=lambda x: x[::2])
    finalops = {"min", "max", "sum"}
    __slots__ = (
        "arrs",
        "ops",
        "offsets",
        "lengths",
        "final",
        "inputfields",
        "opfuncs_custom",
    )

    def __init__(
        self,
        offsets: NDArray,
        lengths: NDArray,
        arrs: Dict[str, da.Array],
        ops=None,
        inputfields=None,
    ):
        self.offsets = offsets
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
        """
        Chain another operation to this one.

        Parameters
        ----------
        add_op: str or function
            Operation to add. Can be a string (e.g. "min", "max", "sum") or a function.
        final: bool
            Whether this is the final operation in the chain.

        Returns
        -------
        GroupAwareOperation
        """
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
        """
        Get the minimum value for each group member.
        """
        if field is not None:
            if self.inputfields is not None:
                raise ValueError("Cannot change input field anymore.")
            self.inputfields = [field]
        return self.chain(add_op="min", final=True)

    def max(self, field=None):
        """
        Get the maximum value for each group member.
        """
        if field is not None:
            if self.inputfields is not None:
                raise ValueError("Cannot change input field anymore.")
            self.inputfields = [field]
        return self.chain(add_op="max", final=True)

    def sum(self, field=None):
        """
        Sum the values for each group member.
        """
        if field is not None:
            if self.inputfields is not None:
                raise ValueError("Cannot change input field anymore.")
            self.inputfields = [field]
        return self.chain(add_op="sum", final=True)

    def half(self):
        """
        Half the number of particles in each group member. For testing purposes.

        Returns
        -------
        GroupAwareOperation
        """
        return self.chain(add_op="half", final=False)

    def apply(self, func, final=False):
        """
        Apply a passed function.

        Parameters
        ----------
        func: function
            Function to apply.
        final: bool
            Whether this is the final operation in the chain.

        Returns
        -------
        GroupAwareOperation
        """
        return self.chain(add_op=func, final=final)

    def __copy__(self):
        # overwrite method so that copy holds a new ops list.
        c = type(self)(
            self.offsets,
            self.lengths,
            self.arrs,
            ops=list(self.ops),
            inputfields=self.inputfields,
        )
        return c

    def evaluate(self, nmax=None, idxlist=None, compute=True):
        """
        Evaluate the operation.

        Parameters
        ----------
        nmax: Optional[int]
            Maximum number of halos to process.
        idxlist: Optional[np.ndarray]
            List of halo indices to process. If not provided, (and nmax not set) all halos are processed.
        compute: bool
            Whether to compute the result immediately or return a dask object to compute later.

        Returns
        -------

        """
        # TODO: figure out return type
        # final operations: those that can only be at end of chain
        # intermediate operations: those that can only be prior to end of chain
        funcdict = dict()
        funcdict.update(**self.opfuncs)
        funcdict.update(**self.opfuncs_custom)

        func = ChainOps(*[funcdict[k] for k in self.ops])

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

        res = map_group_operation(
            func,
            self.offsets,
            self.lengths,
            self.arrs,
            fieldnames=fieldnames,
            nmax=nmax,
            idxlist=idxlist,
        )
        if compute:
            res = res.compute()
        return res


def wrap_func_scalar(
    func,
    offsets_in_chunks,
    lengths_in_chunks,
    *arrs,
    block_info=None,
    block_id=None,
    func_output_shape=(1,),
    func_output_dtype="float64",
    fill_value=0,
):
    """
    Wrapper for applying a function to each halo in the passed chunk.
    Parameters
    ----------
    func
    offsets_in_chunks
    lengths_in_chunks
    arrs
    block_info
    block_id
    func_output_shape
    func_output_dtype
    fill_value

    Returns
    -------

    """
    offsets = offsets_in_chunks[block_id[0]]
    lengths = lengths_in_chunks[block_id[0]]

    res = []
    for i, length in enumerate(lengths):
        o = offsets[i]
        if length == 0:
            res.append(fill_value * np.ones(func_output_shape, dtype=func_output_dtype))
            if func_output_shape == (1,):
                res[-1] = res[-1].item()
            continue
        arrchunks = [arr[o : o + length] for arr in arrs]
        res.append(func(*arrchunks))
    return np.array(res)


@jit(nopython=True)
def get_hidx(gidx_start, gidx_count, celloffsets, index_unbound=None):
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
    index_unbound : integer, optional
        The index to use for unbound particles. If None, the maximum integer value
        of the dtype is used.
    """
    dtype = np.int64
    if index_unbound is None:
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


def get_hidx_daskwrap(gidx, halocelloffsets, index_unbound=None):
    gidx_start = gidx[0]
    gidx_count = gidx.shape[0]
    return get_hidx(
        gidx_start, gidx_count, halocelloffsets, index_unbound=index_unbound
    )


def get_haloquantity_daskwrap(gidx, halocelloffsets, valarr):
    hidx = get_hidx_daskwrap(gidx, halocelloffsets)
    dmax = np.iinfo(hidx.dtype).max
    mask = ~((hidx == -1) | (hidx == dmax))
    result = -1.0 * np.ones(hidx.shape, dtype=valarr.dtype)
    result[mask] = valarr[hidx[mask]]
    return result


def compute_haloindex(gidx, halocelloffsets, *args, index_unbound=None):
    """Computes the halo index for each particle with dask."""
    return da.map_blocks(
        get_hidx_daskwrap,
        gidx,
        halocelloffsets,
        index_unbound=index_unbound,
        meta=np.array((), dtype=np.int64),
    )


def compute_haloquantity(gidx, halocelloffsets, hvals, *args):
    """Computes a halo quantity for each particle with dask."""
    units = None
    if hasattr(hvals, "units"):
        units = hvals.units
    res = map_blocks(
        get_haloquantity_daskwrap,
        gidx,
        halocelloffsets,
        hvals,
        meta=np.array((), dtype=hvals.dtype),
        output_units=units,
    )
    return res


@jit(nopython=True)
def get_localshidx(
    gidx_start: int,
    gidx_count: int,
    celloffsets: NDArray[np.int64],
    shnumber,
    shcounts,
    shcellcounts,
    index_unbound=None,
):
    """
    Get the local subhalo index for each particle. This is the subhalo index within each
    halo group. Particles belonging to the central galaxies will have index 0, particles
    belonging to the first satellite will have index 1, etc.
    Parameters
    ----------
    gidx_start
    gidx_count
    celloffsets
    shnumber
    shcounts
    shcellcounts
    index_unbound: integer, optional
        The index to use for unbound particles. If None, the maximum integer value
        of the dtype is used.

    Returns
    -------
    np.ndarray
    """
    dtype = np.int32
    if index_unbound is None:
        index_unbound = np.iinfo(dtype).max
    res = index_unbound * np.ones(gidx_count, dtype=dtype)  # fuzz has negative index.

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


def get_local_shidx_daskwrap(
    gidx: NDArray[np.int64],
    halocelloffsets: NDArray[np.int64],
    shnumber,
    shcounts,
    shcellcounts,
    index_unbound=None,
) -> np.ndarray:
    gidx_start = int(gidx[0])
    gidx_count = gidx.shape[0]
    res = get_localshidx(
        gidx_start,
        gidx_count,
        halocelloffsets,
        shnumber,
        shcounts,
        shcellcounts,
        index_unbound=index_unbound,
    )
    return res


def compute_localsubhaloindex(
    gidx, halocelloffsets, shnumber, shcounts, shcellcounts, index_unbound=None
) -> da.Array:
    """
    Compute the local subhalo index for each particle with dask.
    The local subhalo index is the index of the subhalo within each halo,
    starting at 0 for the central subhalo.

    Parameters
    ----------
    gidx
    halocelloffsets
    shnumber
    shcounts
    shcellcounts
    index_unbound

    Returns
    -------

    """
    res = da.map_blocks(
        get_local_shidx_daskwrap,
        gidx,
        halocelloffsets,
        shnumber,
        shcounts,
        shcellcounts,
        index_unbound=index_unbound,
        meta=np.array((), dtype=np.int64),
    )
    return res


@jit(nopython=True)
def get_shcounts_shcells(SubhaloGrNr, hlength):
    """
    Returns the id of the first subhalo and count of subhalos per halo.

    Parameters
    ----------
    SubhaloGrNr: np.ndarray
        The group identifier that each subhalo belongs to respectively
    hlength: int
        The number of halos in the snapshot

    Returns
    -------
    shcounts: np.ndarray
        The number of subhalos per halo
    shnumber: np.ndarray
        The index of the first subhalo per halo
    """
    shcounts = np.zeros(hlength, dtype=np.int32)  # number of subhalos per halo
    shnumber = np.zeros(hlength, dtype=np.int32)  # index of first subhalo per halo
    i = 0
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


def map_group_operation_get_chunkedges(
    lengths,
    entry_nbytes_in,
    entry_nbytes_out,
    cpucost_halo=1.0,
    nchunks_min=None,
    chunksize_bytes=None,
):
    """
    Compute the chunking of a halo operation.

    Parameters
    ----------
    lengths: np.ndarray
        The number of particles per halo.
    entry_nbytes_in
    entry_nbytes_out
    cpucost_halo
    nchunks_min
    chunksize_bytes

    Returns
    -------
    None
    """
    cpucost_particle = 1.0  # we only care about ratio, so keep particle cost fixed.
    cost = cpucost_particle * lengths + cpucost_halo
    sumcost = cost.cumsum()

    # let's allow a maximal chunksize of 16 times the dask default setting for an individual array [here: multiple]
    if chunksize_bytes is None:
        chunksize_bytes = 16 * parse_humansize(dask.config.get("array.chunk-size"))
    cost_memory = entry_nbytes_in * lengths + entry_nbytes_out

    if not np.max(cost_memory) < chunksize_bytes:
        raise ValueError(
            "Some halo requires more memory than allowed (%i allowed, %i requested). Consider overriding "
            "chunksize_bytes." % (chunksize_bytes, np.max(cost_memory))
        )

    nchunks = int(np.ceil(np.sum(cost_memory) / chunksize_bytes))
    nchunks = int(np.ceil(1.3 * nchunks))  # fudge factor
    if nchunks_min is not None:
        nchunks = max(nchunks_min, nchunks)
    targetcost = sumcost[-1] / nchunks  # chunk target cost = total cost / nchunks

    arr = np.diff(sumcost % targetcost)  # find whenever exceeding modulo target cost
    idx = [0] + list(np.where(arr < 0)[0] + 1)
    if idx[-1] != sumcost.shape[0]:
        idx.append(sumcost.shape[0])
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


def map_group_operation(
    func,
    offsets,
    lengths,
    arrdict,
    cpucost_halo=1e4,
    nchunks_min: Optional[int] = None,
    chunksize_bytes: Optional[int] = None,
    entry_nbytes_in: Optional[int] = 4,
    fieldnames: Optional[List[str]] = None,
    nmax: Optional[int] = None,
    idxlist: Optional[np.ndarray] = None,
) -> da.Array:
    """
    Map a function to all halos in a halo catalog.
    Parameters
    ----------
    idxlist: Optional[np.ndarray]
        Only process the halos with these indices.
    nmax: Optional[int]
        Only process the first nmax halos.
    func
    offsets: np.ndarray
        Offset of each group in the particle catalog.
    lengths: np.ndarray
        Number of particles per halo.
    arrdict
    cpucost_halo
    nchunks_min: Optional[int]
        Lower bound on the number of halos per chunk.
    chunksize_bytes
    entry_nbytes_in
    fieldnames

    Returns
    -------
    None
    """
    if isinstance(func, ChainOps):
        dfltkwargs = func.kwargs
    else:
        dfltkwargs = get_kwargs(func)
    if fieldnames is None:
        fieldnames = dfltkwargs.get("fieldnames", None)
    if fieldnames is None:
        fieldnames = get_args(func)
    units = dfltkwargs.get("units", None)
    shape = dfltkwargs.get("shape", None)
    dtype = dfltkwargs.get("dtype", "float64")
    fill_value = dfltkwargs.get("fill_value", 0)

    if idxlist is not None and nmax is not None:
        raise ValueError("Cannot specify both idxlist and nmax.")

    lengths_all = lengths
    offsets_all = offsets
    if len(lengths) == len(offsets):
        # the offsets array here is one longer here, holding the total number of particles in the last halo.
        offsets_all = np.concatenate([offsets_all, [offsets_all[-1] + lengths[-1]]])

    if nmax is not None:
        lengths = lengths[:nmax]
        offsets = offsets[:nmax]

    if idxlist is not None:
        # make sure idxlist is sorted and unique
        if not np.all(np.diff(idxlist) > 0):
            raise ValueError("idxlist must be sorted and unique.")
        # make sure idxlist is within range
        if np.min(idxlist) < 0 or np.max(idxlist) >= lengths.shape[0]:
            raise ValueError(
                "idxlist elements must be in [%i, %i), but covers range [%i, %i]."
                % (0, lengths.shape[0], np.min(idxlist), np.max(idxlist))
            )
        offsets = offsets[idxlist]
        lengths = lengths[idxlist]

    if len(lengths) == len(offsets):
        # the offsets array here is one longer here, holding the total number of particles in the last halo.
        offsets = np.concatenate([offsets, [offsets[-1] + lengths[-1]]])

    # shape/units inference
    infer_shape = shape is None or (isinstance(shape, str) and shape == "auto")
    infer_units = units is None
    infer = infer_shape or infer_units
    if infer:
        # attempt to determine shape.
        if infer_shape:
            log.debug(
                "No shape specified. Attempting to determine shape of func output."
            )
        if infer_units:
            log.debug(
                "No units specified. Attempting to determine units of func output."
            )
        arrs = [arrdict[f][:1].compute() for f in fieldnames]
        # remove units if present
        # arrs = [arr.magnitude if hasattr(arr, "magnitude") else arr for arr in arrs]
        # arrs = [arr.magnitude for arr in arrs]
        dummyres = None
        try:
            dummyres = func(*arrs)
        except Exception as e:  # noqa
            log.warning("Exception during shape/unit inference: %s." % str(e))
        if dummyres is not None:
            if infer_units and hasattr(dummyres, "units"):
                units = dummyres.units
            log.debug("Shape inference: %s." % str(shape))
        if infer_units and dummyres is None:
            units_present = any([hasattr(arr, "units") for arr in arrs])
            if units_present:
                log.warning("Exception during unit inference. Assuming no units.")
        if dummyres is None and infer_shape:
            # due to https://github.com/hgrecco/pint/issues/1037 innocent np.array operations on unit scalars can fail.
            # we can still attempt to infer shape by removing units prior to calling func.
            arrs = [arr.magnitude if hasattr(arr, "magnitude") else arr for arr in arrs]
            try:
                dummyres = func(*arrs)
            except Exception as e:  # noqa
                # no more logging needed here
                pass
        if dummyres is not None and infer_shape:
            if np.isscalar(dummyres):
                shape = (1,)
            else:
                shape = dummyres.shape
        if infer_shape and dummyres is None and shape is None:
            log.warning("Exception during shape inference. Using shape (1,).")
            shape = ()
    # unit inference

    # Determine chunkedges automatically
    # TODO: very messy and inefficient routine. improve some time.
    # TODO: Set entry_bytes_out
    nbytes_dtype_out = 4  # TODO: hardcode 4 byte output dtype as estimate for now
    entry_nbytes_out = nbytes_dtype_out * np.prod(shape)

    # list_chunkedges refers to bounds of index intervals to be processed together
    # if idxlist is specified, then these indices do not have to refer to group indices.
    # if idxlist is given, we enforce that particle data is contiguous
    # by putting each idx from idxlist into its own chunk.
    # in the future, we should optimize this
    if idxlist is not None:
        list_chunkedges = [[idx, idx + 1] for idx in np.arange(len(idxlist))]
    else:
        list_chunkedges = map_group_operation_get_chunkedges(
            lengths,
            entry_nbytes_in,
            entry_nbytes_out,
            cpucost_halo=cpucost_halo,
            nchunks_min=nchunks_min,
            chunksize_bytes=chunksize_bytes,
        )

    minentry = offsets[0]
    maxentry = offsets[-1]  # the last particle that needs to be processed

    # chunks specify the number of groups in each chunk
    chunks = [tuple(np.diff(list_chunkedges, axis=1).flatten())]
    # need to add chunk information for additional output axes if needed
    new_axis = None
    if isinstance(shape, tuple) and shape != (1,):
        chunks += [(s,) for s in shape]
        new_axis = np.arange(1, len(shape) + 1).tolist()

    # slcoffsets = [offsets[chunkedge[0]] for chunkedge in list_chunkedges]
    # the actual length of relevant data in each chunk
    slclengths = [
        offsets[chunkedge[1]] - offsets[chunkedge[0]] for chunkedge in list_chunkedges
    ]
    if idxlist is not None:
        # the chunk length to be fed into map_blocks
        tmplist = np.concatenate([idxlist, [len(lengths_all)]])
        slclengths_map = [
            offsets_all[tmplist[chunkedge[1]]] - offsets_all[tmplist[chunkedge[0]]]
            for chunkedge in list_chunkedges
        ]
        slcoffsets_map = [
            offsets_all[tmplist[chunkedge[0]]] for chunkedge in list_chunkedges
        ]
        slclengths_map[0] = slcoffsets_map[0]
        slcoffsets_map[0] = 0
    else:
        slclengths_map = slclengths

    slcs = [slice(chunkedge[0], chunkedge[1]) for chunkedge in list_chunkedges]
    offsets_in_chunks = [offsets[slc] - offsets[slc.start] for slc in slcs]
    lengths_in_chunks = [lengths[slc] for slc in slcs]
    d_oic = delayed(offsets_in_chunks)
    d_hic = delayed(lengths_in_chunks)

    arrs = [arrdict[f][minentry:maxentry] for f in fieldnames]
    for i, arr in enumerate(arrs):
        arrchunks = ((tuple(slclengths)),)
        if len(arr.shape) > 1:
            arrchunks = arrchunks + (arr.shape[1:],)
        arrs[i] = arr.rechunk(chunks=arrchunks)
    arrdims = np.array([len(arr.shape) for arr in arrs])

    assert np.all(arrdims == arrdims[0])  # Cannot handle different input dims for now

    drop_axis = []
    if arrdims[0] > 1:
        drop_axis = np.arange(1, arrdims[0])

    if dtype is None:
        raise ValueError(
            "dtype must be specified, dask will not be able to automatically determine this here."
        )

    calc = map_blocks(
        wrap_func_scalar,
        func,
        d_oic,
        d_hic,
        *arrs,
        dtype=dtype,
        chunks=chunks,
        new_axis=new_axis,
        drop_axis=drop_axis,
        func_output_shape=shape,
        func_output_dtype=dtype,
        fill_value=fill_value,
        output_units=units,
    )

    return calc
