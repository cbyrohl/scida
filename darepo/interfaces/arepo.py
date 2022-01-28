import inspect
import re
import astropy.units as u
import numbers
import logging
import numpy as np
from dask import delayed
import dask.array as da
from numba import njit

from ..interface import BaseSnapshot,Selector
from ..helpers_misc import computedecorator, get_args, get_kwargs
from .arepo_units import unitstr_arepo,get_unittuples_from_TNGdocs_groups, \
    get_unittuples_from_TNGdocs_particles, get_unittuples_from_TNGdocs_subhalos


class ArepoSelector(Selector):
    def __init__(self):
        super().__init__()
        self.keys = ["haloID"]
    def prepare(self,*args,**kwargs):
        snap = args[0]
        if snap.catalog is None:
            raise ValueError("Cannot select for haloID without catalog loaded.")
        haloID = kwargs.get("haloID",None)
        if haloID is None:
            return
        lengths = self.data_backup["Group"]["GroupLenType"][haloID,:].compute()
        offsets = self.data_backup["Group"]["GroupOffsetsType"][haloID,:].compute()
        for p in self.data_backup:
            splt = p.split("PartType")
            if len(splt)==1:
                for k,v in self.data_backup[p].items():
                    self.data[p][k] = v
            else:
                pnum = int(splt[1])
                offset = offsets[pnum]
                length = lengths[pnum]
                for k,v in self.data_backup[p].items():
                    self.data[p][k] = v[offset:offset+length]
        snap.data = self.data


class ArepoSnapshot(BaseSnapshot):
    def __init__(self, path, catalog=None):
        self.header = {}
        self.config = {}
        self.parameters = {}
        super().__init__(path)

        self.catalog = catalog
        if catalog is not None:
            self.catalog = BaseSnapshot(catalog)
            for k in self.catalog.data:
                if k not in self.data:
                    self.data[k] = self.catalog.data[k]

        # Add halo/subhalo IDs to particles (if catalogs available)
        if self.catalog is not None:
            self.add_catalogIDs()

        # Add certain metadata to class __dict__
        if isinstance(self.header["BoxSize"], float):
            self.boxsize[:] = self.header["BoxSize"]
        else:
            # Have not thought about non-cubic cases yet.
            raise NotImplementedError

    @ArepoSelector()
    def return_data(self):
        return super().return_data()

    def register_field(self, parttype, name=None, construct=True):
        num = partTypeNum(parttype)
        if num == -1: # TODO: all particle species
            key = "all"
            raise NotImplementedError
        elif isinstance(num, int):
            key = "PartType"+str(num)
        else:
            key = parttype
        return super().register_field(key, name=name, construct=construct)

    def add_catalogIDs(self):
        # Get Offsets for particles in snapshot
        self.data["Group"]["GroupOffsetsType"] = da.concatenate([np.zeros((1,6),dtype=np.int64),da.cumsum(self.data["Group"]["GroupLenType"],axis=0)[:-1]])


        # TODO: make these delayed objects and properly pass into (delayed?) numba functions:
        # https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-repeatedly-putting-large-inputs-into-delayed-calls
        halocelloffsets = self.data["Group"]["GroupOffsetsType"].compute()
        halocellcounts = self.data["Group"]["GroupLenType"].compute()
        subhalogrnr = self.data["Subhalo"]["SubhaloGrNr"].compute()
        subhalocellcounts = self.data["Subhalo"]["SubhaloLenType"].compute()


        for key in self.data:
            if not(key.startswith("PartType")):
                continue
            num = int(key[-1])

            # Group ID
            gidx = self.data[key]["uid"]
            hidx = compute_haloindex(gidx, halocelloffsets[:, num], halocellcounts[:,num])
            self.data[key]["GroupID"] = hidx
            # Subhalo ID
            gidx = self.data[key]["uid"]
            shcounts, shnumber = get_shcounts_shcells(subhalogrnr, halocelloffsets[:, num].shape[0])
            sidx = compute_subhaloindex(gidx, halocelloffsets[:, num], shnumber, shcounts, subhalocellcounts[:, num])
            self.data[key]["SubhaloID"] = sidx

    @computedecorator
    def map_halo_operation(self, func, chunksize=int(3e7)):
        # TODO: auto chunksize
        dfltkwargs = get_kwargs(func)
        fieldnames = dfltkwargs.get("fieldnames",None)
        parttype = dfltkwargs.get("parttype","PartType0")
        partnum = int(parttype[-1])

        if fieldnames is None:
            fieldnames = get_args(func)

        lengths = self.data["Group"]["GroupLenType"][:, partnum].compute()
        offsets = self.data["Group"]["GroupOffsetsType"][:, partnum].compute()

        totlength = offsets[-1] - 1  # why this? instead of "offsets[-1]+lengths[-1]"?
        chunksize = max(chunksize,np.max(lengths))
        nbins = int(np.ceil(totlength / chunksize))
        bins = np.linspace(0, totlength, nbins + 1, dtype=int)
        oindex = np.searchsorted(offsets, bins, side="right") - 1
        assert np.all(np.diff(oindex) != 0)  # chunk smaller than largest halo. Increase chunk size

        chunks = np.diff(oindex)
        chunks[-1] += 1  # TODO: Why is this manually needed?
        chunks = [tuple(chunks.tolist())]

        slcoffsets = offsets[oindex]
        slcoffsets[-1] = totlength
        slclengths = np.diff(slcoffsets)

        slcs = [slice(oindex[i], oindex[i + 1]) for i in range(len(oindex) - 1)]
        slcs[-1] = slice(oindex[-2], oindex[-1] + 1)  # hacky! why needed?

        halolengths_in_chunks = [lengths[slc] for slc in slcs]

        d_hic = delayed(halolengths_in_chunks)

        arrs = [self.data[parttype][f][:totlength] for f in fieldnames]
        for i, arr in enumerate(arrs):
            arrs[i] = arr.rechunk(chunks=[tuple(slclengths.tolist())])

        calc = da.map_blocks(wrap_func_scalar, func, d_hic, *arrs, dtype=arr.dtype, chunks=chunks)

        return calc





def wrap_func_scalar(func, halolengths_in_chunks, *arrs, block_id=None):
    lengths = halolengths_in_chunks[block_id[0]]
    n = len(lengths)

    offsets = np.cumsum([0] + list(lengths))
    res = []
    for i, o in enumerate(offsets[:-1]):
        arrchunks = [arr[o:offsets[i + 1]] for arr in arrs]
        # assert len(np.unique(arrchunk))==1
        res.append(func(*arrchunks))
    return np.array(res)




class ArepoSnapshotWithUnits(ArepoSnapshot):
    def __init__(self, path, catalog=None):
        super().__init__(path, catalog=catalog)
        # unitdict = get_units_from_AREPOdocs(unitstr_arepo, self.header) # from AREPO public docs
        unitdict = get_units_from_TNGdocs(self.header) # from TNG public docs
        for k in self.data:
            if k.startswith("PartType"):
                dct = unitdict["particles"]
            elif k=="Group":
                dct = unitdict["groups"]
            elif k=="Subhalo":
                dct = unitdict["subhalos"]
            else:
                dct = unitdict[k]
            for name in self.data[k]:
                if name in dct:
                    self.data[k][name] = self.data[k][name]*dct[name]
                else:
                    logging.info("No units for '%s'"%name)


@njit
def get_hidx(gidx_start, gidx_count, celloffsets, cellcounts):
    """Get halo index of a given cell

    Parameters
    ----------
    gidx_start: integer
        The first unique integer ID for the first particle
    gidx_count: integer
        The amount of halo indices we are querying after "gidx_start"
    celloffsets : array
        An array holding the starting cell offset for each halo
    cellcounts : array
        The count of cells for each halo
    """
    res = -1 * np.ones(gidx_count, dtype=np.int32)
    # find initial celloffset
    hidx_start_idx = np.searchsorted(celloffsets, gidx_start, side="right") - 1
    startid = 0
    endid = celloffsets[hidx_start_idx + 1] - gidx_start
    # Now iterate through list.
    while startid < gidx_count:
        res[startid:endid] = hidx_start_idx
        hidx_start_idx += 1
        startid = endid
        if (hidx_start_idx >= celloffsets.shape[0]):
            break
        count = cellcounts[hidx_start_idx]
        endid = startid + count
    return res


def get_hidx_daskwrap(gidx, halocelloffsets, halocellcounts):
    gidx_start = gidx[0]
    gidx_count = gidx.shape[0]
    return get_hidx(gidx_start, gidx_count, halocelloffsets, halocellcounts)


def compute_haloindex(gidx, halocelloffsets, halocellcounts, *args):
    return da.map_blocks(get_hidx_daskwrap, gidx, halocelloffsets, halocellcounts, meta=np.array((), dtype=np.int64))


@njit
def get_shidx(gidx_start, gidx_count, celloffsets, shnumber, shcounts, shcellcounts):
    res = -1 * np.ones(gidx_count, dtype=np.int32)  # fuzz has negative index.
    # find initial Group we are in
    hidx_start_idx = np.searchsorted(celloffsets, gidx_start, side="right") - 1
    startid = 0
    if (hidx_start_idx + 1 >= celloffsets.shape[0]):
        # Cells not in halos anymore. Thus already return.
        # TODO Possible bug for last halo of catalog (if it would contain cells in subhalo) (?)
        return res
    endid = celloffsets[hidx_start_idx + 1] - gidx_start

    hidx = hidx_start_idx
    shcumsum = np.zeros(shcounts[hidx] + 1, dtype=np.int64)
    shcumsum[1:] = np.cumsum(shcellcounts[shnumber[hidx]:shnumber[hidx] + shcounts[hidx]])
    shcumsum += celloffsets[hidx_start_idx]
    sidx_start_idx = np.searchsorted(shcumsum, gidx_start, side="right") - 1
    if sidx_start_idx < shcounts[hidx]:
        endid = shcumsum[sidx_start_idx + 1] - gidx_start
    # Now iterate through list.
    cont = True
    while cont:
        if (startid >= gidx_count):
            break
        res[startid:endid] = sidx_start_idx
        sidx_start_idx += 1
        if sidx_start_idx < shcounts[hidx_start_idx]:
            count = shcumsum[sidx_start_idx + 1] - shcumsum[sidx_start_idx]
            startid = endid
        else:
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
            if (hidx_start_idx >= celloffsets.shape[0] - 1):
                startid = gidx_count
            else:
                count = celloffsets[hidx_start_idx + 1] - celloffsets[hidx_start_idx]
                if hidx < shcounts.shape[0]:
                    shcumsum = np.zeros(shcounts[hidx] + 1, dtype=np.int64)
                    shcumsum[1:] = np.cumsum(shcellcounts[shnumber[hidx]:shnumber[hidx] + shcounts[hidx]])
                    shcumsum += celloffsets[hidx_start_idx]
                    sidx_start_idx = 0
                    if sidx_start_idx < shcounts[hidx]:
                        count = shcumsum[sidx_start_idx + 1] - shcumsum[sidx_start_idx]
                    startid = celloffsets[hidx_start_idx] - gidx_start
        endid = startid + count
    return res


def get_shidx_daskwrap(gidx, halocelloffsets, shnumber, shcounts, shcellcounts):
    gidx_start = gidx[0]
    gidx_count = gidx.shape[0]
    return get_shidx(gidx_start, gidx_count, halocelloffsets, shnumber, shcounts, shcellcounts)


def compute_subhaloindex(gidx, halocelloffsets, shnumber, shcounts, shcellcounts):
    return da.map_blocks(get_shidx_daskwrap, gidx, halocelloffsets, shnumber, shcounts, shcellcounts,
                         meta=np.array((), dtype=np.int64))


@njit
def get_shcounts_shcells(SubhaloGrNr, hlength):
    """Returns the number offset and count of subhalos per halo."""
    shcounts = np.zeros(hlength, dtype=np.int32)
    shnumber = np.zeros(hlength, dtype=np.int32)
    i = 0
    hid = 0
    hid_old = 0
    while True:
        if i == SubhaloGrNr.shape[0]:
            break
        hid = SubhaloGrNr[i]
        if (hid == hid_old):
            shcounts[hid] += 1
        else:
            shnumber[hid] = i
            shcounts[hid] += 1
            hid_old = hid
        i += 1
    return shcounts, shnumber

def get_units_from_AREPOdocs(unitstr, codeunitdict):
    # first, extract list of unit tuples (consisting of unit as string and exponent)
    unittupledict = {}
    typemapping = {"Field name": "particles", "Group field name": "groups", "Subhalo field name": "subhalos"}
    pattern = re.compile("\s*(?:ext)*\{*([a-zA-Z]+)\}*\^*\{*([0-9.-]*)\}*")
    lines = unitstr.split("\n")
    list_started = False
    typekey = None
    for line in lines:
        if line.startswith("| "):
            fieldentry = [l.strip() for l in line.split("|")[1:-1]]
        else:
            continue
        name, blkid, units, desc = fieldentry
        if " name" in name:
            typekey = typemapping[name]
            if typekey not in unittupledict:
                unittupledict[typekey] = {}
            continue
        if len(units) == 0:  # no units
            unittupledict[typekey][name] = []
            continue

        unittuples = [ut for ut in pattern.findall(units) if ut[0] != "math"]
        unittupledict[typekey][name] = unittuples
        
    # next convert tuples to units
    unitdict = {}
    for k,dct in unittupledict.items():
        unitdict[k] = {}
        for name,uts in dct.items():
            units = get_units_from_unittuples(uts,codeunitdict)
            unitdict[k][name] = units
    return unitdict


def get_units_from_unittuples(unittuples, codeunitdict):
    # require cgs unitlengths for now
    assert 'UnitLength_in_cm' in codeunitdict
    assert 'UnitMass_in_g' in codeunitdict
    assert 'UnitVelocity_in_cm_per_s' in codeunitdict
    assert 'HubbleParam' in codeunitdict
    assert 'Redshift' in codeunitdict
    cudict = {}
    cudict["UnitLength"] = codeunitdict["UnitLength_in_cm"]*u.cm
    cudict["UnitMass"] = codeunitdict["UnitMass_in_g"]*u.g 
    cudict["UnitVelocity"] = codeunitdict["UnitVelocity_in_cm_per_s"]*u.cm/u.s
    # Pressure: M/L/T^2 = M/L^3 * V^2
    cudict["UnitPressure"] = cudict["UnitMass"]/cudict["UnitLength"]**3 * cudict["UnitVelocity"]**2
    cudict["h"] = codeunitdict["HubbleParam"]
    cudict["a"] = 1.0/(1.0+codeunitdict["Redshift"])
    cudict["ckpc"] = cudict["a"]*u.kpc # as we express everything physical => multiply with a
    units = 1.0
    for ut in unittuples:
        if isinstance(ut,tuple):
            ut = list(ut)
            if len(ut)==1:
                ut = ut + [1]
            elif isinstance(ut[1],str) and len(ut[1])==0:
                ut[1] = 1.0
        if isinstance(ut[0],numbers.Number):
            units *= ut[0]**(float(ut[1]))
        else:
            try:
                units *= cudict[ut[0]]**(float(ut[1]))
            except KeyError:
                units *= u.__dict__[ut[0]]**(float(ut[1]))
    return units


def get_units_from_TNGdocs(codeunitdict):
    units = dict(particles={},groups={},subhalos={})
    unittuples = get_unittuples_from_TNGdocs_particles()
    units["particles"] = {k: get_units_from_unittuples(ut, codeunitdict) for k, ut in unittuples.items()}
    unittuples = get_unittuples_from_TNGdocs_groups()
    units["groups"] = {k: get_units_from_unittuples(ut, codeunitdict) for k, ut in unittuples.items()}
    unittuples = get_unittuples_from_TNGdocs_subhalos()
    units["subhalos"] = {k: get_units_from_unittuples(ut, codeunitdict) for k, ut in unittuples.items()}
    return units







def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['dmlowres']:
        return 2 # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes','black']:
        return 5
    if str(partType).lower() in ["all"]:
        return -1
