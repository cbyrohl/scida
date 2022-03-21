import inspect
import re
import astropy.units as u
import numbers
import logging
import numpy as np
import dask
from dask import delayed
import dask.array as da
from numba import njit
import warnings

from ..interface import BaseSnapshot,Selector
from ..helpers_misc import computedecorator, get_args, get_kwargs, parse_humansize
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
    def __init__(self, path, chunksize="auto", catalog=None):
        self.header = {}
        self.config = {}
        self.parameters = {}
        super().__init__(path, chunksize=chunksize)
        self.cosmology = None
        if "HubbleParam" in self.header and "Omega0" in self.header:
            import astropy.units as u
            from astropy.cosmology import FlatLambdaCDM
            h = self.header["HubbleParam"]
            omega0 = self.header["Omega0"]
            self.cosmology = FlatLambdaCDM(H0=100 * h * u.km / u.s / u.Mpc, Om0=omega0)
        self.redshift = np.nan
        try:
            self.redshift = self.header["Redshift"]
        except KeyError: # no redshift attribute
            pass

        self.catalog = catalog
        if catalog is not None:
            self.catalog = BaseSnapshot(catalog, virtualcache=False) # copy catalog for better performance
            for k in self.catalog.data:
                if k not in self.data:
                    self.data[k] = self.catalog.data[k]
                self.catalog.data.derivedfields_kwargs["snap"] = self

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

    def register_field(self, parttype, name=None, construct=True): # TODO: introduce (immediate) construct option later 
        num = partTypeNum(parttype)
        if num == -1: # TODO: all particle species
            key = "all"
            raise NotImplementedError
        elif isinstance(num, int):
            key = "PartType"+str(num)
        else:
            key = parttype
        return super().register_field(key, name=name)

    def add_catalogIDs(self):
        # TODO: make these delayed objects and properly pass into (delayed?) numba functions:
        # https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-repeatedly-putting-large-inputs-into-delayed-calls

        # Get Offsets for particles in snapshot
        glen = self.data["Group"]["GroupLenType"]
        da_halocelloffsets = da.concatenate([np.zeros((1, 6), dtype=np.int64), da.cumsum(glen, axis=0)])
        # remove last entry to match shapematch shape
        self.data["Group"]["GroupOffsetsType"] = da_halocelloffsets[:-1].rechunk(glen.chunks)
        halocelloffsets = da_halocelloffsets.compute()

        subhalogrnr = self.data["Subhalo"]["SubhaloGrNr"].compute()
        subhalocellcounts = self.data["Subhalo"]["SubhaloLenType"].compute()

        for key in self.data:
            if not(key.startswith("PartType")):
                continue
            num = int(key[-1])

            # Group ID
            gidx = self.data[key]["uid"]
            hidx = compute_haloindex(gidx, halocelloffsets[:, num])
            self.data[key]["GroupID"] = hidx
            # Subhalo ID
            gidx = self.data[key]["uid"]
            shcounts, shnumber = get_shcounts_shcells(subhalogrnr, halocelloffsets[:, num].shape[0])
            sidx = compute_subhaloindex(gidx, halocelloffsets[:, num], shnumber, shcounts, subhalocellcounts[:, num])
            self.data[key]["SubhaloID"] = sidx

    @computedecorator
    def map_halo_operation(self, func, chunksize=int(3e7), cpucost_halo=1e4, Nmin=None, chunksize_bytes=None):
        if chunksize is not None:
            warnings.warn('"chunksize" parameter is depreciated and has no effect. Specify Nmin for control.',
                          DeprecationWarning)
        dfltkwargs = get_kwargs(func)
        fieldnames = dfltkwargs.get("fieldnames",None)
        parttype = dfltkwargs.get("parttype","PartType0")
        shape = dfltkwargs.get("shape", (1,))
        dtype = dfltkwargs.get("dtype", "float64")
        default = dfltkwargs.get("default", 0)
        partnum = int(parttype[-1])

        if fieldnames is None:
            fieldnames = get_args(func)

        lengths = self.data["Group"]["GroupLenType"][:, partnum].compute()
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        # Determine chunkedges automatically
        # TODO: very messy and inefficient routine. improve some time.
        # TODO: Set entry_bytes_out
        entry_nbytes_in = np.sum([self.data[parttype][f][0].nbytes for f in fieldnames]) # nbytes
        nbytes_dtype_out = 4 # TODO: hardcode 4 byte output dtype as estimate for now
        entry_nbytes_out = nbytes_dtype_out * np.product(shape)
        list_chunkedges = map_halo_operation_get_chunkedges(lengths, entry_nbytes_in, entry_nbytes_out,
                                                            cpucost_halo=cpucost_halo, Nmin=Nmin,
                                                            chunksize_bytes=chunksize_bytes)

        # TODO: Get rid of oindex; only here because have not adjusted code to map_halo_operation_get_chunkedges
        totlength = offsets[-1]
        oindex = np.array(list(list_chunkedges[:,0])+[list_chunkedges[-1,-1]])

        new_axis = None
        chunks = np.diff(oindex)
        chunks[-1] += lengths.shape[0]
        chunks = [tuple(chunks.tolist())]
        if isinstance(shape,tuple) and shape != (1,):
            chunks += [(s,) for s in shape]
            new_axis = np.arange(1,len(shape)+1).tolist()

        slcoffsets = offsets[oindex]
        slclengths = np.diff(slcoffsets)

        slcs = [slice(oindex[i], oindex[i + 1]) for i in range(len(oindex) - 1)]
        slcs[-1] = slice(oindex[-2], oindex[-1] + 2)  # hacky! why needed? see TODO above.

        halolengths_in_chunks = [lengths[slc] for slc in slcs]

        d_hic = delayed(halolengths_in_chunks)

        arrs = [self.data[parttype][f][:totlength] for f in fieldnames]
        for i, arr in enumerate(arrs):
            arrchunks = (tuple(slclengths.tolist()),)
            if len(arr.shape)>1:
                arrchunks = arrchunks+(arr.shape[1:],)
            arrs[i] = arr.rechunk(chunks=arrchunks)
        arrdims = np.array([len(arr.shape) for arr in arrs])
        assert np.all(arrdims==arrdims[0]) # Cannot handle different input dims for now

        drop_axis = [] 
        if arrdims[0]>1:
            drop_axis = np.arange(1,arrdims[0])

        calc = da.map_blocks(wrap_func_scalar, func, d_hic, *arrs, dtype=dtype, chunks=chunks,new_axis=new_axis,
                             drop_axis=drop_axis,
                             func_output_shape=shape, func_output_dtype=dtype, func_output_default=default)

        return calc

    def add_groupquantity_to_particles(self, name, parttype="PartType0"):
        assert name not in self.data[parttype]  # we simply map the name from Group to Particle for now. Should work (?)
        glen = self.data["Group"]["GroupLenType"]
        da_halocelloffsets = da.concatenate([np.zeros((1, 6), dtype=np.int64), da.cumsum(glen, axis=0)])
        if "GroupOffsetsType" not in self.data["Group"]:
            self.data["Group"]["GroupOffsetsType"] = da_halocelloffsets[:-1].rechunk(glen.chunks)  # remove last entry to match shape
        halocelloffsets = da_halocelloffsets.compute()

        gidx = self.data[parttype]["uid"]
        num = int(parttype[-1])
        hquantity = compute_haloquantity(gidx, halocelloffsets[:, num], self.data["Group"][name])
        self.data[parttype][name] = hquantity


def wrap_func_scalar(func, halolengths_in_chunks, *arrs, block_info=None, block_id=None,
                     func_output_shape=(1,), func_output_dtype="float64",
                     func_output_default=0):
    lengths = halolengths_in_chunks[block_id[0]]
    n = len(lengths)

    offsets = np.cumsum([0] + list(lengths))
    res = []
    for i, o in enumerate(offsets[:-1]):
        if o==offsets[i+1]:
            res.append(func_output_default*np.ones(func_output_shape, dtype=func_output_dtype))
            if func_output_shape==(1,):
                res[-1] = res[-1].item()
            continue
        arrchunks = [arr[o:offsets[i + 1]] for arr in arrs]
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
    res = -1 * np.ones(gidx_count, dtype=np.int32)
    # find initial celloffset
    hidx_idx = np.searchsorted(celloffsets, gidx_start, side="right") - 1
    if hidx_idx+1>=celloffsets.shape[0]:
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
        if (hidx_idx >= celloffsets.shape[0]-1):
            break
        count = celloffsets[hidx_idx+1] - celloffsets[hidx_idx]
        endid = startid + count
    return res


def get_hidx_daskwrap(gidx, halocelloffsets):
    gidx_start = gidx[0]
    gidx_count = gidx.shape[0]
    return get_hidx(gidx_start, gidx_count, halocelloffsets)


def get_haloquantity_daskwrap(gidx, halocelloffsets, valarr):
    hidx = get_hidx_daskwrap(gidx, halocelloffsets)
    result = np.where(hidx>-1, valarr[hidx], -1)
    return result


def compute_haloindex(gidx, halocelloffsets, *args):
    """Computes the halo index for each particle with dask."""
    return da.map_blocks(get_hidx_daskwrap, gidx, halocelloffsets, meta=np.array((), dtype=np.int64))


def compute_haloquantity(gidx, halocelloffsets, hvals, *args):
    """Computes a halo quantity for each particle with dask."""
    return da.map_blocks(get_haloquantity_daskwrap, gidx, halocelloffsets, hvals, meta=np.array((), dtype=hvals.dtype))


@njit
def get_shidx(gidx_start, gidx_count, celloffsets, shnumber, shcounts, shcellcounts):
    res = -1 * np.ones(gidx_count, dtype=np.int32)  # fuzz has negative index.

    # find initial Group we are in
    hidx_start_idx = np.searchsorted(celloffsets, gidx_start, side="right") - 1
    if hidx_start_idx+1>=celloffsets.shape[0]:
        # we are done. Already out of scope of lookup => all unbound gas.
        return res
    celloffset = celloffsets[hidx_start_idx + 1]
    endid = celloffset - gidx_start
    startid = 0

    # find initial subhalo we are in
    hidx = hidx_start_idx
    shcumsum = np.zeros(shcounts[hidx] + 1, dtype=np.int64)
    shcumsum[1:] = np.cumsum(shcellcounts[shnumber[hidx]:shnumber[hidx] + shcounts[hidx]]) # collect halo's subhalo offsets
    shcumsum += celloffsets[hidx_start_idx]
    sidx_start_idx = np.searchsorted(shcumsum, gidx_start, side="right") - 1
    if sidx_start_idx < shcounts[hidx]:
        endid = shcumsum[sidx_start_idx + 1] - gidx_start

    # Now iterate through list.
    cont = True
    while cont and (startid < gidx_count):
        res[startid:endid] = sidx_start_idx if sidx_start_idx+1<shcumsum.shape[0] else -1
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


def compute_subhaloindex(gidx, halocelloffsets,shnumber, shcounts, shcellcounts):
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

def memorycost_limiter(cost_memory, cost_cpu, list_chunkedges, cost_memory_max):
    """If a chunk too memory expensive, split into equal cpu expense operations."""
    list_chunkedges_new = []
    for chunkedges in list_chunkedges:
        slc = slice(*chunkedges)
        totcost_mem = np.sum(cost_memory[slc])
        list_chunkedges_new.append(chunkedges)
        if totcost_mem>cost_memory_max:
            sumcost = cost_cpu[slc].cumsum()
            sumcost /= sumcost[-1]
            idx = slc.start+np.argmin(np.abs(sumcost-0.5))
            if idx==chunkedges[0]:
                idx += 1
            elif idx==chunkedges[-1]:
                idx -= 1
            chunkedges1 = [chunkedges[0], idx]
            chunkedges2 = [idx, chunkedges[1]]
            if idx==chunkedges[0] or idx==chunkedges[1]:
                raise ValueError("This should not happen.")
            list_chunkedges_new.pop()
            list_chunkedges_new += memorycost_limiter(cost_memory, cost_cpu, [chunkedges1], cost_memory_max)
            list_chunkedges_new += memorycost_limiter(cost_memory, cost_cpu, [chunkedges2], cost_memory_max)
    return list_chunkedges_new

def map_halo_operation_get_chunkedges(lengths, entry_nbytes_in, entry_nbytes_out,
                                      cpucost_halo=1.0, Nmin=None, chunksize_bytes = None):
    cpucost_particle = 1.0 # we only care about ratio, so keep particle cost fixed.
    cost = cpucost_particle * lengths + cpucost_halo
    sumcost = cost.cumsum()

    #let's allow a maximal chunksize of 16 times the dask default setting for an individual array [here: multiple]
    if chunksize_bytes is None:
        chunksize_bytes = 16*parse_humansize(dask.config.get('array.chunk-size'))
    cost_memory = entry_nbytes_in*lengths + entry_nbytes_out

    if not np.max(cost_memory)<chunksize_bytes:
        raise ValueError("Some halo requires more memory than allowed (%i allowed, %i requested). Consider overriding chunksize_bytes."%(chunksize_bytes, np.max(cost_memory)))

    N = int(np.ceil(np.sum(cost_memory)/chunksize_bytes))
    N = int(np.ceil(1.3*N)) #  fudge factor
    if Nmin is not None:
        N = max(Nmin,N)
    targetcost = sumcost[-1]/N
    arr = np.diff(sumcost%targetcost)
    idx = [0]+list(np.where(arr<0)[0]+1)
    if len(idx)==N+1:
        idx[-1] = sumcost.shape[0]
    elif len(idx)-N in [0,-1,-2]:
        idx.append(sumcost.shape[0])
    else:
        raise ValueError("Unexpected chunk indices.")
    list_chunkedges = []
    for i in range(len(idx)-1):
        list_chunkedges.append([idx[i], idx[i+1]])

    list_chunkedges = np.asarray(memorycost_limiter(cost_memory, cost, list_chunkedges, chunksize_bytes))

    # make sure we did not lose any halos.
    assert np.all(~(list_chunkedges.flatten()[2:-1:2]-list_chunkedges.flatten()[1:-1:2]).astype(bool))
    return list_chunkedges
