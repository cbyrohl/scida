import re
import astropy.units as u
import numbers
import logging
from ..interface import BaseSnapshot

from .arepo_units import unitstr_arepo,get_unittuples_from_TNGdocs_groups, \
    get_unittuples_from_TNGdocs_particles, get_unittuples_from_TNGdocs_subhalos


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

        if isinstance(self.header["BoxSize"], float):
            self.boxsize[:] = self.header["BoxSize"]
        else:
            # Have not thought about non-cubic cases yet.
            raise NotImplementedError

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
