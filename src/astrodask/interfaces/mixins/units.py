import importlib.resources
import logging
import os

import numpy as np
import yaml
from pint import UnitRegistry

from astrodask.config import get_config
from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import sprint

log = logging.getLogger(__name__)


def extract_units_from_attrs(attrs, require=False, mode="cgs", ureg=None):
    """
    Extract the units from given attributes
    """
    assert ureg is not None, "Always require passing registry now."
    udict = {}
    if mode == "cgs":
        udict["length"] = ureg("cm")
        udict["mass"] = ureg("g")
        udict["velocity"] = ureg("cm/s")
        udict["time"] = ureg("s")
    elif mode == "code":
        raise NotImplementedError("TBD")
    else:
        raise KeyError("Unknown unit mode '%s'." % mode)
    if "h" in ureg:
        udict["h"] = ureg("h")
    # the common mode is to expect some hints how to convert to cgs
    # get the conversion factor
    cgskey = [k for k in attrs if "cgs" in k.lower()]
    if "cgsunits" in cgskey:
        cgskey.remove("cgsunits")  # these are the units not the normalization factor
    assert len(cgskey) == 1
    cgskey = cgskey[0]
    cgsfactor = attrs[cgskey]
    # get dimensions
    unit = cgsfactor
    if isinstance(unit, np.ndarray):
        assert len(unit) == 1
        unit = unit[0]
    if unit == 0.0:
        unit = 1.0  # zero makes no sense.
    # TODO: Missing a scaling!
    # TODO: Missing h scaling!
    ukeys = ["length", "mass", "velocity", "time", "h"]
    if any([k + "_scaling" in attrs.keys() for k in ukeys]):  # like TNG
        for k in ukeys:
            aname = k + "_scaling"
            unit *= udict[k] ** attrs.get(aname, 0.0)
    elif "Conversion factor" in attrs.keys():  # like SWIFT
        ustr = str(attrs["Conversion factor"])
        ustr = ustr.split("[")[-1].split("]")[0]
        if ustr.strip() == "-":  # no units, done
            return unit
        unit *= ureg(ustr)
    elif "cgsunits" in attrs.keys():  # like EAGLE
        if attrs["cgsunits"] is not None:  # otherwise, this field has no units
            unitstr = attrs["cgsunits"]
            unit *= ureg(unitstr)
    elif require:
        raise ValueError("Could not find units.")
    return unit


def get_unithints_fromfile(resource):
    """Find and load of the unit file"""
    # order:
    # 1. absolute path?
    path = os.path.expanduser(resource)
    if os.path.isabs(path):
        with open(path, "r") as file:
            unithints = yaml.safe_load(file)["fields"]
        return unithints
    bpath = os.path.expanduser("~/.config/astrodask/units")
    path = os.path.join(bpath, resource)
    # 2. non-absolute path?
    # 2.1. check ~/.config/astrodask/units/
    if os.path.isfile(path):
        with open(path, "r") as file:
            unithints = yaml.safe_load(file)["fields"]
        return unithints
    # 2.2 check astrodask package resource units/
    resource_path = "astrodask.configfiles.units"
    with importlib.resources.path(resource_path, resource) as fp:
        with open(fp, "r") as file:
            unithints = yaml.safe_load(file)["fields"]
    return unithints


class UnitMixin(Mixin):
    def __init__(self, *args, **kwargs):
        self.units = {}
        self.data = {}
        self._metadata_raw = {}
        ureg = UnitRegistry()  # before unit
        # TODO: Define the default unit registry somewhere else
        ureg.define("Msun = 1.98847e33 * g")
        self.unitregistry = ureg
        super().__init__(*args, **kwargs)

        # discover units
        units = kwargs.pop("units")
        unithints = {}
        unitfile = ""
        if "dsname" in self.hints:
            c = get_config()
            dsprops = c["data"][self.hints["dsname"]]
            unitfile = dsprops.get("unitfile", "")
        unitfile = kwargs.pop("unitfile", unitfile)

        if isinstance(units, bool):
            assert units is not False, "Mixin should not be called here."
            units = "cgs"
        if units not in ["cgs", "code"]:
            # assuming that 'units' holds the path to the configuration
            raise ValueError("Unknown unit mode '%s'" % unitfile)
        if unitfile != "":
            unithints = get_unithints_fromfile(unitfile)
        for ptype in sorted(self.data):
            self.units[ptype] = {}
            pfields = self.data[ptype]
            for k in sorted(pfields.keys(allfields=True)):
                h5path = "/" + ptype + "/" + k
                if h5path in self._metadata_raw.keys():
                    # any hints available?
                    uh = unithints.get(ptype, {}).get(k, {})
                    attrs = dict(self._metadata_raw[h5path])
                    attrs.update(**uh)
                    try:
                        unit = extract_units_from_attrs(
                            attrs, require=True, mode=units, ureg=self.unitregistry
                        )
                    except ValueError as e:
                        if str(e) != "Could not find units.":
                            raise e
                        print("Hint: Did you pass a unit file? Is it complete?")
                        raise ValueError(
                            "Could not find units for '%s/%s'" % (ptype, k)
                        )
                    self.units[ptype][k] = unit

                    # redefine dask arrays with units
                    # TODO: This will instatiate all dask fields, need to user register_field
                    # as we run into a memory issue (https://github.com/h5py/h5py/issues/2220)
                    pfields[k] = unit * pfields[k]

    def _info_custom(self):
        rep = ""
        if hasattr(super(), "_info_custom"):
            if super()._info_custom() is not None:
                rep += super()._info_custom()
        rep += sprint("=== Unit-aware Dataset ===")
        rep += sprint("==========================")
        return rep
