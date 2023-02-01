import importlib.resources
import logging
import os
import re

import numpy as np
import unyt
import yaml
from unyt import UnitRegistry

from astrodask.config import get_config
from astrodask.interfaces.mixins.base import Mixin

log = logging.getLogger(__name__)


def str_to_unit(unitstr):
    unit = 1.0
    pattern = r"\s*[*/]*[a-zA-Z]+(\*\*-*\d+)*"
    # unitstr = "".join(unitstr.split())  # remove all whitespaces
    for u in re.finditer(pattern, unitstr):
        ustr = u.group()
        if u.group()[0] == "/":
            unit /= unyt.unyt_quantity.from_string(ustr[1:])
        elif u.group()[0] == "*":
            unit *= unyt.unyt_quantity.from_string(ustr[1:])
        else:
            # this should be the first unit in str
            unit *= unyt.unyt_quantity.from_string(ustr)
    return unit


def extract_units_from_attrs(attrs, require=False):
    """
    Extract the units from given attributes
    """
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
    # TODO: Missing h scaling!
    if any(
        [k + "_scaling" in attrs.keys() for k in ["length", "mass", "velocity", "time"]]
    ):  # like TNG
        # try to infer from dimensional scaling attributes
        unit *= unyt.cm ** attrs.get("length_scaling", 0.0)
        unit *= unyt.g ** attrs.get("mass_scaling", 0.0)
        unit *= (unyt.cm / unyt.s) ** attrs.get("velocity_scaling", 0.0)
        unit *= unyt.s ** attrs.get("time_scaling", 0.0)
    elif "Conversion factor" in attrs.keys():  # like SWIFT
        ustr = str(attrs["Conversion factor"])
        ustr = ustr.split("[")[-1].split("]")[0]
        if ustr.strip() == "-":  # no units, done
            return unit
        unit *= str_to_unit(ustr)
    elif "cgsunits" in attrs.keys():  # like EAGLE
        if attrs["cgsunits"] is not None:  # otherwise, this field has no units
            unitstr = attrs["cgsunits"]
            # bugged unyt: https://github.com/yt-project/unyt/issues/361
            # thus cannot pass all in one; instead pass unit one by one
            unit *= str_to_unit(unitstr)

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
        self.unitregistry = UnitRegistry()  # before unit
        super().__init__(*args, **kwargs)

        # discover units
        units = kwargs.pop("units")
        unithints = {}
        if "dsname" in self.hints:
            c = get_config()
            print("hints", self.hints)
            dsprops = c["data"][self.hints["dsname"]]
            units = dsprops.get("unitfile", units)

        if not isinstance(units, bool):
            # assuming that 'units' holds the path to the configuration
            unithints = get_unithints_fromfile(units)
        for ptype in self.data:
            self.units[ptype] = {}
            for k in self.data[ptype].keys(allfields=True):
                h5path = "/" + ptype + "/" + k
                if h5path in self._metadata_raw.keys():
                    # any hints available?
                    uh = unithints.get(ptype, {}).get(k, {})
                    attrs = dict(self._metadata_raw[h5path])
                    attrs.update(**uh)
                    try:
                        unit = extract_units_from_attrs(attrs, require=True)
                    except ValueError as e:
                        if str(e) != "Could not find units.":
                            raise e
                        print("Hint: Did you pass a unit file? Is it complete?")
                        raise ValueError(
                            "Could not find units for '%s/%s'" % (ptype, k)
                        )
                    self.units[ptype][k] = unit
        # redefine dask arrays with unyt units ?
