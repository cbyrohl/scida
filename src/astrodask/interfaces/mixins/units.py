import logging
from typing import Optional

import numpy as np
import pint
from pint import UnitRegistry

from astrodask.config import get_config, get_config_fromfile
from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import sprint

log = logging.getLogger(__name__)


def extract_units_from_attrs(
    attrs,
    require: bool = False,
    mode: str = "cgs",
    ureg: Optional[pint.UnitRegistry] = None,
) -> pint.Quantity:
    """
    Extract units from given attributes.
    Parameters
    ----------
    attrs
        The attributes to extract the units from.
    require
        Whether to raise an error if no units are found.
    mode
        The unit mode to use (right now: cgs or code).
    ureg
        The unit registry to use.

    Returns
    -------

    """
    assert ureg is not None, "Always require passing registry now."
    udict = {}
    if mode == "cgs":
        udict["length"] = ureg("cm")
        udict["mass"] = ureg("g")
        udict["velocity"] = ureg("cm/s")
        udict["time"] = ureg("s")
    elif mode == "code":
        udict["length"] = ureg("code_length")
        udict["mass"] = ureg("code_mass")
        udict["velocity"] = ureg("code_velocity")
        udict["time"] = ureg("code_time")
    else:
        raise KeyError("Unknown unit mode '%s'." % mode)
    if "h" in ureg:
        udict["h"] = ureg("h")
    cgsfactor = ureg.Quantity(
        1.0
    )  # nothing to do if mode == "code" as we take values as they are
    if mode == "cgs":
        # the common mode is to expect some hints how to convert to cgs
        # get the conversion factor
        cgskey = [k for k in attrs if "cgs" in k.lower()]
        if "cgsunits" in cgskey:
            cgskey.remove(
                "cgsunits"
            )  # these are the units not the normalization factor
        assert len(cgskey) == 1
        cgskey = cgskey[0]
        cgsfactor = attrs[cgskey]
    # get dimensions
    unit = cgsfactor
    if isinstance(unit, np.ndarray):
        assert len(unit) == 1
        unit = unit[0]
    if unit == 0.0:
        unit = ureg.Quantity(1.0)  # zero makes no sense.
    # TODO: Missing a scaling!
    # TODO: Missing h scaling!
    ukeys = ["length", "mass", "velocity", "time", "h"]
    if any([k + "_scaling" in attrs.keys() for k in ukeys]):  # like TNG
        for k in ukeys:
            if mode == "code" and k == "h":
                continue  # h scaling absorbed into code units
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


def update_unitregistry(filepath: str, ureg: UnitRegistry):
    ulist = []
    conf = get_config_fromfile(filepath)
    for k, v in conf.get("units", {}).items():
        ulist.append("%s = %s" % (k, v))
    ureg.load_definitions(ulist)


class UnitMixin(Mixin):
    def __init__(self, *args, **kwargs):
        self.units = {}
        self.data = {}
        self._metadata_raw = {}
        # first, initialize self.data by calling super init
        super().__init__(*args, **kwargs)

        # get unit hints
        unithints = {}
        unitfile = ""
        if "dsname" in self.hints:
            c = get_config()
            dsprops = c["data"][self.hints["dsname"]]
            unitfile = dsprops.get("unitfile", "")
        unitfile = kwargs.pop("unitfile", unitfile)
        if unitfile != "":
            unithints = get_config_fromfile(unitfile)

        units = kwargs.pop("units")
        if isinstance(units, bool):
            assert units is not False, "Mixin should not be called here."
            units = "code"
        if units not in ["cgs", "code"]:
            # assuming that 'units' holds the path to the configuration
            raise ValueError("Unknown unit mode '%s'" % unitfile)

        # initialize unit registry
        ureg = UnitRegistry()  # before unit
        if unitfile != "":
            update_unitregistry(unitfile, ureg)
        ureg.default_system = "cgs"
        self.ureg = self.unitregistry = ureg

        # update fields with units
        fields_with_units = unithints.get("fields", {})
        for ptype in sorted(self.data):
            self.units[ptype] = {}
            pfields = self.data[ptype]
            for k in sorted(pfields.keys(allfields=True)):
                # first we check whether we are explicitly given a unit by a unit file
                if ptype in fields_with_units and k in fields_with_units[ptype]:
                    # we have two options: either the unit is given as a string, reflecting code units
                    # or as a dictionary with either or both keys 'cgsunits' and 'codeunits' who hold the unit strings
                    funit = fields_with_units[ptype][k]
                    if isinstance(funit, dict):
                        if units + "units" in funit:
                            unit = ureg(funit[units + "units"])
                            pfields[k] = unit * pfields[k]
                            if units == "cgs" and isinstance(pfields[k], pint.Quantity):
                                pfields[k] = pfields[k].to_base_units()
                            continue
                    else:
                        unit = ureg(funit)
                        pfields[k] = unit * pfields[k]
                        if units == "cgs" and isinstance(pfields[k], pint.Quantity):
                            pfields[k] = pfields[k].to_base_units()
                        continue
                # if not, we try to extract the unit from the metadata
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
