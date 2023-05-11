import logging
from typing import Optional

import numpy as np
import pint
from pint import UnitRegistry

from astrodask.config import get_config_fromfile, get_simulationconfig
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
        udict = dict(
            length=ureg("cm"), mass=ureg("g"), velocity=ureg("cm/s"), time=ureg("s")
        )
    elif mode == "code":
        udict = {k: ureg("code_" + k) for k in ["length", "mass", "velocity", "time"]}
    else:
        raise KeyError("Unknown unit mode '%s'." % mode)
    if "h" in ureg:
        udict["h"] = ureg("h")
    if "a" in ureg:
        udict["a"] = ureg("a")
    # nothing to do if mode == "code" as we take values as they are
    cgsfactor = ureg.Quantity(1.0)
    # for non-codeunits, we need to be provided some conversion factor
    if mode != "code":
        # the common mode is to expect some hints how to convert to cgs
        # get the conversion factor
        cgskey = [k for k in attrs if "cgs" in k.lower()]
        if mode + "units" in cgskey:
            # these are the units not the normalization factor
            cgskey.remove(mode + "units")
        assert len(cgskey) == 1
        cgskey = cgskey[0]
        cgsfactor = attrs[cgskey]
    # get dimensions
    resunit = None  # this is the variable we will return
    unit = cgsfactor
    if isinstance(unit, np.ndarray):
        assert len(unit) == 1
        unit = unit[0]
    if unit == 0.0:
        unit = ureg.Quantity(1.0)  # zero makes no sense.
    # TODO: Missing a scaling!
    # TODO: Missing h scaling!
    ukeys = ["length", "mass", "velocity", "time", "h", "a"]
    if any([k + "_scaling" in attrs.keys() for k in ukeys]):  # like TNG
        if mode != "cgs":
            raise ValueError("Only cgs supported here.")
        for k in ukeys:
            if mode == "code" and k == "h":
                # TODO: double check this...
                continue  # h scaling absorbed into code units
            aname = k + "_scaling"
            unit *= udict[k] ** attrs.get(aname, 0.0)
        resunit = unit
    elif "Conversion factor" in attrs.keys():  # like SWIFT
        ustr = str(attrs["Conversion factor"])
        ustr = ustr.split("[")[-1].split("]")[0]
        if ustr.strip() == "-":  # no units, done
            return unit
        unit *= ureg(ustr)
        resunit = unit
    elif "cgsunits" in attrs.keys():  # like EAGLE
        if attrs["cgsunits"] is not None:  # otherwise, this field has no units
            unitstr = attrs["cgsunits"]
            unit *= ureg(unitstr)
        resunit = unit
    elif require:
        raise ValueError("Could not find units.")
    return resunit


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
        # set true if we require units from either metadata or unitfie
        self.require_unitsspecs = kwargs.pop("require_unitspecs", True)

        ureg = UnitRegistry()  # before unit
        self.ureg = self.unitregistry = ureg

        super().__init__(*args, **kwargs)

        # get unit hints
        unithints = {}
        unitfile = ""
        if "dsname" in self.hints:
            c = get_simulationconfig()
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
        if unitfile != "":
            update_unitregistry(unitfile, self.ureg)
        self.ureg.default_system = "cgs"

        # update fields with units
        fwu = unithints.get("fields", {})
        for ptype in sorted(self.data):
            require_unitspecs = self.require_unitsspecs
            self.units[ptype] = {}
            pfields = self.data[ptype]
            if fwu.get(ptype, "") == "no_units":
                continue  # no units for any field in this group
            for k in sorted(pfields.keys(withrecipes=True)):
                unit = None
                h5path = "/" + ptype + "/" + k
                # first we check whether we are explicitly given a unit by a unit file
                override = False  # whether to override the metadata units
                gfwu = fwu.get(ptype, {})
                if gfwu is None:
                    gfwu = {}  # marginal case where no fields are given
                funit = gfwu.get(k, "N/A")
                if funit == "N/A":
                    funit = fwu.get("_all", {}).get(
                        k, "N/A"
                    )  # check whether defined for all particles
                if funit != "N/A":
                    # we have two options: either the unit is given as a string, reflecting code units
                    # or as a dictionary with either or both keys 'cgsunits' and 'codeunits' who hold the unit strings
                    if isinstance(funit, dict):
                        if units + "units" in funit:
                            unit = ureg(funit[units + "units"])
                        elif "units" in funit:
                            unit = ureg(funit["units"])
                        override = funit.get("override", False)
                    else:
                        unit = ureg(funit)
                    if units == "cgs":
                        if units == "cgs" and isinstance(pfields[k], pint.Quantity):
                            unit = unit.to_base_units()
                unit_metadata = None
                if h5path in self._metadata_raw.keys():
                    mode_metadata = unithints.get("metadata_unitsystem", units)
                    # if not, we try to extract the unit from the metadata
                    # check if any hints available
                    uh = unithints.get(ptype, {}).get(k, {})
                    attrs = dict(self._metadata_raw[h5path])
                    attrs.update(**uh)
                    require = unit is None
                    try:
                        unit_metadata = extract_units_from_attrs(
                            attrs,
                            require=require,
                            mode=mode_metadata,
                            ureg=self.unitregistry,
                        )
                    except ValueError as e:
                        if str(e) != "Could not find units.":
                            raise e
                        print("Hint: Did you pass a unit file? Is it complete?")
                        raise ValueError(
                            "Could not find units for '%s/%s'" % (ptype, k)
                        )

                if unit is not None and unit_metadata is not None:
                    # check whether both metadata and unit file agree
                    val_cgs_uf = unit.to_base_units().magnitude
                    val_cgs_md = unit_metadata.to_base_units().magnitude
                    if not override and not np.isclose(
                        val_cgs_uf, val_cgs_md, rtol=1e-3
                    ):
                        print("(units were checked against each other in cgs units.)")
                        print(
                            "cgs-factor comparison: %.5e (unitfile) != %.5e (metadata)"
                            % (val_cgs_uf, val_cgs_md)
                        )
                        raise ValueError(
                            "Unit mismatch for '%s': '%s' (unit file) vs. %s (metadata)"
                            % (h5path, unit, unit_metadata)
                        )

                if unit is None:
                    unit = unit_metadata

                if require_unitspecs and unit is None:
                    print(self.data[ptype][k])
                    raise ValueError(
                        "Cannot determine units from neither unit file nor metadata for '%s'."
                        % h5path
                    )
                elif unit is not None:
                    self.units[ptype][k] = unit
                    # redefine dask arrays with units
                    # TODO: This will instatiate all dask fields, need to user register_field
                    # as we run into a memory issue (https://github.com/h5py/h5py/issues/2220)
                    pfields[k] = unit * pfields[k]
                    if units == "cgs" and isinstance(pfields[k], pint.Quantity):
                        pfields[k] = pfields[k].to_base_units()

    def _info_custom(self):
        rep = ""
        if hasattr(super(), "_info_custom"):
            if super()._info_custom() is not None:
                rep += super()._info_custom()
        rep += sprint("=== Unit-aware Dataset ===")
        rep += sprint("==========================")
        return rep
