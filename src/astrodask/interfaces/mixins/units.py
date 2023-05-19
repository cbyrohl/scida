import logging
from typing import Optional

import numpy as np
import pint
from pint import UnitRegistry

from astrodask.config import get_config, get_config_fromfile, get_simulationconfig
from astrodask.fields import FieldContainer, walk_container
from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import sprint

log = logging.getLogger(__name__)


def str_to_unit(unitstr: Optional[str], ureg: pint.UnitRegistry) -> pint.Unit:
    """
    Convert a string to a unit.
    Parameters
    ----------
    ureg: pint.UnitRegistry
        The unit registry to use.
    unitstr
        The string to convert.

    Returns
    -------
    pint.Unit
        The unit.
    """
    if unitstr is not None:
        unitstr = unitstr.replace("dex", "decade")
    return ureg(unitstr)


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
        udict["length"] = str_to_unit("cm", ureg)
        udict["mass"] = str_to_unit("g", ureg)
        udict["velocity"] = str_to_unit("cm/s", ureg)
        udict["time"] = str_to_unit("s", ureg)
    elif mode == "code":
        for k in ["length", "mass", "velocity", "time"]:
            cstr = "code_" + k
            if cstr in ureg:
                udict[k] = str_to_unit(cstr, ureg)
    else:
        raise KeyError("Unknown unit mode '%s'." % mode)
    if "h" in ureg:
        udict["h"] = str_to_unit("h", ureg)
    if "a" in ureg:
        udict["a"] = str_to_unit("a", ureg)
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
        return unit
    if "Conversion factor" in attrs.keys():  # like SWIFT
        ustr = str(attrs["Conversion factor"])
        ustr = ustr.split("[")[-1].split("]")[0]
        if ustr.strip() == "-":  # no units, done
            return unit
        unit *= str_to_unit(ustr, ureg)
        return unit
    if "cgsunits" in attrs.keys():  # like EAGLE
        if attrs["cgsunits"] is not None:  # otherwise, this field has no units
            unitstr = attrs["cgsunits"]
            unit *= str_to_unit(unitstr, ureg)
        return unit
    if (
        "description" in attrs.keys()
    ):  # see whether we have units in the description key
        unitstr = None
        desc = attrs["description"]
        try:
            unitstr = desc.split("[")[1].split("]")[0]
        except IndexError:
            try:
                unitstr = desc.split("(")[1].split(")")[0]
            except IndexError:
                pass
        if unitstr is not None and unitstr != desc and unitstr != "None":
            unitstr = unitstr.strip("'")
            # parsing from the description is usually messy and not guaranteed to work. We thus allow failure here.
            try:
                unit *= str_to_unit(unitstr, ureg)
            except pint.errors.UndefinedUnitError:
                log.info(
                    "Cannot parse unit string '%s' from metadata description. Skipping."
                    % unitstr
                )
            return unit
    if require:
        raise ValueError("Could not find units.")
    return str_to_unit("", ureg)


def update_unitregistry_fromdict(udict: dict, ureg: UnitRegistry):
    ulist = []
    for k, v in udict.items():
        ulist.append("%s = %s" % (k, v))
    ureg.load_definitions(ulist)


def update_unitregistry(filepath: str, ureg: UnitRegistry):
    conf = get_config_fromfile(filepath).get("units", {})
    update_unitregistry_fromdict(conf, ureg)


class UnitMixin(Mixin):
    def __init__(self, *args, **kwargs):
        self.units = {}
        self.data = {}
        self._metadata_raw = {}

        ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        self.ureg = self.unitregistry = ureg

        super().__init__(*args, **kwargs)

        # get unit hints
        unithints = {}
        unitfile = ""
        userconf = get_config()
        missing_units = userconf.get("missing_units", "warn")
        assert missing_units in ["warn", "raise", "ignore"]
        if "dsname" in self.hints:
            c = get_simulationconfig()
            dsprops = c["data"][self.hints["dsname"]]
            missing_units = dsprops.get("missing_units", missing_units)
            unitfile = dsprops.get("unitfile", "")
        unitfile = kwargs.pop("unitfile", unitfile)
        unitdefs = get_config_fromfile("units/general.yaml").get("units", {})
        if unitfile != "":
            unithints = get_config_fromfile(unitfile)
            unitdefs.update(unithints.get("units", {}))

        units = kwargs.pop("units")
        if isinstance(units, bool):
            assert units is not False, "Mixin should not be called here."
            units = "code"
        if units not in ["cgs", "code"]:
            # assuming that 'units' holds the path to the configuration
            raise ValueError("Unknown unit mode '%s'" % unitfile)

        # initialize unit registry
        if unitfile != "":
            update_unitregistry_fromdict(unitdefs, self.ureg)
        self.ureg.default_system = "cgs"

        # update fields with units
        fwu = unithints.get("fields", {})
        mode_metadata = unithints.get("metadata_unitsystem", units)
        # TODO: Not sure about block below needed again
        # if mode_metadata == "code":
        #     if "code_length" not in self.ureg:
        #         log.debug("No code units given, assuming cgs.")
        #         mode_metadata = "cgs"

        def add_units(container: FieldContainer, basepath: str):
            # first we check whether we are explicitly given a unit by a unit file
            override = False  # whether to override the metadata units
            gfwu = fwu
            k = basepath.split("/")
            for p in basepath.split("/"):
                gfwu = fwu.get(p, {})
            if gfwu == "no_units":
                return  # no units for this container
            if gfwu is None:
                gfwu = {}  # marginal case where no fields are given
            for k in sorted(container.keys(withgroups=False)):
                path = basepath + "/" + k
                unit = None
                if basepath == "/":
                    path = "/" + k
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
                            unit = str_to_unit(funit[units + "units"], ureg)
                        elif "units" in funit:
                            unit = str_to_unit(funit["units"], ureg)
                        override = funit.get("override", False)
                    else:
                        unit = str_to_unit(funit, ureg)
                    if units == "cgs" and isinstance(unit, pint.Quantity):
                        unit = unit.to_base_units()
                unit_metadata = None
                if path in self._metadata_raw.keys():
                    # if not, we try to extract the unit from the metadata
                    # check if any hints available
                    uh = unithints
                    for p in basepath.split("/"):
                        uh = uh.get(p, {})
                    uh = uh.get(k, {})
                    attrs = dict(self._metadata_raw[path])
                    attrs.update(**uh)
                    # require = unit is None
                    try:
                        unit_metadata = extract_units_from_attrs(
                            attrs,
                            require=False,
                            mode=mode_metadata,
                            ureg=self.unitregistry,
                        )
                    except ValueError as e:
                        if str(e) != "Could not find units.":
                            raise e
                        print("Hint: Did you pass a unit file? Is it complete?")
                        raise ValueError("Could not find units for '%s'" % path)

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
                            % (path, unit, unit_metadata)
                        )

                if unit is None:
                    unit = unit_metadata

                # if we still don't have a unit, we raise/warn as needed
                if unit is None:
                    unit = ureg("")
                    msg = (
                        "Cannot determine units from neither unit file nor metadata for '%s'."
                        % path
                    )
                    if missing_units == "raise":
                        raise ValueError(msg)
                    elif missing_units == "warn":
                        log.warning(msg)

                udict = self.units
                for p in basepath.split("/"):
                    if p == "":
                        continue
                    if p not in udict:
                        udict[p] = {}
                    udict = udict[p]
                udict[k] = unit
                # redefine dask arrays with units
                # we treat recipes separately so that they stay recipes
                # this is important due to the following memory issue for now:
                # as we run into a memory issue (https://github.com/h5py/h5py/issues/2220)
                if k not in container.fields and k in container.fieldrecipes:
                    container.fieldrecipes[k].units = unit
                    # TODO: Add cgs conversion for recipes, see else-statement.
                else:
                    container[k] = unit * container[k]
                    if units == "cgs" and isinstance(container[k], pint.Quantity):
                        container[k] = container[k].to_base_units()

        # fwu = unithints.get("fields", {})
        # for ptype in sorted(self.data):
        #    self.units[ptype] = {}
        #    pfields = self.data[ptype]
        #    if fwu.get(ptype, "") == "no_units":
        #        continue  # no units for any field in this group
        #    for k in sorted(pfields.keys(withrecipes=True)):

        # manually run for "/" rest will be handled in walk_container
        add_units(self.data, "/")
        walk_container(self.data, handler_group=add_units)

    def _info_custom(self):
        rep = ""
        if hasattr(super(), "_info_custom"):
            if super()._info_custom() is not None:
                rep += super()._info_custom()
        rep += sprint("=== Unit-aware Dataset ===")
        rep += sprint("==========================")
        return rep
