import contextlib
import logging
from typing import Optional, Union

import numpy as np
import pint
from pint import UnitRegistry

from scida.config import (
    combine_configs,
    get_config,
    get_config_fromfile,
    get_simulationconfig,
)
from scida.fields import FieldContainer, walk_container
from scida.helpers_misc import sprint
from scida.interfaces.mixins.base import Mixin

log = logging.getLogger(__name__)


def str_to_unit(
    unitstr: Optional[str], ureg: pint.UnitRegistry
) -> Union[pint.Unit, str]:
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
        if unitstr.lower() == "none":
            return "none"
    return ureg(unitstr)


def extract_units_from_attrs(
    attrs: dict,
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
    # initialize unit dictionary and registry to chosen mode
    if mode not in ["code", "mks", "cgs"]:
        raise KeyError("Unknown unit mode '%s'." % mode)
    udict = _get_default_units(mode, ureg)

    # determine any conversion factor as possible/needed
    cgskey, cgsfactor = _get_cgs_params(attrs, ureg, mode=mode)

    # for non-codeunits, we need to be provided some conversion factor or explicit units
    has_convfactor = cgskey is not None
    has_expl_units = False

    unit = 1.0
    # we can already set the value if we have a conversion factor
    if has_convfactor:
        unit = cgsfactor

    # now, need to get correct dimensions to the value
    if isinstance(unit, np.ndarray):
        if len(unit) != 1:
            log.debug("Unexpected shape (%s) of unit factor." % unit.shape)
        unit = unit[0]
    if unit == 0.0:
        unit = ureg.Quantity(1.0)  # zero makes no sense.

    # TODO: recheck that a and h scaling are added IFF code units are used
    ukeys = ["length", "mass", "velocity", "time", "h", "a"]
    if any([k + "_scaling" in attrs.keys() for k in ukeys]):  # like TNG
        if mode != "cgs":
            log.debug("Only have cgs conversion instead of code units.")
        for k in ukeys:
            if mode == "code" and k == "h":
                # TODO: double check this...
                continue  # h scaling absorbed into code units
            aname = k + "_scaling"
            unit *= udict[k] ** attrs.get(aname, 0.0)
        return unit
    # like a) generic SWIFT or b) FLAMINGO-SWIFT
    swiftkeys = ["Conversion factor", "Expression for physical CGS units"]
    swiftkeymatches = [k in attrs.keys() for k in swiftkeys]
    if any(swiftkeymatches):  # like SWIFT
        swiftkey = swiftkeys[np.where(swiftkeymatches)[0][0]]
        ustr = str(attrs[swiftkey])
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
                has_expl_units = True
            except pint.errors.UndefinedUnitError:
                log.info(
                    "Cannot parse unit string '%s' from metadata description. Skipping."
                    % unitstr
                )
            return unit
    if not has_expl_units and not has_convfactor:
        if require:
            raise ValueError("Could not find explicit units nor cgs-factor.")
    if require:
        raise ValueError("Could not find units.")
    return str_to_unit("", ureg)


def _get_cgs_params(attrs: dict, ureg: UnitRegistry, mode: str = "code") -> tuple:
    cgskeys = None
    if mode == "code":
        # nothing to do if mode == "code" as we take values as they are
        cgsfactor = ureg.Quantity(1.0)
        return cgskeys, cgsfactor

    # the common mode is to expect some hints how to convert to cgs
    # get the conversion factor
    cgskeys = [k for k in attrs if "cgs" in k.lower()]

    # there should be only one key holding the cgs factor.
    # remove those that are not
    if mode + "units" in cgskeys:
        # these are the units not the normalization factor
        cgskeys.remove(mode + "units")

    correct_keynames = [
        "Conversion factor to physical CGS (including cosmological corrections)",
        "to_cgs",
    ]
    cgskeys = [k for k in cgskeys if k in correct_keynames]

    assert len(cgskeys) <= 1  # only one key should be left
    if cgskeys is not None and len(cgskeys) > 0:
        cgskey = cgskeys[0]
        cgsfactor = attrs[cgskey]
        return cgskey, cgsfactor
    return None, None


def _get_default_units(mode: str, ureg: pint.UnitRegistry) -> dict:
    """
    Get the default units for the given mode.
    Parameters
    ----------
    mode
        The unit mode to use (right now: cgs or code).
    ureg
        The unit registry to use.

    Returns
    -------
    dict
        The default units.
    """
    if mode not in ["code", "mks", "cgs"]:
        raise KeyError("Unknown unit mode '%s'." % mode)
    if mode == "code":
        keys = ["length", "mass", "velocity", "time"]
        udict = {k: ureg["code_" + k] for k in keys if "code_" + k in ureg}
    if mode == "mks":
        udict = {
            "length": ureg.m,
            "mass": ureg.kg,
            "velocity": ureg.m / ureg.s,
            "time": ureg.s,
        }
    if mode == "cgs":
        udict = {
            "length": ureg.cm,
            "mass": ureg.g,
            "velocity": ureg.cm / ureg.s,
            "time": ureg.s,
        }

    # common factors in cosmological simulations
    if "h" in ureg:
        udict["h"] = str_to_unit("h", ureg)
    if "a" in ureg:
        udict["a"] = str_to_unit("a", ureg)

    return udict


def update_unitregistry_fromdict(udict: dict, ureg: UnitRegistry, warn_redef=False):
    ulist = []
    for k, v in udict.items():
        ulist.append("%s = %s" % (k, v))
    if warn_redef:
        ureg.load_definitions(ulist)
    else:
        with ignore_warn_redef(ureg):
            ureg.load_definitions(ulist)


def new_unitregistry() -> UnitRegistry:
    ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    ureg.define("unknown = 1.0")
    return ureg


def update_unitregistry(filepath: str, ureg: UnitRegistry):
    conf = get_config_fromfile(filepath).get("units", {})
    update_unitregistry_fromdict(conf, ureg)


class UnitMixin(Mixin):
    def __init__(self, *args, **kwargs):
        self.units = {}
        self.data = {}
        self._metadata_raw = {}

        ureg = kwargs.pop("ureg", None)
        if ureg is None:
            ureg = new_unitregistry()

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
        unitfile = kwargs.pop("unitfile", unitfile)  # passed kwarg takes precedence
        unitdefs = get_config_fromfile("units/general.yaml").get("units", {})
        apply_defaultunitfiles = kwargs.get("apply_defaultunitfiles", True)
        if apply_defaultunitfiles and hasattr(self, "_defaultunitfiles"):
            for uf in self._defaultunitfiles:
                unitdefs.update(get_config_fromfile(uf).get("units", {}))

        if unitfile != "":
            unitdefs.update(get_config_fromfile(unitfile).get("units", {}))

        units = kwargs.pop("units")
        if isinstance(units, bool):
            assert units is not False, "Mixin should not be called here."
            units = "code"
        if units not in ["cgs", "code"]:
            # assuming that 'units' holds the path to the configuration
            raise ValueError("Unknown unit mode '%s'" % unitfile)

        # initialize unit registry
        update_unitregistry_fromdict(unitdefs, self.ureg)
        self.ureg.default_system = "cgs"

        # update fields with units
        fieldudefs = []
        if apply_defaultunitfiles and hasattr(self, "_defaultunitfiles"):
            for uf in self._defaultunitfiles:
                fieldudefs.append(get_config_fromfile(uf).get("fields", {}))

        fieldudefs.append(unithints.get("fields", {}))

        fwu = combine_configs(fieldudefs, mode="overwrite_keys")  # merge udefs
        mode_metadata = unithints.get("metadata_unitsystem", "cgs")

        def add_units(container: FieldContainer, basepath: str):
            # first we check whether we are explicitly given a unit by a unit file
            override = False  # whether to override the metadata units
            gfwu = dict(**fwu)
            splt = basepath.split("/")
            for p in splt:
                if p == "":
                    continue  # skip empty path
                gfwu = gfwu.get(p, {})
                # if any parent container specifies "no_units", we will not add units, thus return
                if gfwu == "no_units" or gfwu == "none":
                    return
            if gfwu is None:
                gfwu = {}  # marginal case where no fields are given
            keys = sorted(
                container.keys(withfields=False, withrecipes=True, withgroups=False)
            )
            nrecipes = len(keys)
            keys += sorted(
                container.keys(withfields=True, withrecipes=False, withgroups=False)
            )
            recipe_mask = np.zeros(len(keys), dtype=bool)
            recipe_mask[:nrecipes] = True
            for i, k in enumerate(keys):
                is_recipe = recipe_mask[i]
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
                without_units = isinstance(unit, str) and unit == "none"
                if (
                    unit is not None and not without_units
                ) and unit_metadata is not None:
                    # check whether both metadata and unit file agree
                    val_cgs_uf = unit.to_base_units().magnitude
                    val_cgs_md = unit_metadata.to_base_units().magnitude
                    if not override and not np.isclose(
                        val_cgs_uf, val_cgs_md, rtol=1e-3
                    ):
                        # print("(units were checked against each other in cgs units.)")
                        msg = (
                            "Unit mismatch for '%s': '%s' (unit file) vs. %s (metadata)"
                            % (path, unit, unit_metadata)
                        )
                        msg += " [cgs-factors %.5e (unitfile) != %.5e (metadata)]" % (
                            val_cgs_uf,
                            val_cgs_md,
                        )
                        log.warning(msg)

                if unit is None:
                    unit = unit_metadata

                # if we still don't have a unit, we raise/warn as needed
                if unit is None:
                    unit = ureg("unknown")
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
                without_units = isinstance(unit, str) and unit == "none"
                if not without_units:
                    udict[k] = unit
                    # redefine dask arrays with units
                    # we treat recipes separately so that they stay recipes
                    # this is important due to the following memory issue for now:
                    # as we run into a memory issue (https://github.com/h5py/h5py/issues/2220)
                    if is_recipe:
                        container._fieldrecipes[k].units = unit
                        # TODO: Add cgs conversion for recipes, see else-statement.
                    else:
                        if isinstance(container[k], pint.Quantity):
                            log.debug("Field %s already has units, overwriting." % k)
                            container[k] = container[k].magnitude * unit
                        else:
                            if np.issubdtype(container[k].dtype, np.number):
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

        # add ureg to fieldcontainers
        def add_ureg(container: FieldContainer, basepath: str):
            container.set_ureg(self.ureg)

        self.data.set_ureg(self.ureg)
        walk_container(self.data, handler_group=add_ureg)

    def _info_custom(self):
        rep = ""
        if hasattr(super(), "_info_custom"):
            if super()._info_custom() is not None:
                rep += super()._info_custom()
        rep += sprint("=== Unit-aware Dataset ===")
        rep += sprint("==========================")
        return rep

    # def save(self, *args, fields: Union[str, Dict[str, Union[List[str], Dict[str, da.Array]]]] = "all", **kwargs):
    #    print("Wrapping save call with UnitMixin")
    #    if fields == "all":
    #        fields = dict()
    #        newcontainer = FieldContainer()

    #        def handler_field(entry, newpath, parent=None):
    #            key = pathlib.Path(newpath).name
    #            cntr = get_container_from_path(newpath, newcontainer, create_missing=True)
    #            cntr[key] = parent[key].magnitude
    #            print(entry, newpath, parent)
    #        walk_container(self.data, handler_field=handler_field)
    #    super().save(*args, fields=newcontainer, **kwargs)


@contextlib.contextmanager
def ignore_warn_redef(ureg):
    backupval = ureg._on_redefinition
    ureg._on_redefinition = "ignore"
    try:
        yield
    finally:
        ureg._on_redefinition = backupval
