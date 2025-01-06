"""
Functionality to handle units.
"""

import contextlib
import logging
import tokenize
from enum import Enum
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

# enum for unit state
UnitState = Enum(
    "UnitState", ["success", "success_none", "missing", "mismatch", "parse_error"]
)


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
            # should not get any units attached (-> no pint decorated object)
            # this is different from unitless.
            return "none"
    unit = None
    try:
        unit = ureg(unitstr)
    except tokenize.TokenError as e:  # cannot parse; raises exception since python 3.12
        log.debug("Cannot parse unit string '%s'. Skipping." % unitstr)
        raise e
    except pint.errors.UndefinedUnitError as e:
        log.debug(
            "Cannot parse unit string '%s' from metadata description due to unknown units. Skipping."
            % unitstr
        )
        raise e
    return unit


def get_unitstr_from_attrs(attrs: dict) -> Optional[str]:
    """
    Get the unit string from the given attributes.

    Parameters
    ----------
    attrs: dict

    Returns
    -------
    Optional[str]
        The unit string.
    """

    unitstr = None

    # like a) generic SWIFT or b) FLAMINGO-SWIFT
    swiftkeys = ["Conversion factor", "Expression for physical CGS units"]
    swiftkeymatches = [k in attrs.keys() for k in swiftkeys]
    if any(swiftkeymatches):
        swiftkey = swiftkeys[np.where(swiftkeymatches)[0][0]]
        unitstr = str(attrs[swiftkey])
        unitstr = unitstr.split("[")[-1].split("]")[0]
        if unitstr.strip() == "-":  # no units, done
            unitstr = ""
    elif "cgsunits" in attrs.keys():
        # like EAGLE simulation
        if attrs["cgsunits"] is not None:  # otherwise, this field has no units
            unitstr = attrs["cgsunits"]
        return unitstr
    elif "description" in attrs.keys():
        # see whether we have units in the description key
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
            unitstr = unitstr.lower()

    return unitstr


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
    pint.Unit
        The extracted units.
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
    if unit in [0.0, 1.0]:
        unit = ureg.Quantity(1.0)

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
            exp = attrs.get(aname, 0.0)
            if exp != 0.0:
                unit *= udict[k] ** attrs.get(aname, 0.0)
        return unit

    unitstr = get_unitstr_from_attrs(attrs)
    if unitstr == "none":
        return "none"
    if unitstr is not None:
        has_expl_units = True
        ures = str_to_unit(unitstr, ureg)
        if ures == "none":
            return "none"
        unit *= ures
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
    """
    Create a new base unit registry.

    Returns
    -------
    UnitRegistry
        The new unit registry.
    """
    ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
    ureg.define("unknown = 1.0")
    # we remove the hours "h" unit, so we cannot confuse it with the Hubble factor
    del ureg._units["h"]
    # we remove the year "a" unit, so we cannot confuse it with the scale factor
    del ureg._units["a"]
    return ureg


def update_unitregistry(filepath: str, ureg: UnitRegistry):
    """
    Update the given unit registry with the units from the given file.

    Parameters
    ----------
    filepath: str
        The path to the unit file.
    ureg: UnitRegistry
        The unit registry to update.

    Returns
    -------
    None
    """
    conf = get_config_fromfile(filepath).get("units", {})
    update_unitregistry_fromdict(conf, ureg)


class UnitMixin(Mixin):
    """
    Mixin class for unit-aware datasets.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a UnitMixin object.

        Parameters
        ----------
        args
        kwargs
        """
        self.units = {}
        self.data = {}
        self._metadata_raw = {}
        self._logger_missingunits = log.getChild("missing_units")
        self._unitstates = dict()

        ureg = kwargs.pop("ureg", None)
        if ureg is None:
            ureg = new_unitregistry()

        self.ureg = self.unitregistry = ureg
        print("a in ureg?", "a" in self.ureg)

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
            if isinstance(unitfile, list):
                unithints = combine_configs(
                    [get_config_fromfile(uf) for uf in unitfile],
                    mode="overwrite_values",
                )
            else:
                unithints = get_config_fromfile(unitfile)
            newdefs = unithints.get("units", {})
            if newdefs is None:
                newdefs = {}
            unitdefs.update(**newdefs)

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

        fwu = combine_configs(fieldudefs, mode="overwrite_values")  # merge udefs
        mode_metadata = unithints.get("metadata_unitsystem", "cgs")

        def add_units(container: FieldContainer, basepath: str):
            """
            Add units to the given container.

            Parameters
            ----------
            container: FieldContainer
                The container to add units to.
            basepath: str
                The base path of the container.

            Returns
            -------
            None
            """
            # TODO: This function is a mess; unclear what it does/how generic it is.
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
                    except pint.errors.UndefinedUnitError:
                        self._unitstates[path] = UnitState.parse_error
                    except (
                        tokenize.TokenError
                    ):  # cannot parse; raises exception since python 3.12
                        self._unitstates[path] = UnitState.parse_error
                    except ValueError as e:
                        if str(e) != "Could not find units.":
                            raise e
                        print("Hint: Did you pass a unit file? Is it complete?")
                        raise ValueError("Could not find units for '%s'" % path)

                    # we do not want any pint decorated objects for index fields
                    # hardcoded for now
                    index_fields = ["ParticleIDs", "FOFGroupIDs", "SOIDs"]
                    if k in index_fields:
                        unit_metadata = "none"

                umatch = check_unit_mismatch(
                    unit,
                    unit_metadata,
                    override=override,
                    path=path,
                    logger=self._logger_missingunits,
                )
                if not umatch and path not in self._unitstates.keys():
                    self._unitstates[path] = UnitState.mismatch

                if unit is None:
                    unit = unit_metadata

                uexist = check_missing_units(
                    unit, missing_units, path, logger=self._logger_missingunits
                )
                if not uexist and path not in self._unitstates.keys():
                    self._unitstates[path] = UnitState.missing

                if unit is None:
                    unit = ureg("unknown")

                udict = self.units
                for p in basepath.split("/"):
                    if p == "":
                        continue
                    if p not in udict:
                        udict[p] = {}
                    udict = udict[p]
                without_units = isinstance(unit, str) and unit == "none"

                if without_units and path not in self._unitstates.keys():
                    self._unitstates[path] = UnitState.success_none
                if unit is not None and path not in self._unitstates.keys():
                    self._unitstates[path] = UnitState.success

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

        # manually run for "/" rest will be handled in walk_container
        add_units(self.data, "/")
        walk_container(self.data, handler_group=add_units)

        # add ureg to fieldcontainers
        def add_ureg(container: FieldContainer, basepath: str):
            container.set_ureg(self.ureg)

        self.data.set_ureg(self.ureg)
        walk_container(self.data, handler_group=add_ureg)

        self.missing_units()

    def _info_custom(self):
        """
        Return a custom info string for this mixin object.

        Returns
        -------
        str
            Custom info string for the mixin.
        """
        rep = ""
        if hasattr(super(), "_info_custom"):
            if super()._info_custom() is not None:
                rep += super()._info_custom()
        rep += sprint("=== Unit-aware Dataset ===")
        rep += sprint("==========================")
        return rep

    def missing_units(self, verbose=True):
        """
        Print information about fields with missing units.
        Parameters
        ----------
        verbose

        Returns
        -------
        bool:
            Whether there are any missing units.
        """
        success_states = [UnitState.success, UnitState.success_none]
        count = len([k for k, v in self._unitstates.items() if v not in success_states])
        if count > 0:
            print("Missing units for %d fields." % count)
            if verbose:
                print("Fields with missing units:")
                for k, v in self._unitstates.items():
                    if v not in success_states:
                        print("  - %s (%s)" % (k, v.name))
            print(
                "Re-run with\n\t>>> import logging\n\t>>> logging.getLogger().setLevel(logging.DEBUG)\n"
                "to learn more."
            )
        return count > 0

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
    """
    Context manager to ignore warnings about redefinitions of units.

    Parameters
    ----------
    ureg: pint.UnitRegistry
        The unit registry to temporarily ignore redefinition warnings for.

    Returns
    -------
    None
    """
    backupval = ureg._on_redefinition
    ureg._on_redefinition = "ignore"
    try:
        yield
    finally:
        ureg._on_redefinition = backupval


def check_unit_mismatch(unit, unit_metadata, override=False, path="", logger=log):
    """
    Check whether the units in the unit file and the metadata agree.

    Parameters
    ----------
    logger: logging.Logger
        The logger to use.
    unit: Union[pint.Unit, str]
        The unit from the unit file.
    unit_metadata: Union[pint.Unit, str]
        The unit from the metadata.
    override: bool
        Whether to override the metadata units.
    path: str
        Location of the field in the dataset (for logging purposes).

    Returns
    -------
    bool:
        Whether the units agree.
    """
    without_units = isinstance(unit, str) and unit == "none"
    without_units |= (
        isinstance(unit_metadata, str) and unit_metadata == "none"
    )  # explicitly no units
    if unit is not None and unit_metadata is not None:
        msg = None
        if without_units and unit == unit_metadata:
            # all good
            return True
        elif without_units and unit != unit_metadata:
            msg = "Unit mismatch for '%s': '%s' (unit file) vs. %s (metadata)" % (
                path,
                unit,
                unit_metadata,
            )
        else:
            # check whether both metadata and unit file agree
            print(unit, "vs.", unit_metadata)
            val_cgs_uf = unit.to_base_units().magnitude
            val_cgs_md = unit_metadata.to_base_units().magnitude
            if not override and not np.isclose(val_cgs_uf, val_cgs_md, rtol=1e-3):
                msg = "Unit mismatch for '%s': '%s' (unit file) vs. %s (metadata)" % (
                    path,
                    unit,
                    unit_metadata,
                )
                msg += " [cgs-factors %.5e (unitfile) != %.5e (metadata)]" % (
                    val_cgs_uf,
                    val_cgs_md,
                )
        if msg is not None:
            logger.info(msg)
            return False
    return True


def check_missing_units(unit, missing_units, path, logger=log):
    """
    Check whether units are missing and raise/warn as needed.
    Parameters
    ----------
    unit: Union[pint.Unit, str]
    missing_units: str
    path: str
    logger: logging.Logger
        The logger to use.

    Returns
    -------
    bool:
        Whether the units exist

    """
    # if we still don't have a unit, we raise/warn as needed
    if unit is None:
        msg = (
            "Cannot determine units from neither unit file nor metadata for '%s'."
            % path
        )
        if missing_units == "raise":
            raise ValueError(msg)
        elif missing_units == "warn":
            logger.info(msg)
        elif missing_units == "ignore":
            pass
        else:
            raise ValueError("Unknown missing_units setting '%s'." % missing_units)
        return False
    return True
