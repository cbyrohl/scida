import numbers
import re
from math import pi

# TODO: Switch to pint for unit consistency
import astropy.units
import astropy.units as u
import dask.array as da
import numpy as np
import pint
import pytest
from packaging import version

from astrodask.convenience import load
from tests.testdata_properties import require_testdata_path


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_tng_units(testdatapath):
    # TODO: TNG50-4_group needs to be a fixture, not just hacked in as here
    catalog_path = testdatapath.replace("snapshot", "group")
    print(catalog_path)
    ds = load(testdatapath, units=True, catalog=catalog_path)
    ds_nounits = load(testdatapath, units=False, catalog=catalog_path)
    codeunitdict = {k: v for k, v in ds.header.items() if k.startswith("Unit")}
    codeunitdict["HubbleParam"] = ds.header["HubbleParam"]
    codeunitdict["Redshift"] = ds.header["Redshift"]
    units = get_units_from_TNGdocs(codeunitdict)
    ureg = ds.ureg
    ureg.default_system = "cgs"

    assert ureg("h").to_base_units().magnitude == 0.6774
    if not np.isclose(ds.redshift, 0.0):
        assert ureg("a").to_base_units().magnitude != 1.0

    for pk1, pk2 in zip(
        ["PartType0", "Group", "Subhalo"], ["particles", "groups", "subhalos"]
    ):
        # if pk1 not in ds.data.keys():
        #     continue
        for k in sorted(ds.data[pk1].keys()):
            if k in ["uid", "GroupID", "SubhaloID", "GroupOffsetsType"]:
                continue  # defined in package, dont need to check
            v = ds.data[pk1][k]
            # print("%s/%s" % (pk1, k))
            val1 = v[0].astype(np.float64)
            if isinstance(val1, pint.Quantity):
                mag1 = val1.magnitude
                if np.any(np.isnan(mag1)):
                    print(
                        "Skipping validation of '%s' due to underlying Nan entry." % k
                    )
                    continue
                val1 = val1.to_base_units().compute()
                mag1 = val1.magnitude
            else:
                val1 = val1.compute()
                mag1 = val1
                if np.any(np.isnan(mag1)):
                    print(
                        "Skipping validation of '%s' due to underlying Nan entry." % k
                    )
                print("WARNING (TODO): Field '%s' has no units." % k)
            u2 = units[pk2][k]
            mag2 = ds_nounits.data[pk1][k][0].compute()
            if isinstance(u2, astropy.units.Quantity):
                mag2 = mag2 * u2.cgs.value
            if not np.allclose(mag1, mag2, rtol=1e-4):
                print(pk1, k)
                print("conversion factor from TNG docs:")
                print("in cgs: ", u2.cgs)
                print("mag1, mag2:", mag1, mag2)
                print("ratio:", mag1 / mag2)
                raise ValueError("Entries do not match.")
    pass


@pytest.mark.skipif(
    version.parse(pint.__version__) <= version.parse("0.20.1"),
    reason="pint bug, hopefully fixed in next release, see https://github.com/hgrecco/pint/issues/1765",
)
def test_dask_pint1():
    ureg = pint.UnitRegistry()
    ureg.default_system = "cgs"
    ureg.define("halfmeter = 0.5 * meter")
    arr = da.ones(1) * ureg("halfmeter")
    print(arr[0].compute().to_base_units())


def test_dask_pint2():
    ureg = pint.UnitRegistry()
    ureg.default_system = "cgs"
    ureg.define("halfmeter = 0.5 * meter")
    arr = da.ones(1) * ureg("halfmeter")
    print(arr[0].compute().magnitude)
    print(arr[0].to_base_units().compute())


# TODO: only copied over a few functions.
#  Eventually we want to compare the "manual" units here against the unit discovery


def get_units_from_arepodocs(unitstr, codeunitdict):
    # first, extract list of unit tuples (consisting of unit as string and exponent)
    unittupledict = {}
    typemapping = {
        "Field name": "particles",
        "Group field name": "groups",
        "Subhalo field name": "subhalos",
    }
    pattern = re.compile(r"\s*(?:ext)*\{*([a-zA-Z]+)\}*\^*\{*([0-9.-]*)\}*")
    lines = unitstr.split("\n")
    typekey = None
    for line in lines:
        if line.startswith("| "):
            fieldentry = [s.strip() for s in line.split("|")[1:-1]]
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
    for k, dct in unittupledict.items():
        unitdict[k] = {}
        for name, uts in dct.items():
            units = get_units_from_unittuples(uts, codeunitdict)
            unitdict[k][name] = units
    return unitdict


def get_units_from_unittuples(unittuples, codeunitdict):
    # require cgs unitlengths for now
    assert "UnitLength_in_cm" in codeunitdict
    assert "UnitMass_in_g" in codeunitdict
    assert "UnitVelocity_in_cm_per_s" in codeunitdict
    assert "HubbleParam" in codeunitdict
    assert "Redshift" in codeunitdict
    cudict = {}
    cudict["UnitLength"] = codeunitdict["UnitLength_in_cm"] * u.cm
    cudict["UnitMass"] = codeunitdict["UnitMass_in_g"] * u.g
    cudict["UnitVelocity"] = codeunitdict["UnitVelocity_in_cm_per_s"] * u.cm / u.s
    # Pressure: M/L/T^2 = M/L^3 * V^2
    # P = M / L / T^2
    cudict["UnitPressure"] = (
        cudict["UnitMass"] / cudict["UnitLength"] ** 3 * cudict["UnitVelocity"] ** 2
    )
    cudict["h"] = codeunitdict["HubbleParam"]
    cudict["a"] = 1.0 / (1.0 + codeunitdict["Redshift"])
    cudict["ckpc"] = (
        cudict["a"] * u.kpc
    )  # as we express everything physical => multiply with a
    units = 1.0
    for ut in unittuples:
        if isinstance(ut, tuple):
            ut = list(ut)
            if len(ut) == 1:
                ut = ut + [1]
            elif isinstance(ut[1], str) and len(ut[1]) == 0:
                ut[1] = 1.0
        if isinstance(ut[0], numbers.Number):
            units *= ut[0] ** (float(ut[1]))
        else:
            try:
                units *= cudict[ut[0]] ** (float(ut[1]))
            except KeyError:
                units *= u.__dict__[ut[0]] ** (float(ut[1]))
    return units


def get_units_from_TNGdocs(codeunitdict):
    units = dict(particles={}, groups={}, subhalos={})
    unittuples = get_unittuples_from_TNGdocs_particles()
    units["particles"] = {
        k: get_units_from_unittuples(ut, codeunitdict) for k, ut in unittuples.items()
    }
    unittuples = get_unittuples_from_TNGdocs_groups()
    units["groups"] = {
        k: get_units_from_unittuples(ut, codeunitdict) for k, ut in unittuples.items()
    }
    unittuples = get_unittuples_from_TNGdocs_subhalos()
    units["subhalos"] = {
        k: get_units_from_unittuples(ut, codeunitdict) for k, ut in unittuples.items()
    }
    return units


# IllustrisTNG units from documentation (https://arepo-code.org/wp-content/userguide/snapshotformat.html, Jan 22)
def get_unittuples_from_TNGdocs_particles():
    uts = {}
    # Parts0
    uts["CenterOfMass"] = [("ckpc",), ("h", -1)]
    uts["Coordinates"] = [("ckpc",), ("h", -1)]
    uts["Density"] = [(10, 10), ("Msun",), ("h", -1), ("ckpc", -3), ("h", 3)]
    uts["ElectronAbundance"] = []
    uts["EnergyDissipation"] = [
        ("a", -1),
        (10, 10),
        ("Msun",),
        ("ckpc", -1),
        ("km", 3),
        ("s", -3),
    ]
    uts["GFM_AGNRadiation"] = [("erg",), ("s", -1), ("cm", -2), (4 * pi,)]
    uts["GFM_CoolingRate"] = [("erg",), ("cm", 3), ("s", -1)]
    uts["GFM_Metallicity"] = []
    uts["GFM_Metals"] = []
    uts["GFM_MetalsTagged"] = []
    uts["GFM_WindDMVelDisp"] = [("km",), ("s", -1)]
    uts["GFM_WindHostHaloMass"] = [(10, 10), ("Msun",), ("h", -1)]
    uts["InternalEnergy"] = [("km", 2), ("s", -2)]
    uts["InternalEnergyOld"] = [("km", 2), ("s", -2)]
    uts["Machnumber"] = []
    uts["MagneticField"] = [("h",), ("a", -2), ("UnitPressure", 0.5)]
    uts["MagneticFieldDivergence"] = [
        ("h", 3),
        ("a", -2),
        (10, 5),
        ("Msun", 0.5),
        ("km",),
        ("s", -1),
        ("ckpc", -5 / 2),
    ]
    uts["Masses"] = [(10, 10), ("Msun",), ("h", -1)]
    uts["NeutralHydrogenAbundance"] = []
    uts["ParticleIDs"] = []
    uts["Potential"] = [("km", 2), ("s", -2), ("a", -1)]
    uts["StarFormationRate"] = [("Msun",), ("yr", -1)]
    uts["SubfindDMDensity"] = uts["Density"]
    uts["SubfindDensity"] = uts["Density"]
    uts["SubfindHsml"] = [("ckpc",), ("h", "-1")]
    uts["SubfindVelDisp"] = [("km",), ("s", -1)]
    uts["Velocities"] = [("km",), ("a", 0.5), ("s", -1)]
    # Parts1 (only fields like in Parts0)
    # -
    # Parts2 (like DM)
    #
    # Parts3
    uts["FluidQuantities"] = []
    uts["ParentID"] = []
    uts["TracerID"] = []
    # Parts4
    uts["BirthPos"] = [("ckpc",), ("h", "-1")]
    uts["BirthVel"] = [("km",), ("a", 0.5), ("s", -1)]
    uts["GFM_InitialMass"] = [(10, 10), ("Msun",), ("h", -1)]
    uts["GFM_StellarFormationTime"] = []
    uts["GFM_StellarPhotometrics"] = [("mag",)]
    uts["StellarHsml"] = [("ckpc",), ("h", "-1")]
    # Parts5
    uts["BH_BPressure"] = [
        ("h", 4),
        ("a", -4),
        (10, 10),
        ("Msun",),
        ("km", 2),
        ("s", -2),
        ("ckpc", -3),
    ]
    uts["BH_CumEgyInjection_QM"] = [
        (10, 10),
        ("Msun",),
        ("h", -1),
        ("ckpc", 2),
        ("h", -2),
        (0.978, -2),
        ("Gyr", -2),
        ("h", 2),
    ]
    uts["BH_CumEgyInjection_RM"] = uts["BH_CumEgyInjection_QM"]
    uts["BH_CumMassGrowth_QM"] = [(10, 10), ("Msun",), ("h", -1)]
    uts["BH_CumMassGrowth_RM"] = uts["BH_CumMassGrowth_QM"]
    uts["BH_Density"] = uts["Density"]
    uts["BH_HostHaloMass"] = uts["Masses"]
    uts["BH_Hsml"] = uts["Coordinates"]
    uts["BH_Mass"] = uts["Masses"]
    uts["BH_Mdot"] = uts["Masses"] + [(0.978, -1), ("Gyr", -1), ("h",)]
    uts["BH_MdotBondi"] = uts["BH_Mdot"]
    uts["BH_MdotEddington"] = uts["BH_Mdot"]
    uts["BH_Pressure"] = [
        (10, 10),
        ("Msun",),
        ("h", -1),
        ("ckpc", -1),
        ("h", 1),
        (0.978, -2),
        ("Gyr", -2),
        ("h", 2),
    ]
    uts["BH_U"] = [("km", 2), ("s", -2)]
    uts["BH_Progs"] = []

    return uts


def get_unittuples_from_TNGdocs_groups():
    uts = {}
    uts["GroupBHMass"] = [(10, 10), ("Msun",), ("h", -1)]
    uts["GroupBHMdot"] = uts["GroupBHMass"] + [(0.978, -1), ("Gyr", -1), ("h", 1)]
    uts["GroupCM"] = [("ckpc",), ("h", -1)]
    uts["GroupFirstSub"] = []
    uts["GroupGasMetalFractions"] = []
    uts["GroupGasMetallicity"] = []
    uts["GroupLen"] = []
    uts["GroupLenType"] = []
    uts["GroupMass"] = uts["GroupBHMass"]
    uts["GroupMassType"] = uts["GroupBHMass"]
    uts["GroupNsubs"] = []
    uts["GroupPos"] = uts["GroupCM"]
    uts["GroupSFR"] = [("Msun",), ("yr", -1)]
    uts["GroupStarMetalFractions"] = []
    uts["GroupStarMetallicity"] = []
    uts["GroupVel"] = [("km",), ("s", -1), ("a", -1)]
    uts["GroupWindMass"] = uts["GroupBHMass"]
    uts["Group_M_Crit200"] = uts["GroupBHMass"]
    uts["Group_M_Crit500"] = uts["GroupBHMass"]
    uts["Group_M_Mean200"] = uts["GroupBHMass"]
    uts["Group_M_TopHat200"] = uts["GroupBHMass"]
    uts["Group_R_Crit200"] = uts["GroupCM"]
    uts["Group_R_Crit500"] = uts["GroupCM"]
    uts["Group_R_Mean200"] = uts["GroupCM"]
    uts["Group_R_TopHat200"] = uts["GroupCM"]
    return uts


def get_unittuples_from_TNGdocs_subhalos():
    uts = {}
    uts["SubhaloFlag"] = []
    uts["SubhaloBHMass"] = [(10, 10), ("Msun",), ("h", -1)]
    uts["SubhaloBHMdot"] = uts["SubhaloBHMass"] + [(0.978, -1), ("Gyr", -1), ("h", 1)]
    uts["SubhaloBfldDisk"] = [("h",), ("a", -2), ("UnitPressure", 0.5)]
    uts["SubhaloBfldHalo"] = uts["SubhaloBfldDisk"]
    uts["SubhaloCM"] = [("ckpc",), ("h", -1)]
    uts["SubhaloGasMetalFractions"] = []
    uts["SubhaloGasMetalFractionsHalfRad"] = []
    uts["SubhaloGasMetalFractionsMaxRad"] = []
    uts["SubhaloGasMetalFractionsSfr"] = []
    uts["SubhaloGasMetalFractionsSfrWeighted"] = []
    uts["SubhaloGasMetallicity"] = []
    uts["SubhaloGasMetallicityHalfRad"] = []
    uts["SubhaloGasMetallicityMaxRad"] = []
    uts["SubhaloGasMetallicitySfr"] = []
    uts["SubhaloGasMetallicitySfrWeighted"] = []
    uts["SubhaloGrNr"] = []
    uts["SubhaloHalfmassRad"] = uts["SubhaloCM"]
    uts["SubhaloHalfmassRadType"] = uts["SubhaloCM"]
    uts["SubhaloIDMostbound"] = []
    uts["SubhaloLen"] = []
    uts["SubhaloLenType"] = []
    uts["SubhaloMass"] = uts["SubhaloBHMass"]
    uts["SubhaloMassInHalfRad"] = uts["SubhaloBHMass"]
    uts["SubhaloMassInHalfRadType"] = uts["SubhaloBHMass"]
    uts["SubhaloMassInMaxRad"] = uts["SubhaloBHMass"]
    uts["SubhaloMassInMaxRadType"] = uts["SubhaloBHMass"]
    uts["SubhaloMassInRad"] = uts["SubhaloBHMass"]
    uts["SubhaloMassInRadType"] = uts["SubhaloBHMass"]
    uts["SubhaloMassType"] = uts["SubhaloBHMass"]
    uts["SubhaloParent"] = []
    uts["SubhaloPos"] = uts["SubhaloCM"]
    uts["SubhaloSFR"] = [("Msun",), ("yr", -1)]
    uts["SubhaloSFRinHalfRad"] = [("Msun",), ("yr", -1)]
    uts["SubhaloSFRinMaxRad"] = [("Msun",), ("yr", -1)]
    uts["SubhaloSFRinRad"] = [("Msun",), ("yr", -1)]
    uts["SubhaloSpin"] = [("kpc",), ("h", -1), ("km",), ("s", -1)]
    uts["SubhaloStarMetalFractions"] = []
    uts["SubhaloStarMetalFractionsHalfRad"] = []
    uts["SubhaloStarMetalFractionsMaxRad"] = []
    uts["SubhaloStarMetallicity"] = []
    uts["SubhaloStarMetallicityHalfRad"] = []
    uts["SubhaloStarMetallicityMaxRad"] = []
    uts[
        "SubhaloStellarPhotometrics"
    ] = []  # [("mag",)]  # TODO: Do not treat "mag" unit for now
    uts["SubhaloStellarPhotometricsMassInRad"] = uts["SubhaloBHMass"]
    uts["SubhaloStellarPhotometricsRad"] = uts["SubhaloCM"]
    uts["SubhaloVel"] = [("km",), ("s", -1)]
    uts["SubhaloVelDisp"] = uts["SubhaloVel"]
    uts["SubhaloVmax"] = uts["SubhaloVel"]
    uts["SubhaloVmaxRad"] = uts["SubhaloCM"]
    uts["SubhaloWindMass"] = uts["SubhaloBHMass"]
    return uts


unitstr_arepo = ""  # TODO: load from tng_unitstr.txt
