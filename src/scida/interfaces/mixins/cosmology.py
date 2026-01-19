from __future__ import annotations

import logging

import numpy as np

from scida.helpers_misc import sprint
from scida.interfaces.mixins.base import Mixin
from scida.interfaces.mixins.units import ignore_warn_redef
from scida.misc import get_scalar

log = logging.getLogger(__name__)


class CosmologyMixin(Mixin):
    """
    Mixin class for cosmological simulations. Adds cosmology and redshift attributes.
    """

    _mixin_name = "cosmology"

    def __init__(self, *args, **kwargs):
        """
        Initialize a CosmologyMixin object. Requires _metadata_raw to be filled.
        """
        self.metadata = {}
        if not hasattr(self, "hints"):
            self.hints = {}
        if hasattr(self, "_mixins"):
            self._mixins.append(self._mixin_name)
        else:
            self._mixins = [self._mixin_name]
        self.hints["cosmological"] = True

        # now call the parent class constructor as we need the metadata_raw
        super().__init__(*args, **kwargs)

        metadata_raw = self._metadata_raw
        c = get_cosmology_from_rawmetadata(metadata_raw)
        # sometimes, we want to inherit the cosmology from the parent dataset (e.g. catalogs for snapshots)
        if c is None and "metadata_raw_parent" in kwargs:
            c = get_cosmology_from_rawmetadata(kwargs["metadata_raw_parent"])
        # if still no cosmology, try alternative file prefixes in the same directory
        # (e.g., groups_* files may lack cosmology params but fof_subhalo_tab_* have them)
        if c is None and hasattr(self, "path"):
            c = _try_cosmology_from_alternative_files(self.path)
        self.cosmology = c
        if c is not None:
            self.hints["cosmology"] = c
        z = get_redshift_from_rawmetadata(metadata_raw)
        self.redshift = z
        self.metadata["redshift"] = self.redshift

        if hasattr(self, "ureg"):
            ureg = self.ureg
            with ignore_warn_redef(ureg):
                if c is not None:
                    ureg.define("h = %s" % str(c.h))
                elif "cosmology" in self.hints:
                    ureg.define("h = %s" % str(self.hints["cosmology"].h))
                if z is not None:
                    a = 1.0 / (1.0 + z)
                    ureg.define("a = %s" % str(float(a)))

    def _info_custom(self):
        """
        Return a custom info string for this mixin object.

        Returns
        -------
        str
            Custom info string for the mixin.
        """
        rep = sprint("=== Cosmological Simulation ===")
        if self.redshift is not None:
            rep += sprint("z = %.2f" % self.redshift)
        if self.cosmology is not None:
            rep += sprint("cosmology =", str(self.cosmology))
        rep += sprint("===============================")
        return rep

    @classmethod
    def validate(cls, metadata: dict, *args, **kwargs):
        """
        Validate whether the dataset is valid for this mixin.
        Parameters
        ----------
        metadata: dict
        args
        kwargs

        Returns
        -------
        bool

        """
        valid = False
        # in case of AREPO/GIZMO, we have the "ComovingIntegrationOn" key in the "Config" group set to 1
        if "/Config" in metadata:
            if "ComovingIntegrationOn" in metadata["/Config"]:
                comoving_integration_on = metadata["/Config"]["ComovingIntegrationOn"]
                if comoving_integration_on == 1:
                    return True
            # do not return False here, because we still might have a legacy run with no ComovingIntegrationOn key
            # for now, we check legacy runs by consistency of Time and Redshift key
        if (
            "/Header" in metadata
            and "Time" in metadata["/Header"]
            and "Redshift" in metadata["/Header"]
        ):
            time = get_scalar(metadata["/Header"]["Time"])
            redshift = get_scalar(metadata["/Header"]["Redshift"])
            # for cosmological runs, time is the scale factor a = 1/(1+z)
            if np.isclose(time, 1.0 / (1.0 + redshift)):
                # print("Legacy metadata detected for Cosmology Mixin detection.")
                valid = True
        # sometimes, we have a cosmological runs if we only have the redshift key but no time key (e.g. LGalaxies)
        if "/Header" in metadata and "Redshift" in metadata["/Header"]:
            if "Time" not in metadata["/Header"]:
                valid = True

        # in case of SWIFT, we have the "Cosmological run" key in the "Cosmology" group set to 1
        if "/Cosmology" in metadata:
            if "Cosmological run" in metadata["/Cosmology"]:
                # saved as array, thus get first element
                cosmological_run = get_scalar(
                    metadata["/Cosmology"]["Cosmological run"]
                )
                if cosmological_run == 1 or cosmological_run is True:
                    return True
            return False
        return valid


def get_redshift_from_rawmetadata(metadata_raw):
    """
    Get the redshift from the raw metadata.

    Parameters
    ----------
    metadata_raw: dict
        Raw metadata.

    Returns
    -------
    float
        Redshift.
    """
    z = metadata_raw.get("/Header", {}).get("Redshift", np.nan)
    z = get_scalar(z)
    return z


def get_cosmology_from_rawmetadata(metadata_raw):
    """
    Get the cosmology from the raw metadata.

    Parameters
    ----------
    metadata_raw: dict
        Raw metadata.

    Returns
    -------
    astropy.cosmology.Cosmology
        Defines a Flat Lambda CDM cosmology.

    """
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    # gadgetstyle
    aliasdict = dict(
        h=["HubbleParam", "Cosmology:h"],
        om0=["Omega0", "Cosmology:Omega_m"],
        ob0=["OmegaBaryon", "Cosmology:Omega_b"],
    )
    cparams = dict(h=None, om0=None, ob0=None)
    for grp in ["Parameters", "Header"]:
        for p, v in cparams.items():
            if v is not None:
                continue  # already acquired some value for this item.
            for alias in aliasdict[p]:
                cparams[p] = metadata_raw.get("/" + grp, {}).get(alias, cparams[p])

    # rockstar
    if cparams["h"] is None and "/cosmology:hubble" in metadata_raw:
        cparams["h"] = metadata_raw["/cosmology:hubble"]
    if cparams["om0"] is None and "/cosmology:omega_matter" in metadata_raw:
        cparams["om0"] = metadata_raw["/cosmology:omega_matter"]
    if cparams["ob0"] is None and "/cosmology:omega_baryon" in metadata_raw:
        cparams["ob0"] = metadata_raw["/cosmology:omega_baryon"]

    # flamingo-swift
    if cparams["om0"] is not None and float(cparams["om0"]) <= 0.0:
        # sometimes -1.0, then need to query Om_cdm + OM_b
        if (
            "/Parameters" in metadata_raw
            and "Cosmology:Omega_cdm" in metadata_raw["/Parameters"]
        ):
            omdm = float(metadata_raw["/Parameters"]["Cosmology:Omega_cdm"])
            omb = float(cparams["ob0"]) if cparams["ob0"] is not None else None
            if omb is not None:
                cparams["om0"] = omdm + omb

    h, om0, ob0 = cparams["h"], cparams["om0"], cparams["ob0"]
    if None in [h, om0]:
        log.info("Cannot infer cosmology.")
        return None
    if ob0 is None:
        log.info(
            "No Omega baryon given, we will assume a value of '0.0486' for the cosmology."
        )
        ob0 = 0.0486
    h = float(h)
    om0 = float(om0)
    ob0 = float(ob0)
    hubble0 = 100.0 * h * u.km / u.s / u.Mpc
    cosmology = FlatLambdaCDM(H0=hubble0, Om0=om0, Ob0=ob0)
    return cosmology


def _try_cosmology_from_alternative_files(path):
    """
    Try to infer cosmology from alternative file prefixes in the same directory.

    Some file formats (e.g., Illustris 'groups_*' files) don't contain cosmological
    parameters in their header, but other files in the same directory
    (e.g., 'fof_subhalo_tab_*') do. This function tries those alternatives.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the dataset directory.

    Returns
    -------
    astropy.cosmology.Cosmology or None
        The inferred cosmology, or None if not found.
    """
    import os
    import re

    if not os.path.isdir(path):
        return None

    # Alternative prefixes to try (in order of preference)
    alternative_prefixes = ["fof_subhalo_tab", "fof_subhalo", "snap"]

    try:
        files = os.listdir(path)
    except OSError:
        return None

    # Find which alternative prefixes exist
    available_prefixes = set()
    for fn in files:
        match = re.search(r"^(\w*)_(\d*)", fn)
        if match:
            available_prefixes.add(match.group(1))

    # Try each alternative prefix
    for prefix in alternative_prefixes:
        if prefix not in available_prefixes:
            continue

        try:
            from scida.io import load_metadata

            alt_metadata = load_metadata(path, fileprefix=prefix)
            c = get_cosmology_from_rawmetadata(alt_metadata)
            if c is not None:
                log.debug(f"Inferred cosmology from alternative file prefix '{prefix}'")
                return c
        except Exception as e:
            log.debug(f"Failed to load cosmology from prefix '{prefix}': {e}")
            continue

    return None
