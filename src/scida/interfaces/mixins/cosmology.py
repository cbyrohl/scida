import logging

import numpy as np

from scida.helpers_misc import sprint
from scida.interfaces.mixins.base import Mixin
from scida.interfaces.mixins.units import ignore_warn_redef

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
        if hasattr(self, "_mixins"):
            self._mixins.append(self._mixin_name)
        else:
            self._mixins = [self._mixin_name]
        super().__init__(*args, **kwargs)
        metadata_raw = self._metadata_raw
        c = get_cosmology_from_rawmetadata(metadata_raw)
        self.cosmology = c
        z = get_redshift_from_rawmetadata(metadata_raw)
        self.redshift = z
        self.metadata["redshift"] = self.redshift
        if hasattr(self, "ureg"):
            ureg = self.ureg
            with ignore_warn_redef(ureg):
                if c is not None:
                    ureg.define("h = %s" % str(c.h))
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
    z = float(z)
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
