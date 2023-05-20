import logging

import numpy as np

from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import sprint

log = logging.getLogger(__name__)


class CosmologyMixin(Mixin):
    def __init__(self, *args, **kwargs):
        self.metadata = {}
        super().__init__(*args, **kwargs)
        metadata_raw = self._metadata_raw
        c = get_cosmology_from_rawmetadata(metadata_raw)
        self.cosmology = c
        z = get_redshift_from_rawmetadata(metadata_raw)
        self.redshift = z
        self.metadata["redshift"] = self.redshift
        if hasattr(self, "ureg"):
            ureg = self.ureg
            backupval = ureg._on_redefinition
            ureg._on_redefinition = "ignore"
            if c is not None:
                ureg.define("h = %s" % str(c.h))
            if z is not None:
                a = 1.0 / (1.0 + z)
                ureg.define("a = %s" % str(float(a)))
            ureg._on_redefinition = backupval

    def _info_custom(self):
        rep = sprint("=== Cosmological Simulation ===")
        if self.redshift is not None:
            rep += sprint("z = %.2f" % self.redshift)
        if self.cosmology is not None:
            rep += sprint("cosmology =", str(self.cosmology))
        rep += sprint("===============================")
        return rep


def get_redshift_from_rawmetadata(metadata_raw):
    z = metadata_raw.get("/Header", {}).get("Redshift", np.nan)
    return z


def get_cosmology_from_rawmetadata(metadata_raw):
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    aliasdict = dict(h=["HubbleParam"], om0=["Omega0"], ob0=["OmegaBaryon"])
    cparams = dict(h=None, om0=None, ob0=None)
    for grp in ["Parameters", "Header"]:
        for p, v in cparams.items():
            if v is not None:
                continue  # already acquired some value for this item.
            for alias in aliasdict[p]:
                cparams[p] = metadata_raw.get("/" + grp, {}).get(alias, cparams[p])

    h, om0, ob0 = cparams["h"], cparams["om0"], cparams["ob0"]
    if None in [h, om0]:
        log.info("Cannot infer cosmology.")
        return None
    elif ob0 is None:
        log.info(
            "No Omega baryon given, we will assume a value of '0.0486' for the cosmology."
        )
        ob0 = 0.0486
    hubble0 = 100.0 * h * u.km / u.s / u.Mpc
    cosmology = FlatLambdaCDM(H0=hubble0, Om0=om0, Ob0=ob0)
    return cosmology
