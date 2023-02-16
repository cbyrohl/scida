import numpy as np

from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import sprint


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
        if hasattr(self, "unitregistry"):
            if c is not None:
                self.unitregistry.define("h = %s" % str(c.h))
            if z is not None:
                a = 1.0 / (1.0 + z)
                self.unitregistry.define("a = %s" % str(a))

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

    h = metadata_raw.get("/Parameters", {}).get("HubbleParam", None)
    print(metadata_raw.keys())
    om0 = metadata_raw.get("/Parameters", {}).get("Omega0", None)
    ob0 = metadata_raw.get("/Parameters", {}).get("OmegaBaryon", None)
    if None in [h, om0, ob0]:
        print("Cannot infer cosmology.")
        return None
    H0 = 100.0 * h * u.km / u.s / u.Mpc
    cosmology = FlatLambdaCDM(H0=H0, Om0=om0, Ob0=ob0)
    return cosmology
