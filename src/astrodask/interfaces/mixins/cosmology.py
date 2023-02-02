import numpy as np

from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import sprint


class CosmologyMixin(Mixin):
    def __init__(self, *args, **kwargs):
        self.metadata = {}
        super().__init__(*args, **kwargs)
        self.cosmology = None
        self.redshift = np.nan
        if "HubbleParam" in self.header and "Omega0" in self.header:
            import astropy.units as u
            from astropy.cosmology import FlatLambdaCDM

            h = self.header["HubbleParam"]
            om0 = self.header["Omega0"]
            ob0 = self.header["OmegaBaryon"]
            self.cosmology = FlatLambdaCDM(
                H0=100 * h * u.km / u.s / u.Mpc, Om0=om0, Ob0=ob0
            )
        try:
            self.redshift = self.header["Redshift"]
        except KeyError:  # no redshift attribute
            pass
        self.metadata["redshift"] = self.redshift

    def _info_custom(self):
        rep = sprint("=== Cosmological Simulation ===")
        if self.redshift is not None:
            rep += sprint("z = %.2f" % self.redshift)
        if self.cosmology is not None:
            rep += sprint("cosmology =", str(self.cosmology))
        rep += sprint("===============================")
        return rep
