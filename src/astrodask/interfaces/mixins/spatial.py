import logging

import numpy as np

from astrodask.fields import DerivedField
from astrodask.interfaces.mixins.base import Mixin
from astrodask.misc import rectangular_cutout_mask

log = logging.getLogger(__name__)


class SpatialMixin(Mixin):
    pass


class Spatial3DMixin(SpatialMixin):
    pass


class SpatialCartesian3DMixin(Spatial3DMixin):
    def __init__(self, *args, **kwargs):
        self.hints = {}
        super().__init__(*args, **kwargs)
        # TODO: determine whether periodic?
        self.pbc = True
        # TODO: Dynamically determine location of boxsize
        self.boxsize = np.nan * np.ones(3)
        bs = self.header["BoxSize"]
        if isinstance(bs, float) or (
            isinstance(bs, np.ndarray) and np.all(bs == bs[0])
        ):
            self.boxsize[:] = self.header["BoxSize"]
        else:
            # Have not thought about non-cubic cases yet.
            print("Boxsize:", bs)
            raise NotImplementedError
        self.boxsize = self.header["BoxSize"]
        common_coord_names = ["Coordinates", "Position", "GroupPos", "SubhaloPos"]
        for k, cntr in self.data.items():
            for ccn in common_coord_names:
                if ccn in cntr.keys(allfields=True):
                    log.debug("Found CoordinatesName '%s' for species '%s'" % (ccn, k))
                    self.hints["CoordinatesName"] = ccn

                    # register field with field containers as well
                    dfltname = "CoordinatesS"
                    if ccn is dfltname:
                        break

                    def fnc(arrs, cname=ccn):
                        return arrs[cname]

                    cntr.derivedfields[dfltname] = DerivedField(
                        dfltname, fnc, description="Coordinates alias"
                    )
                    break
                log.info("Did not find CoordinatesName for species '%s'" % k)

    def get_coords(self, parttype="PartType0"):
        return self.data[parttype][self.hints["CoordinatesName"]]

    def rectangular_cutout_mask(self, center, width, parttype="PartType0"):
        coords = self.get_coords(parttype=parttype)
        return rectangular_cutout_mask(
            center, width, coords, pbc=self.pbc, boxsize=self.boxsize
        )

    def rectangular_cutout(self, center, width, parttype="PartType0"):
        pass
