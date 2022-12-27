import os
import numpy as np
from typing import Optional
from os.path import join
from pathlib import Path
from astrodask.misc import map_interface_args
from astrodask.interfaces.arepo import ArepoSnapshot


class DatasetSeries(object):
    """A container for collection of interface instances"""

    def __init__(self, paths, *interface_args, datasetclass=None, **interface_kwargs):
        self._dataset_cls = datasetclass
        gen = map_interface_args(paths, *interface_args, **interface_kwargs)
        self.datasets = [datasetclass(p, *a, **kw) for p, a, kw in gen]

    @classmethod
    def from_directory(
        cls, path, *interface_args, datasetclass=None, pattern=None, **interface_kwargs
    ):
        if pattern is None:
            pattern = "*"
        p = Path(path)
        paths = [f for f in p.glob(pattern)]
        return cls(
            paths, *interface_args, datasetclass=datasetclass, **interface_kwargs
        )

    def get_dataset(self, index: Optional[int] = None):
        """Get dataset by some metadata property. In the base class, we go by list index."""
        if index is None:
            raise ValueError("Specify index.")
        return self.datasets[index]


class DirectoryCatalog(object):
    """A catalog consisting of interface instances contained in a directory."""

    def __init__(self, path):
        self.path = path


class ArepoSimulation(DatasetSeries):
    """A container for an arepo simulation."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        outpath = join(path, "output")
        gpaths = [p for p in Path(outpath).glob("groups_*")]
        spaths = [p for p in Path(outpath).glob("snapdir_*")]
        super().__init__(spaths, datasetclass=ArepoSnapshot, catalog=gpaths)

        # get redshifts
        self.redshifts = np.array([ds.redshift for ds in self.datasets])

    def get_dataset(
        self,
        index: Optional[int] = None,
        redshift: Optional[float] = None,
        redshift_reltol=1e-2,
    ):
        if index is None and redshift is None:
            raise ValueError("Specify index or redshift.")
        elif redshift is not None:
            dz = np.abs(self.redshifts - redshift)
            index = np.argmin(dz)
            assert np.isclose(redshift, self.redshifts[index], rtol=redshift_reltol)
        return super().get_dataset(index=index)
