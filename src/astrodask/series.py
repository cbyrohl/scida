import os
from os.path import join
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from astrodask.interfaces.arepo import ArepoSnapshot
from astrodask.misc import map_interface_args
from astrodask.registries import dataseries_type_registry


class DatasetSeries(object):
    """A container for collection of interface instances"""

    def __init__(
        self,
        paths: Union[List[str], List[Path]],
        *interface_args,
        datasetclass=None,
        **interface_kwargs
    ):
        self._dataset_cls = datasetclass
        for p in paths:
            if not (isinstance(p, Path)):
                p = Path(p)
            if not (p.exists()):
                raise ValueError("Specified path '%s' does not exist." % p)
        gen = map_interface_args(paths, *interface_args, **interface_kwargs)
        self.datasets = [datasetclass(p, *a, **kw) for p, a, kw in gen]

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        dataseries_type_registry[cls.__name__] = cls

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        # estimate whether we have a valid path for this dataseries
        return False

    @classmethod
    def from_directory(
        cls, path, *interface_args, datasetclass=None, pattern=None, **interface_kwargs
    ):
        p = Path(path)
        if not (p.exists()):
            raise ValueError("Specified path does not exist.")
        if pattern is None:
            pattern = "*"
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

    def __init__(self, path, **interface_kwargs):
        self.name = os.path.basename(path)
        p = Path(path)
        if not (p.exists()):
            raise ValueError("Specified path does not exist.")
        outpath = join(path, "output")
        gpaths = sorted([p for p in Path(outpath).glob("groups_*")])
        spaths = sorted([p for p in Path(outpath).glob("snapdir_*")])
        assert len(gpaths) == len(spaths)
        super().__init__(
            spaths, datasetclass=ArepoSnapshot, catalog=gpaths, **interface_kwargs
        )

        # get redshifts
        self.redshifts = np.array([ds.redshift for ds in self.datasets])

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        fns = os.listdir(path)
        if "output" in fns:
            opath = join(path, "output")
            folders = os.listdir(opath)
            folders = [f for f in folders if os.path.isdir(join(opath, f))]
            return any([f.startswith("snapdir") for f in folders])
        return False

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
            index = int(np.argmin(dz))
            if not np.isclose(redshift, self.redshifts[index], rtol=redshift_reltol):
                raise ValueError(
                    "Requested redshift unavailable, consider increasing tolerance 'redshift_reltol'."
                )
        return super().get_dataset(index=index)
