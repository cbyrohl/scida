import os
from os.path import join
from pathlib import Path

from scida.discovertypes import _determine_mixins, _determine_type
from scida.interface import create_MixinDataset
from scida.series import DatasetSeries


class ArepoSimulation(DatasetSeries):
    """A series representing an arepo simulation."""

    def __init__(self, path, lazy=True, async_caching=False, **interface_kwargs):
        self.path = path
        self.name = os.path.basename(path)
        p = Path(path)
        if not (p.exists()):
            raise ValueError("Specified path '%s' does not exist." % path)
        outpath = join(path, "output")
        if not os.path.isdir(outpath):
            outpath = path
        fns = os.listdir(outpath)
        fns = [f for f in fns if os.path.isdir(os.path.join(outpath, f))]
        sprefix = set([f.split("_")[0] for f in fns if f.startswith("snap")])
        gprefix = set([f.split("_")[0] for f in fns if f.startswith("group")])
        assert len(sprefix) == 1
        assert len(gprefix) == 1
        sprefix, gprefix = sprefix.pop(), gprefix.pop()
        gpaths = sorted([p for p in Path(outpath).glob(gprefix + "_*")])
        spaths = sorted([p for p in Path(outpath).glob(sprefix + "_*")])

        # sometimes there are backup folders with different suffix, exclude those.
        gpaths = [p for p in gpaths if str(p).split("_")[-1].isdigit()]
        spaths = [p for p in spaths if str(p).split("_")[-1].isdigit()]

        assert len(gpaths) == len(spaths)
        p = spaths[0]
        cls = _determine_type(p)[1][0]

        mixins = _determine_mixins(path=p)
        cls = create_MixinDataset(cls, mixins)

        super().__init__(
            spaths,
            datasetclass=cls,
            catalog=gpaths,
            lazy=lazy,
            async_caching=async_caching,
            **interface_kwargs
        )

        # get redshifts
        # self.redshifts = np.array([ds.redshift for ds in self.datasets])

    @classmethod
    def validate_path(cls, path, *args, **kwargs):
        if not os.path.isdir(path):
            return False
        fns = os.listdir(path)
        sprefixs = ["snapdir", "snapshot"]
        opath = path
        if "output" in fns:
            opath = join(path, "output")
        folders = os.listdir(opath)
        folders = [f for f in folders if os.path.isdir(join(opath, f))]
        return any([f.startswith(k) for f in folders for k in sprefixs])
