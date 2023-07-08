import os
from pathlib import Path
from typing import Dict, Optional

from scida.discovertypes import _determine_mixins, _determine_type
from scida.interface import create_MixinDataset
from scida.series import DatasetSeries


class GadgetStyleSimulation(DatasetSeries):
    """A series representing a gadgetstyle simulation."""

    def __init__(
        self,
        path,
        prefix_dict: Optional[Dict] = None,
        subpath_dict: Optional[Dict] = None,
        arg_dict: Optional[Dict] = None,
        lazy=True,
        async_caching=False,
        **interface_kwargs
    ):
        self.path = path
        self.name = os.path.basename(path)
        if prefix_dict is None:
            prefix_dict = dict()
        if subpath_dict is None:
            subpath_dict = dict()
        if arg_dict is None:
            arg_dict = dict()
        p = Path(path)
        if not (p.exists()):
            raise ValueError("Specified path '%s' does not exist." % path)
        paths_dict = dict()
        for k in prefix_dict.keys():
            prefix = prefix_dict[k]
            subpath = subpath_dict.get(k, "output")
            sp = p / subpath
            if not sp.exists():
                raise ValueError("Specified path '%s' does not exist." % (p / subpath))
            fns = os.listdir(sp)
            prfxs = set([f.split("_")[0] for f in fns if f.startswith(prefix)])
            assert len(prfxs) == 1
            prfx = prfxs.pop()

            paths = sorted([p for p in sp.glob(prfx + "_*")])
            # sometimes there are backup folders with different suffix, exclude those.
            paths = [p for p in paths if str(p).split("_")[-1].isdigit()]
            paths_dict[k] = paths

        # make sure we have the same amount of paths respectively
        length = None
        for k in paths_dict.keys():
            paths = paths_dict[k]
            if length is None:
                length = len(paths)
            else:
                assert length == len(paths)

        paths = paths_dict.pop("paths")
        p = paths[0]
        cls = _determine_type(p)[1][0]

        mixins = _determine_mixins(path=p)
        cls = create_MixinDataset(cls, mixins)

        kwargs = {arg_dict.get(k, "catalog"): paths_dict[k] for k in paths_dict.keys()}

        super().__init__(
            paths,
            datasetclass=cls,
            lazy=lazy,
            async_caching=async_caching,
            **kwargs,
            **interface_kwargs
        )
