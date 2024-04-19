"""
Defines a series representing a Gadget-style simulation.
"""

import os
import pathlib
from pathlib import Path
from typing import Dict, Optional

from scida.discovertypes import _determine_mixins, _determine_type
from scida.interface import create_datasetclass_with_mixins
from scida.series import DatasetSeries


class GadgetStyleSimulation(DatasetSeries):
    """A series representing a Gadget-style simulation."""

    def __init__(
        self,
        path,
        prefix_dict: Optional[Dict] = None,
        subpath_dict: Optional[Dict] = None,
        arg_dict: Optional[Dict] = None,
        lazy=True,
        **interface_kwargs
    ):
        """
        Initialize a GadgetStyleSimulation object.

        Parameters
        ----------
        path: str
            Path to the simulation folder, should contain "output" folder.
        prefix_dict: dict
        subpath_dict: dict
        arg_dict: dict
        lazy: bool
        interface_kwargs: dict
        """
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
        keys = []
        for d in [prefix_dict, subpath_dict, arg_dict]:
            keys.extend(list(d.keys()))
        keys = set(keys)
        for k in keys:
            subpath = subpath_dict.get(k, "output")
            sp = p / subpath
            # by default, we assume that we are given a folder that has an "output" subfolder.
            # this is not always the case - for example for subboxes.
            # in such case, we attempt to continue with the given path.
            if not sp.exists():
                sp = p
            prefix = _get_snapshotfolder_prefix(sp)
            prefix = prefix_dict.get(k, prefix)
            if not sp.exists():
                if k != "paths":
                    continue  # do not require optional sources
                raise ValueError("Specified path '%s' does not exist." % (p / subpath))
            fns = os.listdir(sp)
            prfxs = set([f.split("_")[0] for f in fns if f.startswith(prefix)])
            if len(prfxs) == 0:
                if k != "paths":
                    continue  # do not require optional sources
                raise ValueError(
                    "Could not find any files with prefix '%s' in '%s'." % (prefix, sp)
                )
            prfx = prfxs.pop()

            paths = sorted([p for p in sp.glob(prfx + "_*")])
            # sometimes there are backup folders with different suffix, exclude those.
            paths = [
                p
                for p in paths
                if str(p).split("_")[-1].isdigit() or str(p).endswith(".hdf5")
            ]
            # attempt additional sorting in case zfill not used
            try:
                paths = sorted(paths, key=lambda x: int(str(x).split("_")[-1]))
            except:  # noqa
                pass
            paths_dict[k] = paths

        # make sure we have the same amount of paths respectively
        length = None
        for k in paths_dict.keys():
            paths = paths_dict[k]
            if length is None:
                length = len(paths)
            else:
                assert length == len(paths)

        paths = paths_dict.pop("paths", None)
        if paths is None:
            raise ValueError("Could not find any snapshot paths.")
        p = paths[0]
        cls = _determine_type(p)[1][0]

        mixins = _determine_mixins(path=p)
        cls = create_datasetclass_with_mixins(cls, mixins)

        kwargs = {arg_dict.get(k, "catalog"): paths_dict[k] for k in paths_dict.keys()}
        kwargs.update(**interface_kwargs)

        super().__init__(paths, datasetclass=cls, lazy=lazy, **kwargs)


def _get_snapshotfolder_prefix(path) -> str:
    """Try to infer the snapshot folder prefix"""
    p = pathlib.Path(path)
    if not p.exists():
        raise ValueError("Specified path '%s' does not exist." % path)
    fns = os.listdir(p)
    fns = [f for f in fns if os.path.isdir(p / f)]
    # find most occuring prefix
    prefixes = [f.split("_")[0] for f in fns]
    if len(prefixes) == 0:
        return ""
    prefix = max(set(prefixes), key=prefixes.count)
    return prefix
