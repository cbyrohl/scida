import inspect
import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

# tqdm.auto does not consistently give jupyter support to all users
from tqdm import tqdm

from scida.discovertypes import CandidateStatus
from scida.helpers_misc import hash_path
from scida.interface import create_MixinDataset
from scida.io import load_metadata
from scida.misc import map_interface_args, return_cachefile_path
from scida.registries import dataseries_type_registry


def delay_init(cls):
    class Delay(cls):
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        def __getattribute__(self, name):
            """Replace the class with the actual class and initialize it if needed."""
            # a few special calls do not trigger initialization:
            specialattrs = [
                "__repr__",
                "__str__",
                "__dir__",
                "__class__",
                "_args",
                "_kwargs",
                # https://github.com/jupyter/notebook/issues/2014
                "_ipython_canary_method_should_not_exist_",
            ]
            if name in specialattrs or name.startswith("_repr"):
                return object.__getattribute__(self, name)
            elif hasattr(cls, name) and inspect.ismethod(getattr(cls, name)):
                # do not need to initialize for class methods
                return getattr(cls, name)
            arg = self._args
            kwarg = self._kwargs
            self.__class__ = cls
            del self._args
            del self._kwargs
            self.__init__(*arg, **kwarg)
            if name == "evaluate_lazy":
                return getattr(self, "__repr__")  # some dummy
            return getattr(self, name)

        def __repr__(self):
            return "<Lazy %s>" % cls.__name__

    return Delay


class DatasetSeries(object):
    """A container for collection of interface instances"""

    def __init__(
        self,
        paths: Union[List[str], List[Path]],
        *interface_args,
        datasetclass=None,
        overwrite_cache=False,
        lazy=True,  # lazy will only initialize data sets on demand.
        async_caching=False,
        names=None,
        **interface_kwargs
    ):
        self.paths = paths
        self.names = names
        self.hash = hash_path("".join([str(p) for p in paths]))
        self._metadata = None
        self._metadatafile = return_cachefile_path(os.path.join(self.hash, "data.json"))
        self.lazy = lazy
        for p in paths:
            if not (isinstance(p, Path)):
                p = Path(p)
            if not (p.exists()):
                raise ValueError("Specified path '%s' does not exist." % p)
        dec = delay_init  # lazy loading

        # Catch Mixins and create type:
        mixins = interface_kwargs.pop("mixins", [])
        datasetclass = create_MixinDataset(datasetclass, mixins)
        self._dataset_cls = datasetclass

        gen = map_interface_args(paths, *interface_args, **interface_kwargs)
        self.datasets = [dec(datasetclass)(p, *a, **kw) for p, a, kw in gen]

        if self.metadata is None:
            print("Have not cached this data series. Can take a while.")
            dct = {}
            for i, (path, d) in enumerate(
                tqdm(zip(self.paths, self.datasets), total=len(self.paths))
            ):
                rawmeta = load_metadata(path)
                # class method does not initiate obj.
                dct[i] = d._clean_metadata_from_raw(rawmeta)
            self.metadata = dct
        elif async_caching:
            # hacky and should not be here this explicitly, just a proof of concept
            # for p in paths:
            #     loader = scida.io.determine_loader(p)
            pass

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        dataseries_type_registry[cls.__name__] = cls

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, key):
        return self.datasets[key]

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        # estimate whether we have a valid path for this dataseries
        return CandidateStatus.NO

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

    def get_dataset(
        self,
        index: Optional[int] = None,
        name: Optional[str] = None,
        reltol=1e-2,
        **kwargs
    ):
        """Get dataset by some metadata property. In the base class, we go by list index."""
        if index is None and name is None and len(kwargs) == 0:
            raise ValueError("Specify index/name or some parameter to select for.")
        # aliases for index:
        aliases = ["snap", "snapshot"]
        aliases_given = [k for k in aliases if k in kwargs]
        if index is not None:
            aliases_given += [index]
        if len(aliases_given) > 1:
            raise ValueError("Multiple aliases for index specified.")
        for a in aliases_given:
            if kwargs.get(a) is not None:
                index = kwargs.pop(a)

        if index is not None:
            return self.datasets[index]
        if name is not None:
            if self.names is None:
                raise ValueError("No names specified for members of this series.")
            if name not in self.names:
                raise ValueError("Name %s not found in this series." % name)
            return self.datasets[self.names.index(name)]
        if len(kwargs) > 0 and self.metadata is None:
            if self.lazy:
                raise ValueError(
                    "Cannot select by given keys before dataset evaluation."
                )
            raise ValueError("Unknown error.")  # should not happen?
        candidates = []
        candidates_props = {}
        props_compare = set()  # save names of fields we want to compare
        for k, v in kwargs.items():
            candidates_props[k] = []
        for i, (j, dm) in enumerate(self.metadata.items()):
            assert int(i) == int(j)
            is_candidate = True
            for k, v in kwargs.items():
                if k not in dm:
                    is_candidate = False
                    continue
                if isinstance(v, int) or isinstance(v, float):
                    candidates_props[k].append(dm[k])
                    props_compare.add(k)
                elif v != dm[k]:
                    is_candidate = False
            if is_candidate:
                candidates.append(i)
            else:  # unroll changes
                for p, lst in candidates_props.items():
                    if len(lst) > len(candidates):
                        lst.pop()
        # find candidate closest to request
        idxlist = []
        for k in props_compare:
            idx = np.argmin(np.abs(np.array(candidates_props[k]) - kwargs[k]))
            idxlist.append(idx)
        if len(set(idxlist)) > 1:
            raise ValueError("Ambiguous selection request")
        elif len(idxlist) == 0:
            raise ValueError("No candidate found.")
        index = candidates[idxlist[0]]
        # TODO: reintroduce tolerance check
        return self.get_dataset(index=index)

    @property
    def metadata(self):
        if self._metadata is not None:
            return self._metadata
        fp = self._metadatafile
        if os.path.exists(fp):
            md = json.load(open(fp, "r"))
            ikeys = sorted([int(k) for k in md.keys()])
            mdnew = {}
            for ik in ikeys:
                mdnew[ik] = md[str(ik)]
            self._metadata = mdnew
            return self._metadata
        return None

    @metadata.setter
    def metadata(self, dct):
        class ComplexEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                if isinstance(obj, np.int32):
                    return int(obj)
                if isinstance(obj, np.uint32):
                    return int(obj)
                if isinstance(obj, bytes):
                    return obj.decode("utf-8")
                if isinstance(obj, np.ndarray):
                    assert len(obj) < 1000  # dont want large obs here...
                    return list(obj)
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError as e:
                    print("obj failing json encoding:", obj)
                    raise e

        # def serialize_numpy(obj):
        #    if isinstance(obj, np.int64): return int(obj)
        #    if isinstance(obj, np.int32): return int(obj)
        #    return json.JSONEncoder.default(self, obj)
        self._metadata = dct
        fp = self._metadatafile
        # print(dct)
        if not os.path.exists(fp):
            json.dump(dct, open(fp, "w"), cls=ComplexEncoder)


class DirectoryCatalog(object):
    """A catalog consisting of interface instances contained in a directory."""

    def __init__(self, path):
        self.path = path


class HomogeneousSeries(DatasetSeries):
    """Series consisting of same-type data sets."""

    def __init__(self, path, **interface_kwargs):
        super().__init__()
        pass  # TODO
