import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# tqdm.auto does not consistently give jupyter support to all users
from tqdm import tqdm

from scida.discovertypes import CandidateStatus
from scida.helpers_misc import hash_path, sprint
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
                rawmeta = load_metadata(path, choose_prefix=True)
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

    def info(self):
        rep = ""
        rep += "class: " + sprint(self.__class__.__name__)
        props = self._repr_dict()
        for k, v in props.items():
            rep += sprint("%s: %s" % (k, v))
        if self.metadata is not None:
            rep += sprint("=== metadata ===")
            # we print the range of each metadata attribute
            minmax_dct = {}
            for mdct in self.metadata.values():
                for k, v in mdct.items():
                    if k not in minmax_dct:
                        minmax_dct[k] = [v, v]
                    else:
                        if not np.isscalar(v):
                            continue  # cannot compare arrays
                        minmax_dct[k][0] = min(minmax_dct[k][0], v)
                        minmax_dct[k][1] = max(minmax_dct[k][1], v)
            for k in minmax_dct:
                reprval1, reprval2 = minmax_dct[k][0], minmax_dct[k][1]
                if isinstance(reprval1, float):
                    reprval1 = "%.2f" % reprval1
                    reprval2 = "%.2f" % reprval2
                m1 = minmax_dct[k][0]
                m2 = minmax_dct[k][1]
                if (not np.isscalar(m1)) or (np.isscalar(m1) and m1 == m2):
                    rep += sprint("%s: %s" % (k, minmax_dct[k][0]))
                else:
                    rep += sprint(
                        "%s: %s -- %s" % (k, minmax_dct[k][0], minmax_dct[k][1])
                    )
            rep += sprint("============")
        print(rep)

    @property
    def data(self):
        raise AttributeError(
            "Series do not have 'data' attribute. Load a dataset from series.get_dataset()."
        )

    def _repr_dict(self) -> Dict[str, str]:
        """
        Return a dictionary of properties to be printed by __repr__ method.
        Returns
        -------
        dict
        """
        props = dict()
        sources = [str(p) for p in self.paths]
        props["source(id=0)"] = sources[0]
        props["Ndatasets"] = len(self.datasets)
        return props

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        Returns
        -------
        str
        """
        props = self._repr_dict()
        clsname = self.__class__.__name__
        result = clsname + "["
        for k, v in props.items():
            result += "%s=%s, " % (k, v)
        result = result[:-2] + "]"
        return result

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
                for lst in candidates_props.values():
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
        # tolerance check
        for k in props_compare:
            if not np.isclose(kwargs[k], self.metadata[index][k], rtol=reltol):
                msg = (
                    "Candidate does not match tolerance for %s (%s vs %s requested)"
                    % (
                        k,
                        self.metadata[index][k],
                        kwargs[k],
                    )
                )
                raise ValueError(msg)
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
        # TODO
        super().__init__()
