"""
This module contains the base class for DataSeries, which is a container for collections of dataset instances.
"""

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
from scida.interface import create_datasetclass_with_mixins
from scida.io import load_metadata
from scida.misc import map_interface_args, return_cachefile_path
from scida.registries import dataseries_type_registry


def delay_init(cls):
    """
    Decorate class to delay initialization until an attribute is requested.
    Parameters
    ----------
    cls:
        class to decorate

    Returns
    -------
    Delay
    """

    class Delay(cls):
        """
        Delayed initialization of a class. The class is replaced by the actual class
        when an attribute is requested.
        """

        def __init__(self, *args, **kwargs):
            """Store arguments for later initialization."""
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
            """Return a string representation of the lazy class."""
            return "<Lazy %s>" % cls.__name__

    return Delay


class DatasetSeries(object):
    """A container for collections of dataset instances

    Attributes
    ----------
    datasets: list
        list of dataset instances
    paths: list
        list of paths to data
    names: list
        list of names for datasets
    hash: str
        hash of the object, constructed from dataset paths.
    """

    def __init__(
        self,
        paths: Union[List[str], List[Path]],
        *interface_args,
        datasetclass=None,
        overwrite_cache=False,
        lazy=True,  # lazy will only initialize data sets on demand.
        names=None,
        **interface_kwargs
    ):
        """

        Parameters
        ----------
        paths: list
            list of paths to data
        interface_args:
            arguments to pass to interface class
        datasetclass:
            class to use for dataset instances
        overwrite_cache:
            whether to overwrite existing cache
        lazy:
            whether to initialize datasets lazily
        names:
            names for datasets
        interface_kwargs:
            keyword arguments to pass to interface class
        """
        self.paths = paths
        self.names = names
        self.hash = hash_path("".join([str(p) for p in paths]))
        self._metadata = None
        self._metadatafile = return_cachefile_path(os.path.join(self.hash, "data.json"))
        self.lazy = lazy
        if overwrite_cache and os.path.exists(self._metadatafile):
            os.remove(self._metadatafile)
        for p in paths:
            if not (isinstance(p, Path)):
                p = Path(p)
            if not (p.exists()):
                raise ValueError("Specified path '%s' does not exist." % p)
        dec = delay_init  # lazy loading

        # Catch Mixins and create type:
        ikw = dict(overwrite_cache=overwrite_cache)
        ikw.update(**interface_kwargs)
        mixins = ikw.pop("mixins", [])
        datasetclass = create_datasetclass_with_mixins(datasetclass, mixins)
        self._dataset_cls = datasetclass

        gen = map_interface_args(paths, *interface_args, **ikw)
        self.datasets = [dec(datasetclass)(p, *a, **kw) for p, a, kw in gen]

        if self.metadata is None:
            print("Have not cached this data series. Can take a while.")
            dct = {}
            for i, (path, d) in enumerate(
                tqdm(zip(self.paths, self.datasets), total=len(self.paths))
            ):
                rawmeta = load_metadata(
                    path, choose_prefix=True, use_cachefile=not (overwrite_cache)
                )
                # class method does not initiate obj.
                dct[i] = d._clean_metadata_from_raw(rawmeta)
            self.metadata = dct

    def __init_subclass__(cls, *args, **kwargs):
        """
        Register datasetseries subclass in registry.
        Parameters
        ----------
        args:
            (unused)
        kwargs:
            (unused)
        Returns
        -------
        None
        """
        super().__init_subclass__(*args, **kwargs)
        dataseries_type_registry[cls.__name__] = cls

    def __len__(self):
        """Return number of datasets in series.

        Returns
        -------
        int
        """
        return len(self.datasets)

    def __getitem__(self, key):
        """
        Return dataset by index.
        Parameters
        ----------
        key

        Returns
        -------
        Dataset

        """
        return self.datasets[key]

    def info(self):
        """
        Print information about this datasetseries.

        Returns
        -------
        None
        """
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
    def data(self) -> None:
        """
        Dummy property to make user aware this is not a Dataset instance.
        Returns
        -------
        None

        """
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
        Return a string representation of the datasetseries object.

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
        """
        Check whether a given path is a valid path for this dataseries class.
        Parameters
        ----------
        path: str
            path to check
        args:
            (unused)
        kwargs:
            (unused)

        Returns
        -------
        CandidateStatus
        """
        return CandidateStatus.NO  # base class dummy

    @classmethod
    def from_directory(
        cls, path, *interface_args, datasetclass=None, pattern=None, **interface_kwargs
    ) -> "DatasetSeries":
        """
        Create a datasetseries instance from a directory.
        Parameters
        ----------
        path: str
            path to directory
        interface_args:
            arguments to pass to interface class
        datasetclass: Optional[Dataset]
            force class to use for dataset instances
        pattern:
            pattern to match files in directory
        interface_kwargs:
            keyword arguments to pass to interface class
        Returns
        -------
        DatasetSeries

        """
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
        """
        Get dataset by some metadata property. In the base class, we go by list index.

        Parameters
        ----------
        index: int
            index of dataset to get
        name: str
            name of dataset to get
        reltol:
            relative tolerance for metadata comparison
        kwargs:
            metadata properties to compare for selection

        Returns
        -------
        Dataset

        """
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

        # find candidates from metadata
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
                if isinstance(v, (int, float, np.integer, np.floating)):
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
        if len(candidates) == 0:
            raise ValueError("No candidate found for given metadata.")
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
        """
        Return metadata dictionary for this series.

        Returns
        -------
        Optional[dict]
            metadata dictionary
        """
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
        """
        Set metadata dictionary for this series, and save to disk.
        Parameters
        ----------
        dct: dict
            metadata dictionary

        Returns
        -------
        None

        """

        class ComplexEncoder(json.JSONEncoder):
            """
            JSON encoder that can handle numpy arrays and bytes.
            """

            def default(self, obj):
                """
                Default recipe for encoding objects.
                Parameters
                ----------
                obj: object
                    object to encode

                Returns
                -------
                object

                """
                if isinstance(obj, np.int64):
                    return int(obj)
                if isinstance(obj, np.int32):
                    return int(obj)
                if isinstance(obj, np.uint32):
                    return int(obj)
                if isinstance(obj, bytes):
                    return obj.decode("utf-8")
                if isinstance(obj, np.ndarray):
                    assert len(obj) < 1000  # do not want large obs here...
                    return list(obj)
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError as e:
                    print("obj failing json encoding:", obj)
                    raise e

        self._metadata = dct
        fp = self._metadatafile
        if not os.path.exists(fp):
            json.dump(dct, open(fp, "w"), cls=ComplexEncoder)


class DirectoryCatalog(object):
    """A catalog consisting of interface instances contained in a directory."""

    def __init__(self, path):
        """
        Initialize a directory catalog.

        Parameters
        ----------
        path: str
            path to directory
        """
        self.path = path


class HomogeneousSeries(DatasetSeries):
    """Series consisting of same-type data sets."""

    def __init__(self, path, **interface_kwargs):
        """
        Initialize a homogeneous series.
        Parameters
        ----------
        path:
            path to data
        interface_kwargs:
            keyword arguments to pass to interface class
        """
        super().__init__()
