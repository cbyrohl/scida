from __future__ import annotations

import inspect
from collections.abc import MutableMapping
from enum import Enum
from typing import Dict, Optional

import dask.array as da
import dask.dataframe as dd

from scida.helpers_misc import get_kwargs, sprint


class FieldType(Enum):
    INTERNAL = 1
    IO = 2
    DERIVED = 3


class FieldRecipe(object):
    def __init__(
        self, name, func=None, arr=None, description="", units=None, ftype=FieldType.IO
    ):
        if func is None and arr is None:
            raise ValueError("Need to specify either func or arr.")
        self.type = ftype
        self.name = name
        self.description = description
        self.units = units
        self.func = func
        self.arr = arr


class DerivedFieldRecipe(FieldRecipe):
    def __init__(self, name, func, description="", units=None):
        super().__init__(
            name,
            func=func,
            description=description,
            units=units,
            ftype=FieldType.DERIVED,
        )


class FieldContainer(MutableMapping):
    """A mutable collection of fields. Attempt to construct from derived fields recipes
    if needed."""

    def __init__(
        self,
        *args,
        fieldrecipes_kwargs=None,
        containers=None,
        aliases=None,
        withunits=False,
        parent: Optional[FieldContainer] = None,
        **kwargs,
    ):
        if aliases is None:
            aliases = {}
        if fieldrecipes_kwargs is None:
            fieldrecipes_kwargs = {}
        self.aliases = aliases
        self.name = kwargs.pop("name", None)
        self._fields: Dict[str, da.Array] = {}
        self._fields.update(*args, **kwargs)
        self._recipes = "bla"
        self._fieldrecipes = {}
        self._fieldlength = None
        self.fieldrecipes_kwargs = fieldrecipes_kwargs
        self.withunits = withunits
        self._containers: Dict[
            str, FieldContainer
        ] = dict()  # other containers as subgroups
        if containers is not None:
            for k in containers:
                self.add_container(k)
        self.internals = ["uid"]  # names of internal fields/groups
        self.parent = parent

    def info(self, level=0, name: Optional[str] = None) -> str:
        rep = ""
        length = self.fieldlength
        count = self.fieldcount
        if name is None:
            name = self.name
        ncontainers = len(self._containers)
        statstrs = []
        if length is not None and length > 0:
            statstrs.append("fields: %i" % count)
            statstrs.append("entries: %i" % length)
        if ncontainers > 0:
            statstrs.append("containers: %i" % ncontainers)
        if len(statstrs) > 0:
            statstr = ", ".join(statstrs)
            rep += sprint((level + 1) * "+", name, "(%s)" % statstr)
        for k in sorted(self._containers.keys()):
            v = self._containers[k]
            rep += v.info(level=level + 1)
        return rep

    def merge(self, collection, overwrite=True):
        if not isinstance(collection, FieldContainer):
            raise TypeError("Can only merge FieldContainers.")
        # TODO: support nested containers
        for k in collection._containers:
            if k not in self._containers:
                continue
            if overwrite:
                c1 = self._containers[k]
                c2 = collection._containers[k]
            else:
                c1 = collection._containers[k]
                c2 = self._containers[k]
            c1._fields.update(**c2._fields)
            c1._fieldrecipes.update(**c2._fieldrecipes)

    @property
    def fieldcount(self):
        rcps = set(self._fieldrecipes)
        flds = set([k for k in self._fields if k not in self.internals])
        ntot = len(rcps | flds)
        return ntot

    @property
    def fieldlength(self):
        if self._fieldlength is not None:
            return self._fieldlength
        fvals = self._fields.values()
        itr = iter(fvals)
        if len(fvals) == 0:
            # can we infer from recipes?
            if len(self._fieldrecipes) > 0:
                # get first recipe
                name = next(iter(self._fieldrecipes.keys()))
                first = self._getitem(name, evaluate_recipe=True)
            else:
                return None
        else:
            first = next(itr)
        if all(first.shape[0] == v.shape[0] for v in self._fields.values()):
            self._fieldlength = first.shape[0]
            return self._fieldlength
        else:
            return None

    def keys(
        self, withgroups=True, withrecipes=True, withinternal=False, withfields=True
    ):
        fieldkeys = []
        if withfields:
            fieldkeys = list(self._fields.keys())
            if not withinternal:
                for ikey in self.internals:
                    if ikey in fieldkeys:
                        fieldkeys.remove(ikey)
            if withrecipes:
                recipekeys = self._fieldrecipes.keys()
                fieldkeys = list(set(fieldkeys) | set(recipekeys))
        if withgroups:
            groupkeys = self._containers.keys()
            fieldkeys = list(set(fieldkeys) | set(groupkeys))
        return sorted(fieldkeys)

    def items(self, withrecipes=True, withfields=True, evaluate=True):
        return (
            (k, self._getitem(k, evaluate_recipe=evaluate))
            for k in self.keys(withrecipes=withrecipes, withfields=withfields)
        )

    def values(self, evaluate=True):
        return (self._getitem(k, evaluate_recipe=evaluate) for k in self.keys())

    def register_field(
        self,
        containernames=None,
        name: Optional[str] = None,
        description="",
        units=None,
    ):
        # we only construct field upon first call to it (default)
        # if to_containers, we register to the respective children containers
        containers = []
        if isinstance(containernames, list):
            containers = [self._containers[c] for c in containernames]
        elif containernames == "all":
            containers = self._containers.values()
        elif containernames is None:
            containers = [self]
        elif isinstance(containernames, str):  # just a single container as a string?
            containers.append(self._containers[containernames])
        else:
            raise ValueError("Unknown type.")

        def decorator(func, name=name, description=description, units=units):
            if name is None:
                name = func.__name__
            for container in containers:
                drvfields = container._fieldrecipes
                drvfields[name] = DerivedFieldRecipe(
                    name, func, description=description, units=units
                )
            return func

        return decorator

    def __setitem__(self, key, value):
        if key in self.aliases:
            key = self.aliases[key]
        if isinstance(value, FieldContainer):
            self._containers[key] = value
        elif isinstance(value, DerivedFieldRecipe):
            self._fieldrecipes[key] = value
        else:
            self._fields[key] = value

    def __getitem__(self, key):
        return self._getitem(key)

    def __iter__(self):
        return iter(self.keys())

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        Returns
        -------
        str
        """
        txt = ""
        txt += "FieldContainer[containers=%s, fields=%s]" % (
            len(self._containers),
            self.fieldcount,
        )
        return txt

    @property
    def dataframe(self):
        return self.get_dataframe()

    def get_dataframe(self, fields=None):
        dss = {}
        if fields is None:
            fields = self.keys()
        for k in fields:
            idim = None
            if k not in self.keys():
                # could still be an index two 2D dataset
                i = -1
                while k[i:].isnumeric():
                    i += -1
                i += 1
                if i == 0:
                    raise ValueError("Field '%s' not found" % k)
                idim = int(k[i:])
                k = k.split(k[i:])[0]
            v = self[k]
            assert v.ndim <= 2  # cannot support more than 2 here...
            if idim is not None:
                if v.ndim <= 1:
                    raise ValueError("No second dimensional index for %s" % k)
                if idim >= v.shape[1]:
                    raise ValueError(
                        "Second dimensional index %i not defined for %s" % (idim, k)
                    )

            if v.ndim > 1:
                for i in range(v.shape[1]):
                    if idim is None or idim == i:
                        dss[k + str(i)] = v[:, i]
            else:
                dss[k] = v
        dfs = [dd.from_dask_array(v, columns=[k]) for k, v in dss.items()]
        ddf = dd.concat(dfs, axis=1)
        return ddf

    def add_alias(self, alias, name):
        self.aliases[alias] = name

    def add_container(self, key, **kwargs):
        tkwargs = dict(**kwargs)
        if "name" not in tkwargs:
            tkwargs["name"] = key
        self._containers[key] = FieldContainer(
            fieldrecipes_kwargs=self.fieldrecipes_kwargs,
            withunits=self.withunits,
            parent=self,
            **tkwargs,
        )

    def _getitem(
        self, key, force_derived=False, update_dict=True, evaluate_recipe=True
    ):
        if key in self.aliases:
            key = self.aliases[key]
        if key in self._containers:
            return self._containers[key]
        if key in self._fields and not force_derived:
            return self._fields[key]
        else:
            if key in self._fieldrecipes:
                if not evaluate_recipe:
                    return self._fieldrecipes[key]
                func = self._fieldrecipes[key].func
                units = self._fieldrecipes[key].units
                accept_kwargs = inspect.getfullargspec(func).varkw is not None
                func_kwargs = get_kwargs(func)
                dkwargs = self.fieldrecipes_kwargs
                # first, we overwrite all optional arguments with class instance defaults where func kwarg is None
                kwargs = {
                    k: dkwargs[k]
                    for k in (
                        set(dkwargs)
                        & set([k for k, v in func_kwargs.items() if v is None])
                    )
                }
                # next, we add all optional arguments if func is accepting **kwargs and varname not yet in signature
                if accept_kwargs:
                    kwargs.update(
                        **{
                            k: v
                            for k, v in dkwargs.items()
                            if k not in inspect.getfullargspec(func).args
                        }
                    )
                # finally, instantiate field
                field = func(self, **kwargs)
                if self.withunits and units is not None:
                    field = field * units
                if update_dict:
                    self._fields[key] = field
                return field
            else:
                raise KeyError("Unknown field '%s'" % key)

    def __delitem__(self, key):
        if key in self._fieldrecipes:
            del self._fieldrecipes[key]
        if key in self._containers:
            del self._containers[key]
        elif key in self._fields:
            del self._fields[key]
        else:
            raise KeyError("Unknown key '%s'" % key)

    def __len__(self):
        return len(self.keys())

    def get(self, key, value=None, allow_derived=True, force_derived=False):
        if key in self._fieldrecipes and not allow_derived:
            raise KeyError("Field '%s' is derived (allow_derived=False)" % key)
        else:
            try:
                return self._getitem(
                    key, force_derived=force_derived, update_dict=False
                )
            except KeyError:
                return value


def walk_container(
    cntr, path="", handler_field=None, handler_group=None, withrecipes=False
):
    keykwargs = dict(withgroups=True, withrecipes=withrecipes)
    for ck in cntr.keys(**keykwargs):
        # we do not want to instantiate entry from recipe by calling cntr[ck] here
        entry = cntr[ck]
        newpath = path + "/" + ck
        if isinstance(entry, FieldContainer):
            if handler_group is not None:
                handler_group(entry, newpath)
            walk_container(
                entry,
                newpath,
                handler_field,
                handler_group,
                withrecipes=withrecipes,
            )
        else:
            if handler_field is not None:
                handler_field(entry, newpath, parent=cntr)
