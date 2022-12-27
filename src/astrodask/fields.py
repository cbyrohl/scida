import inspect
from collections.abc import MutableMapping

import dask.dataframe as dd

from .helpers_misc import get_kwargs


class DerivedField(object):
    def __init__(self, name, func, description=""):
        self.name = name
        self.func = func
        self.description = description


class FieldContainerCollection(MutableMapping):
    """A mutable collection of FieldContainers."""

    def __init__(self, types=None, derivedfields_kwargs=None):
        if types is None:
            types = []
        self.store = {
            k: FieldContainer(derivedfields_kwargs=derivedfields_kwargs) for k in types
        }
        self.derivedfields_kwargs = derivedfields_kwargs

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def new_container(self, key, **kwargs):
        self[key] = FieldContainer(
            **kwargs, derivedfields_kwargs=self.derivedfields_kwargs
        )

    def merge(self, collection, overwrite=True):
        assert isinstance(collection, FieldContainerCollection)
        for k in collection.store:
            if k not in self.store:
                self.store[k] = FieldContainer(
                    derivedfields_kwargs=self.derivedfields_kwargs
                )
            if overwrite:
                c1 = self.store[k]
                c2 = collection.store[k]
            else:
                c1 = collection.store[k]
                c2 = self.store[k]
            c1.fields.update(**c2.fields)
            c1.derivedfields.update(**c2.derivedfields)

    def merge_derivedfields(self, collection):
        # to be depreciated. Use self.merge.
        assert isinstance(collection, FieldContainerCollection)
        for k in collection.store:
            if k not in self.store:
                self.store[k] = FieldContainer(
                    derivedfields_kwargs=self.derivedfields_kwargs
                )
            else:
                self.store[k].derivedfields.update(**collection.store[k].derivedfields)

    def register_field(self, parttype, name=None, description=""):
        if parttype == "all":
            parttypes = self.store.keys()
        elif isinstance(parttype, list):
            parttypes = parttype
        else:
            return self.store[parttype].register_field(
                name=name, description=description
            )

        # we only construct field upon first call to it (default)
        def decorator(func, name=name, description=description):
            if name is None:
                name = func.__name__
            for p in parttypes:
                self.store[p].derivedfields[name] = DerivedField(
                    name, func, description=description
                )
            return func

        return decorator


class FieldContainer(MutableMapping):
    """A mutable collection of fields. Attempt to construct from derived fields recipes if needed."""

    def __init__(self, *args, derivedfields_kwargs=None, **kwargs):
        self.fields = {}
        self.fields.update(*args, **kwargs)
        self.derivedfields = {}
        self.derivedfields_kwargs = (
            derivedfields_kwargs if derivedfields_kwargs is not None else {}
        )

    def register_field(self, name=None, description=""):
        # we only construct field upon first call to it (default)
        def decorator(func, name=name, description=description):
            if name is None:
                name = func.__name__
            self.derivedfields[name] = DerivedField(name, func, description=description)
            return func

        return decorator

    def __setitem__(self, key, value):
        self.fields[key] = value

    def __getitem__(self, key):
        return self._getitem(key)

    @property
    def dataframe(self):
        dss = {}
        for k, v in self.items():
            if v.ndim > 1:
                for i in range(v.shape[1]):
                    dss[k + str(i)] = v[:, i]
            else:
                dss[k] = v
        dfs = [dd.from_dask_array(v, columns=[k]) for k, v in dss.items()]
        ddf = dd.concat(dfs, axis=1)
        return ddf

    def _getitem(self, key, force_derived=False, update_dict=True):
        if key in self.fields and not force_derived:
            return self.fields[key]
        else:
            if key in self.derivedfields:
                func = self.derivedfields[key].func
                accept_kwargs = inspect.getfullargspec(func).varkw is not None
                func_kwargs = get_kwargs(func)
                dkwargs = self.derivedfields_kwargs
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
                field = func(self, **kwargs)
                if update_dict:
                    self.fields[key] = field
                return field
            else:
                raise KeyError("Unknown field '%s'" % key)

    def __delitem__(self, key):
        if key in self.derivedfields:
            del self.derivedfields[key]
        del self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def get(self, key, value=None, allow_derived=True, force_derived=False):
        if key in self.derivedfields and not allow_derived:
            raise KeyError("Field '%s' is derived (allow_derived=False)" % key)
        else:
            try:
                return self._getitem(
                    key, force_derived=force_derived, update_dict=False
                )
            except KeyError:
                return value
