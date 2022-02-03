import dask.array as da
import functools
import inspect
from .helpers_misc import get_kwargs


class DerivedField(object):
    def __init__(self, name, func, description=""):
        self.name = name
        self.func = func
        self.description = description


class FieldContainer(dict):
    def __init__(self, derivedfields_kwargs={}, **kwargs):
        super().__init__(self, **kwargs)
        self.derivedfields = {}
        self.derivedfields_kwargs = derivedfields_kwargs

    def register_field(self, name=None, description=""):
        # we only construct field upon first call to it (default)
        def decorator(func, name=name, description=description):
            if name is None:
                name = func.__name__
            self.derivedfields[name] = DerivedField(name, func, description=description)
            return func
        return decorator

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            if key in self.derivedfields:
                func = self.derivedfields[key].func
                accept_kwargs = inspect.getfullargspec(func).varkw is not None
                func_kwargs = get_kwargs(func)
                dkwargs = self.derivedfields_kwargs
                # first, we overwrite all optional arguments with class instance defaults where func kwarg is None
                kwargs = {k:dkwargs[k] for k in (set(dkwargs) & set([k for k,v in func_kwargs.items() if v is None]))}
                # next, we add all optional arguments if func is accepting **kwargs and varname not yet in signature
                if accept_kwargs:
                    kwargs.update(**{k: v for k, v in dkwargs.items() if k not in inspect.getfullargspec(func).args})
                self[key] = func(self,**kwargs)
                return super().__getitem__(key)
            else:
                raise KeyError("Unknown field '%s'" % key)

