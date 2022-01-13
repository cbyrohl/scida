import dask.array as da
import functools


class DerivedField(object):
    def __init__(self, name, func):
        self.name = name
        self.func = func


class FieldContainer(dict):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.derivedfields = {}

    def register_field(self, name=None):
        def decorator(func,name=name):
            if name is None:
                name = func.__name__
            print(name,func)
            self.derivedfields[name] = DerivedField(name, func)
            return func

        return decorator

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            if key in self.derivedfields:
                self[key] = self.derivedfields[key].func(self)
                return super().__getitem__(key)
            else:
                raise KeyError("Unknown field '%s'" % key)

