import types
import hashlib

def hash_path(path):
    sha = hashlib.sha256()
    sha.update(path.strip("/ ").encode())
    return sha.hexdigest()[:16]


class RecursiveNamespace(types.SimpleNamespace):
    # see https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    def __init__(self, **kwargs):
        """Create a SimpleNamespace recursively"""
        super().__init__(**kwargs)
        self.__dict__.update({k: self.__elt(v) for k, v in kwargs.items()})

    def __elt(self, elt):
        """Recurse into elt to create leaf namespace objects"""
        if type(elt) is dict:
            return type(self)(**elt)
        if type(elt) in (list, tuple):
            return [self.__elt(i) for i in elt]
        return elt

