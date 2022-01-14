import types
import hashlib
import numpy as np

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


def make_serializable(v):
    # Attributes need to be JSON serializable. No numpy types allowed.
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return v

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['dmlowres']:
        return 2 # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    if str(partType).lower() in ["all"]:
        return -1
