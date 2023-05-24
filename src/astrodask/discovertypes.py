import logging
import os
from collections import Counter
from functools import reduce
from inspect import getmro
from typing import List, Union

from astrodask.interfaces.mixins import CosmologyMixin
from astrodask.io import load_metadata
from astrodask.registries import dataseries_type_registry, dataset_type_registry

log = logging.getLogger(__name__)


def _determine_mixins(path=None, metadata_raw=None):
    mixins = []
    assert not (path is None and metadata_raw is None)
    if metadata_raw is None:
        metadata_raw = load_metadata(path, fileprefix=None)
    z = metadata_raw.get("/Header", {}).get("Redshift", None)
    if z is not None:
        mixins.append(CosmologyMixin)
    return mixins


def _determine_type(
    path: Union[str, os.PathLike],
    test_datasets: bool = True,
    test_dataseries: bool = True,
    strict: bool = False,
    catch_exception: bool = True,
    **kwargs
):
    available_dtypes: List[str] = []
    reg = dict()
    if test_datasets:
        reg.update(**dataset_type_registry)
    if test_dataseries:
        reg.update(**dataseries_type_registry)

    for k, dtype in reg.items():
        valid = False
        if catch_exception:
            try:
                valid = dtype.validate_path(path, **kwargs)
            except Exception as e:
                log.debug(
                    "Exception raised during validate_path of tested type '%s': %s"
                    % (k, e)
                )
        else:
            valid = dtype.validate_path(path, **kwargs)
        if valid:
            available_dtypes.append(k)

    if len(available_dtypes) == 0:
        raise ValueError("Unknown data type.")
    if len(available_dtypes) > 1:
        # reduce candidates by looking at most specific ones.
        inheritancecounters = [Counter(getmro(reg[k])) for k in reg.keys()]
        # try to find candidate that shows up only once across all inheritance trees.
        # => this will be the most specific candidate(s).
        count = reduce(lambda x, y: x + y, inheritancecounters)
        countdict = {k: count[reg[k]] for k in available_dtypes}
        mincount = min(countdict.values())
        available_dtypes = [k for k in available_dtypes if count[reg[k]] == mincount]
        if len(available_dtypes) > 1:
            # after looking for the most specific candidate(s), do we still have multiple?
            if strict:
                raise ValueError(
                    "Ambiguous data type. Available types:", available_dtypes
                )
            else:
                log.info("Note: Multiple dataset candidates: %s" % available_dtypes)
    return available_dtypes, [reg[k] for k in available_dtypes]
