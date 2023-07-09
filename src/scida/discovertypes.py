import logging
import os
from collections import Counter
from enum import Enum
from functools import reduce
from inspect import getmro
from typing import List, Union

from scida.config import get_simulationconfig
from scida.interfaces.mixins import CosmologyMixin
from scida.io import load_metadata
from scida.misc import check_config_for_dataset
from scida.registries import dataseries_type_registry, dataset_type_registry

log = logging.getLogger(__name__)


class CandidateStatus(Enum):
    NO = 0  # definitely not a candidate
    MAYBE = 1  # not sure yet
    YES = 2  # yes, this is a candidate


def _determine_mixins(path=None, metadata_raw=None):
    mixins = []
    assert not (path is None and metadata_raw is None)
    if metadata_raw is None:
        metadata_raw = load_metadata(path, fileprefix=None)
    z = metadata_raw.get("/Header", {}).get("Redshift", None)
    if z is not None:
        mixins.append(CosmologyMixin)
    return mixins


def _determine_type_from_simconfig(path, classtype="dataset", reg=None):
    if reg is None:
        reg = dict()
        reg.update(**dataset_type_registry)
        reg.update(**dataseries_type_registry)
    metadata_raw = dict()
    if classtype == "dataset":
        metadata_raw = load_metadata(path, fileprefix=None)
    candidates = check_config_for_dataset(metadata_raw, path=path)
    assert len(candidates) <= 1
    cls = None
    if len(candidates) == 1:
        simconf = get_simulationconfig()
        dstype = simconf.get("data", {}).get(candidates[0]).get("dataset_type", None)
        if isinstance(dstype, dict):
            # series or dataset based on past candidate
            if classtype == "series":
                cls = reg[dstype["series"]]
            elif classtype == "dataset":
                cls = reg[dstype["dataset"]]
            else:
                raise ValueError("Unknown class type '%s'." % classtype)
        elif isinstance(dstype, str):
            cls = reg[dstype]  # not split into series and dataset
        elif dstype is None:
            pass  # simply no dataset_type specified
        else:
            raise ValueError(
                "Unknown type of dataset config variable. content: '%s'" % dstype
            )
    return cls


def _determine_type(
    path: Union[str, os.PathLike],
    test_datasets: bool = True,
    test_dataseries: bool = True,
    strict: bool = False,
    catch_exception: bool = True,
    **kwargs
):
    available_dtypes: List[str] = []
    dtypes_status: List[CandidateStatus] = []
    reg = dict()
    if test_datasets:
        reg.update(**dataset_type_registry)
    if test_dataseries:
        reg.update(**dataseries_type_registry)

    for k, dtype in reg.items():
        valid = CandidateStatus.NO

        def is_valid(val):  # map mix of old bool and new Candidate Status enum to bool
            if isinstance(val, bool):
                if val:
                    return CandidateStatus.MAYBE
                else:
                    return CandidateStatus.NO
            else:
                return val

        if catch_exception:
            try:
                valid = is_valid(dtype.validate_path(path, **kwargs))
            except Exception as e:
                log.debug(
                    "Exception raised during validate_path of tested type '%s': %s"
                    % (k, e)
                )
        else:
            valid = is_valid(dtype.validate_path(path, **kwargs))
        if valid != CandidateStatus.NO:
            available_dtypes.append(k)
            dtypes_status.append(valid)  # will be YES or MAYBE

    if len(available_dtypes) == 0:
        raise ValueError("Unknown data type.")
    if len(available_dtypes) > 1:
        # TODO: Rethink how tu use MAYBE/YES information.
        # below lines not suitable for this.
        # good_matches = [
        #     k
        #     for k, v in zip(available_dtypes, dtypes_status)
        #     if v == CandidateStatus.YES
        # ]
        # if len(good_matches) >= 1:
        #     available_dtypes = (
        #         good_matches  # discard all MAYBEs as we have better options
        #     )
        # reduce candidates by looking at most specific ones.
        inheritancecounters = [Counter(getmro(reg[k])) for k in reg.keys()]
        mros = [getmro(reg[k]) for k in reg.keys()]
        lens = [len([m for m in mro if "Mixin" not in m.__name__]) for mro in mros]
        zp = zip(available_dtypes, lens, dtypes_status)
        lst = sorted(zp, key=lambda x: x[2].value)
        lst = sorted(lst, key=lambda x: x[1])
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
