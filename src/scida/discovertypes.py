"""
Functionality to determine the dataset or dataseries type of a given path.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import reduce
from inspect import getmro

from scida.config import get_simulationconfig
from scida.io import load_metadata
from scida.misc import check_config_for_dataset
from scida.registries import (
    dataseries_type_registry,
    dataset_type_registry,
    mixin_type_registry,
)

log = logging.getLogger(__name__)


def is_valid_candidate(val):
    """
    Map legacy bools to candidate status while passing through rich results.
    Parameters
    ----------
    val: bool, CandidateStatus, or DetectionResult
        Value to map.

    Returns
    -------
    CandidateStatus or DetectionResult
    """
    if isinstance(val, bool):
        if val:
            return CandidateStatus.MAYBE
        else:
            return CandidateStatus.NO
    else:
        return val


class CandidateStatus(Enum):
    """
    Enum to indicate our confidence in a candidate.
    """

    # TODO: Rethink how tu use MAYBE/YES information.
    NO = 0  # definitely not a candidate
    MAYBE = 1  # not sure yet
    YES = 2  # yes, this is a candidate


class Confidence(IntEnum):
    """
    Quality of the evidence for a detection candidate.
    """

    NO = 0
    PATH_SHAPE = 10
    GENERIC_HEADER = 30
    FORMAT_MARKER = 60
    UNIQUE_MARKER = 80
    EXPLICIT = 100


class Specificity(IntEnum):
    """
    Narrowness of the class being claimed.
    """

    BASE = 0
    GENERIC = 10
    FAMILY = 20
    SUBFAMILY = 30
    INSTANCE = 40


@dataclass(frozen=True)
class DetectionResult:
    """
    Rich detection result separating evidence quality from class specificity.
    """

    confidence: Confidence
    specificity: Specificity | int | None = None
    evidence: tuple[str, ...] = ()

    @classmethod
    def no(cls, *evidence: str):
        return cls(Confidence.NO, evidence=evidence)

    @classmethod
    def match(
        cls,
        confidence: Confidence,
        *,
        specificity: Specificity | int | None = None,
        evidence: tuple[str, ...] | list[str] = (),
    ):
        return cls(confidence, specificity=specificity, evidence=tuple(evidence))

    @property
    def compatible(self) -> bool:
        return self.confidence != Confidence.NO


def _class_specificity(cls) -> int:
    specificity = getattr(cls, "_detection_specificity", None)
    if specificity is not None:
        return (
            specificity.value if isinstance(specificity, Specificity) else specificity
        )
    mro = getmro(cls)
    return 10 * len(
        [
            c
            for c in mro
            if c is not object
            and "Mixin" not in c.__name__
            and c.__name__ not in {"Dataset", "BaseDataset", "DatasetSeries"}
        ]
    )


def _candidate_to_detection_result(val, cls) -> DetectionResult:
    val = is_valid_candidate(val)
    if isinstance(val, DetectionResult):
        if val.specificity is None:
            return DetectionResult(
                val.confidence,
                specificity=_class_specificity(cls),
                evidence=val.evidence,
            )
        return val
    if val == CandidateStatus.NO:
        return DetectionResult.no()
    if val == CandidateStatus.MAYBE:
        return DetectionResult.match(
            Confidence.GENERIC_HEADER,
            specificity=_class_specificity(cls),
            evidence=("legacy_maybe",),
        )
    if val == CandidateStatus.YES:
        return DetectionResult.match(
            Confidence.FORMAT_MARKER,
            specificity=_class_specificity(cls),
            evidence=("legacy_yes",),
        )
    raise TypeError("Unknown candidate result '%s'." % val)


def _determine_mixins(path=None, metadata_raw=None):
    """
    Determine mixins for a given path or metadata_raw.
    Parameters
    ----------
    path: str or os.PathLike
        Path to dataset to load metadata from.
    metadata_raw: dict
        Raw metadata to use instead of loading from path.

    Returns
    -------
    List[Mixin]
        List of mixins to use for this dataset.


    """
    mixins = []
    assert not (path is None and metadata_raw is None)
    if metadata_raw is None:
        metadata_raw = load_metadata(path, fileprefix=None)
    # go through registry
    for mixin in mixin_type_registry.values():
        valid: bool = mixin.validate(metadata_raw)
        if valid:
            mixins.append(mixin)
    return mixins


def _determine_type_from_simconfig(path, classtype="dataset", reg=None):
    """
    Determine type from simulation config.

    Parameters
    ----------
    path: str or os.PathLike
        Path to dataset to load metadata from.
    classtype: str
        Type of class to determine. Either 'dataset' or 'series'.
    reg: dict
        Registry to use. If None, use the default registry.

    Returns
    -------
    type
        Type of dataset or series to use for this dataset.

    """
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
    path: str | os.PathLike,
    test_datasets: bool = True,
    test_dataseries: bool = True,
    strict: bool = False,
    catch_exception: bool = True,
    **kwargs,
):
    """
    Determine type of dataset or dataseries.

    Parameters
    ----------
    path: str or os.PathLike
        Path to dataset.
    test_datasets: bool
        Whether to test dataset types.
    test_dataseries: bool
        Whether to test dataseries types.
    strict: bool
        Whether to raise an error if multiple candidates are found.
    catch_exception: bool
        Whether to catch exceptions during validation of types.
    kwargs: dict
        Additional keyword arguments to pass to validate_path.

    Returns
    -------
    names: list[str]
        List of names of types that are candidates.
    types: list[type]
        List of classes that are candidates.

    """
    available_dtypes: list[str] = []
    dtypes_results: list[DetectionResult] = []
    result_by_name: dict[str, DetectionResult] = {}
    reg: dict[str, type] = dict()
    if test_datasets:
        reg.update(**dataset_type_registry)
    if test_dataseries:
        reg.update(**dataseries_type_registry)

    for k, dtype in reg.items():
        valid = DetectionResult.no()

        if catch_exception:
            try:
                valid = _candidate_to_detection_result(
                    dtype.validate_path(path, **kwargs), dtype
                )
            except Exception as e:
                log.debug(
                    "Exception raised during validate_path of tested type '%s': %s"
                    % (k, e)
                )
        else:
            valid = _candidate_to_detection_result(
                dtype.validate_path(path, **kwargs), dtype
            )
        if valid.compatible:
            available_dtypes.append(k)
            dtypes_results.append(valid)
            result_by_name[k] = valid

    if len(available_dtypes) == 0:
        raise ValueError("Unknown data type.")
    if len(available_dtypes) > 1:
        max_confidence = max(r.confidence.value for r in dtypes_results)
        available_dtypes = [
            k
            for k, r in zip(available_dtypes, dtypes_results)
            if r.confidence.value == max_confidence
        ]

        # reduce candidates by looking at most specific ones.
        inheritancecounters = [Counter(getmro(reg[k])) for k in reg.keys()]
        # try to find candidate that shows up only once across all inheritance trees.
        # => this will be the most specific candidate(s).
        count = reduce(lambda x, y: x + y, inheritancecounters)
        countdict = {k: count[reg[k]] for k in available_dtypes}
        mincount = min(countdict.values())
        available_dtypes = [k for k in available_dtypes if count[reg[k]] == mincount]
        specificities = {k: result_by_name[k].specificity for k in available_dtypes}
        max_specificity = max(
            s.value if isinstance(s, Specificity) else s for s in specificities.values()
        )
        available_dtypes = [
            k
            for k in available_dtypes
            if (
                specificities[k].value
                if isinstance(specificities[k], Specificity)
                else specificities[k]
            )
            == max_specificity
        ]
        if len(available_dtypes) > 1:
            # after looking for the most specific candidate(s), do we still have multiple?
            if strict:
                raise ValueError(
                    "Ambiguous data type. Available types:", available_dtypes
                )
            else:
                log.info("Note: Multiple dataset candidates: %s" % available_dtypes)
    return available_dtypes, [reg[k] for k in available_dtypes]
