import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pytest

from astrodask.config import get_config

silent_unavailable = False  # unav aiet tests wildataset l be constructed but skipped if true (otherwise no construct)

datapath = get_config().get("testdata_path", os.getcwd())
print(datapath)
testdataskip = os.environ.get("ASTRODASK_TESTDATA_SKIP", "")


@dataclass
class TestDataProperties:
    path: str
    types: List[str] = field(default_factory=list)
    marks: List[str] = field(default_factory=list)


skip_unavail = pytest.mark.skip(reason="testdata not available")


# in types, we can mark a dataset to be passed with others by using "type|A|B|C":
# A is the identifier across testdata entries to be grouped together,
# B the integer of the order of arguments
# C the total number of arguments
testdatadict: Dict[str, TestDataProperties] = {}


def add_testdata_entry(name, types=None, marks=None, fn=None):
    if types is None:
        types = []
    if marks is None:
        marks = []
    if fn is None:
        path = os.path.join(datapath, name)
    elif not os.path.isabs(fn):
        path = os.path.join(datapath, fn)
    else:
        path = fn
    if name in testdatadict:
        raise ValueError("Testdata '%s' already exists." % name)
    testdatadict[name] = TestDataProperties(path, types, marks)


add_testdata_entry(
    "TNG50-4_snapshot",
    ["interface", "areposnapshot", "areposnapshot_withcatalog|A|0|2"],
)
add_testdata_entry(
    "TNG50-3_snapshot",
    ["interface", "areposnapshot", "areposnapshot_withcatalog|B|0|2"],
)
add_testdata_entry(
    "SIMBA50converted_snapshot",
    ["interface", "areposnapshot", "areposnapshot_withcatalog|C|0|2"],
)
add_testdata_entry(
    "AURIGA66_snapshot",
    ["interface", "areposnapshot", "areposnapshot_withcatalog|D|0|2"],
)
add_testdata_entry("EAGLEsmall_snapshot", ["interface"], fn="EAGLEsmall.hdf5")
add_testdata_entry("TNG50-4_group", ["interface", "areposnapshot_withcatalog|A|1|2"])
# add_testdata_entry(
#    "SIMBA50converted_group", ["interface", "areposnapshot_withcatalog|C|1|2"]
# )
add_testdata_entry(
    "TNG50-3_group",
    [
        "interface",
        "illustrissnapshot",
        "areposnapshot_withcatalog|B|1|2",
        "illustrisgroup",
    ],
)
add_testdata_entry(
    "AURIGA66_group",
    ["interface", "illustrissnapshot", "areposnapshot_withcatalog|D|1|2"],
)
add_testdata_entry("TNGvariation_simulation", ["container", "areposimulation"])


testdataskip = testdataskip.split()
testdata_local = []
if datapath is not None:
    testdata_local_fns = [f for f in os.listdir(datapath) if f not in testdataskip]
    path_to_name_dict = {
        v.path.split(datapath)[-1].split("/")[-1]: k for k, v in testdatadict.items()
    }
    testdata_local = [path_to_name_dict.get(k, k) for k in testdata_local_fns]


def parse_typestring(typestr):
    lst = typestr.split("|")
    assert len(lst) in [1, 3]
    if len(lst) == 1:
        return lst[0]
    else:
        pass


def get_testdata_partners(typestr):
    lst = typestr.split("|")
    dct = {}
    for k, td in testdatadict.items():
        for t in td.types:
            splt = t.split("|")
            if splt[0] != lst[0]:
                continue  # wrong type
            if splt[1] != lst[1]:
                continue  # wrong partner
            dct[int(splt[2])] = [k, td]
    assert max(dct.keys()) + 1 == len(dct)
    partners = [dct[i] for i in range(len(dct))]
    partners_name, partners_entry = map(list, zip(*partners))
    return partners_name, partners_entry


def init_param_from_testdata(
    entries: Union[List[TestDataProperties], TestDataProperties], extramarks=None
):
    if extramarks is None:
        extramarks = []
    if not isinstance(entries, list):
        entries = [entries]
    marks = set([m for entry in entries for m in entry.marks])
    p = [entry.path for entry in entries]
    if len(p) == 1:
        p = p[0]
    param = pytest.param(p, marks=extramarks + [getattr(pytest.mark, m) for m in marks])
    return param


def get_testdata_params_ids(datatype, only: Optional[List[str]] = None):
    params, ids = [], []
    for k, td in testdatadict.items():
        if only is not None and k not in only:
            continue  # not interested in this dataset
        for tp in td.types:
            tsplit = tp.split("|")
            tname = tsplit[0]
            if tname != datatype:
                continue  # wrong datatype
            if len(tsplit) == 1:
                extramarks = []
                if k not in testdata_local:
                    if silent_unavailable:
                        continue  # do not add this testdata to stay silent
                    else:
                        extramarks += [skip_unavail]
                param = init_param_from_testdata(td, extramarks=extramarks)
                params.append(param)
                ids.append(k)
            else:
                assert len(tsplit) == 4  # required syntax for types with "|"
                if int(tsplit[2]) > 0:
                    continue  # do not want to double count
                partners_name, partners_entry = get_testdata_partners(tp)
                if len(partners_name) != int(tsplit[3]):
                    print("Incomplete dataset composite (add definitions)")
                    continue  # some dataset definition is missing for a full composite, ignore
                extramarks = []
                alllocal = all([p in testdata_local for p in partners_name])
                if not alllocal:
                    if silent_unavailable:
                        continue  # do not add this testdata to stay silent
                    else:
                        extramarks += [skip_unavail]
                param = init_param_from_testdata(partners_entry, extramarks=extramarks)
                params.append(param)
                ids.append("+".join(partners_name))
    return params, ids


def get_params(datatype, **kwargs):
    params, ids = get_testdata_params_ids(datatype, **kwargs)
    return params


def get_ids(datatype, **kwargs):
    params, ids = get_testdata_params_ids(datatype, **kwargs)
    return ids


def require_testdata(name, scope="function", only=None, specific=True):
    fixturename = "testdata"
    if specific:
        fixturename += "_" + name
    return pytest.mark.parametrize(
        fixturename,
        get_params(name, only=only),
        ids=get_ids(name, only=only),
        indirect=True,
        scope=scope,
    )


def require_testdata_path(
    name,
    scope: str = "function",
    specific: bool = False,
    only: Optional[List[str]] = None,
):
    fixturename = "testdatapath"
    if specific:
        fixturename += "_" + name
    return pytest.mark.parametrize(
        fixturename,
        get_params(name, only=only),
        ids=get_ids(name, only=only),
        scope=scope,
    )
