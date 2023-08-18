import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pytest
import yaml

from scida.config import get_config

# if false, testdata that is not available will be explicitly skipped
# unav aiet tests wildataset l be constructed but skipped if true (otherwise no construct)
silent_unavailable = os.environ.get("SCIDA_TESTDATA_SILENT_UNAVAILABLE", "TRUE")
silent_unavailable = silent_unavailable.upper() in ["TRUE", "1"]

datapath = os.path.expanduser(get_config().get("testdata_path", os.getcwd()))
testdataskip = os.environ.get("SCIDA_TESTDATA_SKIP", "")


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


# read testdata properties from yaml file
with open(os.path.join(os.path.dirname(__file__), "testdata.yaml"), "r") as file:
    testdata_properties = yaml.safe_load(file).get("testdata", {})
    for name, properties in testdata_properties.items():
        add_testdata_entry(name, **properties)
        print(name, properties)


testdataskip = testdataskip.split()
testdata_local = []
for k, v in testdatadict.items():
    print(k, v)
    if os.path.exists(v.path) and k not in testdataskip:
        testdata_local.append(k)
        print("exists")
    # if not os.path.exists(testdatadict_entry.path):
    #    if silent_unavailable:
    #        testdatadict_entry.marks.append(skip_unavail)
    #    else:
    #        raise ValueError("Testdata '%s' not available." % testdatadict_entry.path)

# if datapath is not None:
#    testdata_local_fns = [f for f in os.listdir(datapath) if f not in testdataskip]
#    path_to_name_dict = {
#        v.path.split(datapath)[-1].split("/")[-1]: k for k, v in testdatadict.items()
#    }
#    testdata_local = [path_to_name_dict.get(k, k) for k in testdata_local_fns]


# add_testdata_entry(
#    "SIMBA50converted_group", ["interface", "areposnapshot_withcatalog|C|1|2"]
# )


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


def get_testdata_params_ids(
    datatype,
    only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    exclude_substring: bool = False,
):
    """
    Get testdata parameters and ids for a given datatype.
    Parameters
    ----------
    datatype: str
    only: Optional[List[str]]
        Only use these testdata entries (by name)
    exclude: Optional[List[str]]
        Exclude these testdata entries (by name)
    exclude_substring: bool
        If True, exclude if any of the substrings is in the name (only if exclude is not None)

    Returns
    -------

    """
    params, ids = [], []
    for k, td in testdatadict.items():
        if only is not None and k not in only:
            continue  # not interested in this dataset
        if exclude is not None:
            if exclude_substring:  # exclude if any of the substrings is in the name
                if any([e in k for e in exclude]):
                    continue
            else:
                if k in exclude:
                    continue
        for tp in td.types:
            if tp is None:
                continue
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
    params, _ = get_testdata_params_ids(datatype, **kwargs)
    return params


def get_ids(datatype, **kwargs):
    _, ids = get_testdata_params_ids(datatype, **kwargs)
    return ids


def require_testdata(
    name, scope="function", only=None, specific=True, nmax=None, **kwargs
):
    """

    Parameters
    ----------
    name: str
        name of the testdata type
    scope: str
        pytest scope
    only: list of str
        only use these testdata entries
    specific: bool
        if True, add the name of the testdata type to the fixture name
    nmax: int
        only use the first 'nmax' entries

    Returns
    -------

    """
    fixturename = "testdata"
    if specific:
        fixturename += "_" + name
    params = get_params(name, only=only, **kwargs)
    ids = get_ids(name, only=only, **kwargs)
    if isinstance(nmax, int):
        params = params[:nmax]
        ids = ids[:nmax]
    return pytest.mark.parametrize(
        fixturename,
        params,
        ids=ids,
        indirect=True,
        scope=scope,
    )


def require_testdata_path(
    name,
    scope: str = "function",
    specific: bool = False,
    only: Optional[List[str]] = None,
    **kwargs,
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
