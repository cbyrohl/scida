import pathlib
from typing import Type

from scida import ArepoSnapshot
from scida.customs.arepo.dataset import ArepoCatalog
from scida.customs.arepo.series import ArepoSimulation
from scida.customs.gizmo.series import GizmoSimulation
from scida.customs.rockstar.dataset import RockstarCatalog
from scida.discovertypes import _determine_type
from scida.interface import Dataset
from scida.interfaces.gadgetstyle import GadgetStyleSnapshot, SwiftSnapshot
from scida.series import DatasetSeries
from tests.testdata_properties import require_testdata_path


def return_intended_stype(name) -> Type[DatasetSeries]:
    name = name.lower()
    if any(k in name for k in ["fire"]):
        return GizmoSimulation
    elif any(k in name for k in ["tng"]):
        return ArepoSimulation
    raise ValueError("Have not specified intended type for %s" % name)


def return_intended_dstype(name) -> Type[Dataset]:
    name = name.lower()
    if any(k in name for k in ["tng", "illustris", "auriga"]):
        if "group" in name:
            return ArepoCatalog
        else:
            return ArepoSnapshot
    elif any(k in name for k in ["eagle"]):
        return ArepoSnapshot  # TODO: Change once we abstracted a GadgetSnapshot class
    elif any(k in name for k in ["gaia"]):
        return Dataset
    elif any(k in name for k in ["swift"]):
        return SwiftSnapshot
    elif any(k in name for k in ["simba", "fire2"]):
        # these are GizmoSnapshots. However, right now Gizmo has no identifying metadata,
        # thus cannot infer solely from generic metadata
        # [one can use config files to identify simulations and map to interface]
        return GadgetStyleSnapshot
    elif any(k in name for k in ["rockstar"]):
        return RockstarCatalog

    raise ValueError("Have not specified intended type for %s" % name)


@require_testdata_path("series")
def test_load_check_seriestype(testdatapath):
    # this test intentionally ignores all additional hints from config files
    # catch_exception = True allows us better control for debugging
    check_type(testdatapath, catch_exception=True, classtype="series")


@require_testdata_path("interface")
def test_load_check_interfacetype(testdatapath):
    # this test intentionally ignores all additional hints from config files
    # catch_exception = True allows us better control for debugging
    check_type(testdatapath, catch_exception=True, classtype="dataset")


def check_type(dpath, catch_exception=True, classtype="dataset"):
    path = pathlib.Path(dpath)
    pp = path.parent.parent.parent
    tname = str(path.relative_to(pp))
    names, classes = _determine_type(path, catch_exception=catch_exception)
    if classtype == "dataset":
        cls_intended = return_intended_dstype(tname)
    elif classtype == "series":
        cls_intended = return_intended_stype(tname)
    else:
        raise ValueError("classtype must be either 'dataset' or 'series'")
    if len(names) > 1:
        print("Multiple types found for '%s': %s" % (path, names))
    cls = classes[0]
    print("Intended type (1) vs determined type (2):")
    print("(1)", cls_intended)
    print("(2)", cls)
    # right now, we allow more specific types than intended to be returned
    assert issubclass(cls, cls_intended)
