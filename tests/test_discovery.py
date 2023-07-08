import pathlib
from typing import Type

from scida import ArepoSnapshot, MTNGArepoSnapshot
from scida.customs.arepo.dataset import ArepoCatalog
from scida.customs.arepo.MTNG.dataset import MTNGArepoCatalog
from scida.customs.arepo.series import ArepoSimulation
from scida.customs.gadgetstyle.dataset import SwiftSnapshot
from scida.customs.gizmo.dataset import GizmoSnapshot
from scida.customs.gizmo.series import GizmoSimulation
from scida.customs.rockstar.dataset import RockstarCatalog
from scida.discovertypes import _determine_type, _determine_type_from_simconfig
from scida.interface import Dataset
from scida.series import DatasetSeries
from tests.testdata_properties import require_testdata_path


def return_intended_stype(name) -> Type[DatasetSeries]:
    name = name.lower()
    if any(k in name for k in ["fire"]):
        return GizmoSimulation
    elif any(k in name for k in ["tng"]):
        return ArepoSimulation
    raise ValueError("Have not specified intended type for %s" % name)


def return_intended_dstype(name, simconf=False) -> Type[Dataset]:
    name = name.lower()
    if any(k in name for k in ["mtnggadget4"]):
        return None  # TODO: Gadget4 snapshot
    elif "mtngarepo" in name:
        if "group" in name:
            return MTNGArepoCatalog
        else:
            return MTNGArepoSnapshot
    elif any(k in name for k in ["tng", "illustris", "auriga"]):
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
    elif any(k in name for k in ["simba"]):
        # these are GizmoSnapshots. Current public Gizmo code has an identifying
        # attribute "GIZMO_VERSION" in "/Header"
        # unfortunately, this is not yet the case for fire2
        # hence fire2 might be identified GadgetStyleSnapshot as GizmoSnapshot
        # via the configuration files (which is not tested against here)
        return GizmoSnapshot
    elif "fire2" in name:
        return GizmoSnapshot
    elif any(k in name for k in ["rockstar"]):
        return RockstarCatalog

    raise ValueError("Have not specified intended type for %s" % name)


@require_testdata_path("series")
def test_load_check_seriestype(testdatapath):
    # this test intentionally ignores all additional hints from config files
    # catch_exception = True allows us better control for debugging
    check_type(testdatapath, catch_exception=True, classtype="series")


@require_testdata_path("interface")
def test_load_check_datasettype(testdatapath):
    # this test intentionally ignores all additional hints from config files
    # catch_exception = True allows us better control for debugging
    check_type(
        testdatapath,
        catch_exception=True,
        classtype="dataset",
        allow_more_specific=False,
    )


@require_testdata_path("interface")
def test_load_check_datasettype_simconf(testdatapath):
    check_type(
        testdatapath,
        catch_exception=True,
        classtype="dataset",
        allow_more_specific=False,
        simconf=True,
    )


def check_type(
    dpath,
    catch_exception=True,
    classtype="dataset",
    allow_more_specific=True,
    simconf=False,
):
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
    if simconf:
        cls_intended = return_intended_dstype(tname, simconf=True)
        cls_simconf = _determine_type_from_simconfig(path, classtype=classtype)
        if cls_simconf and not issubclass(cls, cls_simconf):
            cls = cls_simconf
    print("Intended type (1) vs determined type (2):")
    print("(1)", cls_intended)
    print("(2)", cls)
    # get MRO
    print("=== MRO (1) ===")
    mro = [c.__name__ for c in cls_intended.mro()]
    print(mro)
    print("=== MRO (2) ===")
    mro = [c.__name__ for c in cls.mro()]
    print(mro)

    if allow_more_specific:
        assert issubclass(cls, cls_intended)
    else:
        assert cls == cls_intended
