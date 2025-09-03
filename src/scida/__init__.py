"""
Scida is a python package for reading and analyzing scientific big data.
"""

import logging
import sys

# need to import interfaces to register them in registry for loading
# TODO: automatize implicitly
from scida.convenience import load as load
from scida.customs.arepo.dataset import ArepoSnapshot as ArepoSnapshot
from scida.customs.arepo.MTNG.dataset import MTNGArepoSnapshot as MTNGArepoSnapshot
from scida.customs.arepo.series import ArepoSimulation as ArepoSimulation
from scida.customs.arepo.TNGcluster.dataset import TNGClusterSnapshot as TNGClusterSnapshot
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot as GadgetStyleSnapshot
from scida.customs.gizmo.dataset import GizmoSnapshot as GizmoSnapshot
from scida.customs.gizmo.series import GizmoSimulation as GizmoSimulation
from scida.customs.rockstar.dataset import RockstarCatalog as RockstarCatalog
from scida.customs.swift.dataset import SwiftSnapshot as SwiftSnapshot
from scida.customs.swift.series import SwiftSimulation as SwiftSimulation

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
