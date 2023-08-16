import logging
import sys

# need to import interfaces to register them in registry for loading
# TODO: automatize implicitly
from scida.convenience import load
from scida.customs.arepo.dataset import ArepoSnapshot
from scida.customs.arepo.MTNG.dataset import MTNGArepoSnapshot
from scida.customs.arepo.series import ArepoSimulation
from scida.customs.arepo.TNGcluster.dataset import TNGClusterSnapshot
from scida.customs.gadgetstyle.dataset import GadgetStyleSnapshot
from scida.customs.gizmo.dataset import GizmoSnapshot
from scida.customs.gizmo.series import GizmoSimulation
from scida.customs.rockstar.dataset import RockstarCatalog
from scida.customs.swift.dataset import SwiftSnapshot

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
