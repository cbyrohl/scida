import logging
import sys

# need to import interfaces to register them in registry for loading
# TODO: automatize implicitly
from astrodask.convenience import load
from astrodask.customs.arepo.dataset import ArepoSnapshot
from astrodask.customs.arepo.series import ArepoSimulation
from astrodask.interfaces.gadgetstyle import GadgetStyleSnapshot

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
