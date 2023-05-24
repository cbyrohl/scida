import logging
import sys

# need to import interfaces to register them in registry for loading
# TODO: automatize implicitly
from astrodask.convenience import load
from astrodask.interfaces.arepo import ArepoSnapshot
from astrodask.interfaces.gadgetstyle import GadgetStyleSnapshot
from astrodask.interfaces.illustris import IllustrisSnapshot
from astrodask.series import ArepoSimulation

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
