# need to import interfaces to register them in registry for loading
# TODO: automatize implicitly
from astrodask.convenience import load
from astrodask.interface import BaseSnapshot
from astrodask.interfaces.arepo import ArepoSnapshot
from astrodask.interfaces.illustris import IllustrisSnapshot
from astrodask.series import ArepoSimulation
