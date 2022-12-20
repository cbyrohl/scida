import os


class BaseContainer(object):
    """A container for collection of interface instances"""

    def __init__(self, path):
        self.path = path


class DirectoryCatalog(object):
    """A catalog consisting of interface instances contained in a directory."""

    def __init__(self, path):
        self.path = path


class ArepoSimulation(BaseContainer):
    """A container for an arepo simulation."""

    def __init__(self, path):
        super().__init__(path)
        self.name = os.path.basename(self.path)
        self.outpath = os.path.join(self.path, "output")
        self.snapcat = DirectoryCatalog(self.outpath)
        self.groupcat = DirectoryCatalog(self.outpath)
