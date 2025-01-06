import h5py
import numpy as np


def write_hdf5flat_testfile(path):
    # create a flat hdf5 file with a multiple datasets
    with h5py.File(path, "w") as hf:
        hf["field1"] = np.arange(10)
        hf["field2"] = np.arange(10)


class DummyGadgetFile:
    def __init__(self):
        self.header = dict()
        self.parameters = dict()
        self.particles = dict()
        self.path = None  # path to the file after first write
        self.create_dummyheader()
        self.create_dummyparameters()

    def create_dummyheader(self):
        header = self.header
        attrs_float_keys = ["Time", "Redshift", "BoxSize"]
        for key in attrs_float_keys:
            header[key] = 0.0
        attrs_int_keys = [
            "NumFilesPerSnapshot",
            "Flag_Sfr",
            "Flag_Cooling",
            "Flag_StellarAge",
            "Flag_Metals",
            "Flag_Feedback",
            "Flag_DoublePrecision",
            "Flag_IC_Info",
            "Flag_LptInitCond",
        ]
        for key in attrs_int_keys:
            header[key] = 0
        # set cosmology
        header["Omega0"] = 0.3
        header["OmegaBaryon"] = 0.04
        header["OmegaLambda"] = 0.7
        header["HubbleParam"] = 0.7

    def create_dummyparameters(self):
        pass

    def write(self, path):
        with h5py.File(path, "w") as hf:
            if self.header is not None:
                grp = hf.create_group("Header")
                for key in self.header:
                    grp.attrs[key] = self.header[key]
            if self.parameters is not None:
                grp = hf.create_group("Parameters")
                for key in self.parameters:
                    grp.attrs[key] = self.parameters[key]
            for key in self.particles:
                gname = key
                hf.create_group(gname)
                for field in self.particles[key]:
                    hf[gname].create_dataset(field, data=self.particles[key][field])
        self.path = path


class DummyGadgetSnapshotFile(DummyGadgetFile):
    """Creates a dummy GadgetStyleSnapshot in memory that can be modified and saved to disk."""

    def __init__(self):
        super().__init__()
        # by default, create a dummy snapshot structure
        self.create_dummyheader()
        self.create_dummyfieldcontainer()

    def create_dummyheader(self, lengths=None):
        super().create_dummyheader()
        if lengths is None:
            lengths = [1, 1, 0, 1, 1, 1]
        header = self.header
        header["NumPart_ThisFile"] = lengths
        header["NumPart_Total"] = lengths
        header["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        header["MassTable"] = [0, 0, 0, 0, 0, 0]

    def create_dummyparameters(self):
        pass

    def create_dummyfieldcontainer(self, lengths=None):
        pdata = self.particles
        if lengths is None:
            lengths = [1, 1, 0, 1, 1, 1]
        for i in range(6):
            n = lengths[i]
            gname = "PartType%i" % i
            pdata[gname] = dict()
            pdata[gname]["Coordinates"] = np.zeros((n, 3), dtype=float)
            pdata[gname]["ParticleIDs"] = np.zeros((n,), dtype=int)
            pdata[gname]["Velocities"] = np.zeros((n, 3), dtype=float)
            if gname == "PartType0":
                pdata[gname]["Masses"] = np.zeros((n,), dtype=float)
                pdata[gname]["Density"] = np.zeros((n,), dtype=float)
        self.particles = pdata


class DummyGadgetCatalogFile(DummyGadgetFile):
    def __init__(self):
        super().__init__()
        self.create_dummyheader()
        self.create_dummyfieldcontainer()

    def create_dummyheader(self, lengths=None):
        super().create_dummyheader()
        if lengths is None:
            lengths = [1, 1]  # halos and subgroups
        header = self.header
        header["Ngroups_ThisFile"] = lengths[0]
        header["Ngroups_Total"] = lengths[0]
        header["Nsubgroups_ThisFile"] = lengths[1]
        header["Nsubgroups_Total"] = lengths[1]

    def create_dummyfieldcontainer(self, lengths=None):
        pdata = self.particles
        if lengths is None:
            lengths = [1, 1]
        # Groups
        grp = dict()
        pdata["Group"] = grp
        ngroups = lengths[0]
        grp["GroupPos"] = np.zeros((ngroups, 3), dtype=float)
        grp["GroupMass"] = np.zeros((ngroups,), dtype=float)
        grp["GroupVel"] = np.zeros((ngroups, 3), dtype=float)
        grp["GroupLenType"] = np.ones((ngroups, 6), dtype=int)
        grp["GroupLen"] = grp["GroupLenType"].sum(axis=1)
        # Subhalos
        sh = dict()
        pdata["Subhalo"] = sh
        nsubs = lengths[1]
        sh["SubhaloPos"] = np.zeros((nsubs, 3), dtype=float)
        sh["SubhaloMass"] = np.zeros((nsubs,), dtype=float)
        sh["SubhaloVel"] = np.zeros((nsubs, 3), dtype=float)
        sh["SubhaloLenType"] = np.ones((nsubs, 6), dtype=int)
        sh["SubhaloLen"] = sh["SubhaloLenType"].sum(axis=1)
        sh["SubhaloGrNr"] = np.zeros((nsubs,), dtype=int)


class DummyTNGFile(DummyGadgetSnapshotFile):
    def __init__(self):
        super().__init__()
        self.create_dummyheader()
        self.create_dummyparameters()

    def create_dummyheader(self, lengths=None):
        super().create_dummyheader(lengths)
        header = self.header
        header["BoxSize"] = 35000.0
        header["UnitLength_in_cm"] = 3.085678e21
        header["UnitMass_in_g"] = 1.989e43
        header["UnitVelocity_in_cm_per_s"] = 100000.0

    def create_dummyparameters(self):
        super().create_dummyparameters()
        params = self.parameters
        icf = "/zhome/academic/HLRS/lha/zahapill/ics/ics_illustrisTNGboxes/L35n2160TNG/output/ICs"
        params["InitCondFile"] = icf

    def create_dummyfieldcontainer(self, lengths=None):
        super().create_dummyfieldcontainer()
        pdata = self.particles
        ngas = pdata["PartType0"]["Coordinates"].shape[0]
        extra_keys = ["StarFormationRate"]
        for key in extra_keys:
            pdata["PartType0"][key] = np.zeros((ngas,), dtype=float)


def write_gadget_testfile(path):
    # create a dummy Gadget snapshot
    dummy = DummyGadgetSnapshotFile()
    dummy.write(path)
