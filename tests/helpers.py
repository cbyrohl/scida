import h5py
import numpy as np


def write_hdf5flat_testfile(path):
    # create a flat hdf5 file with a multiple datasets
    with h5py.File(path, "w") as hf:
        hf["field1"] = np.arange(10)
        hf["field2"] = np.arange(10)


class DummyGadgetFile:
    """Creates a dummy GadgetStyleSnapshot in memory that can be modified and saved to disk."""
    def __init__(self):
        self.header = dict()
        self.parameters = dict()
        self.particles = dict()
        # by default, create a dummy snapshot structure
        self.create_dummyheader()
        self.create_dummyparticles()
        self.path = None  # will be set on write

    def create_dummyheader(self, lengths=None):
        if lengths is None:
            lengths = [1, 1, 0, 1, 1, 1]
        header = self.header
        header["NumPart_ThisFile"] = lengths
        header["NumPart_Total"] = lengths
        header["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        header["MassTable"] = [0, 0, 0, 0, 0, 0]
        attrs_float_keys = ["Time", "Redshift", "BoxSize"]
        for key in attrs_float_keys:
            header[key] = 0.0
        attrs_int_keys = ["NumFilesPerSnapshot", "Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals",
                          "Flag_Feedback", "Flag_DoublePrecision", "Flag_IC_Info", "Flag_LptInitCond"]
        for key in attrs_int_keys:
            header[key] = 0
        # set "Omega0", "OmegaLambda", "HubbleParam"
        header["Omega0"] = 0.3
        header["OmegaLambda"] = 0.7
        header["HubbleParam"] = 0.7

    def create_dummyparameters(self):
        pass

    def create_dummyparticles(self, lengths=None):
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

    def write(self, path):
        with h5py.File(path, "w") as hf:
            grp = hf.create_group("Header")
            for key in self.header:
                grp.attrs[key] = self.header[key]
            grp = hf.create_group("Parameters")
            for key in self.parameters:
                grp.attrs[key] = self.parameters[key]
            for key in self.particles:
                gname = key
                hf.create_group(gname)
                for field in self.particles[key]:
                    hf[gname].create_dataset(field, data=self.particles[key][field])
        self.path = path


class DummyTNGFile(DummyGadgetFile):
    def __init__(self):
        super().__init__()
        self.create_dummyheader()
        self.create_dummyparameters()

    def create_dummyheader(self, lengths=None):
        super().create_dummyheader(lengths)
        header = self.header
        header["UnitLength_in_cm"] = 3.085678e+21
        header["UnitMass_in_g"] = 1.989e+43
        header["UnitVelocity_in_cm_per_s"] = 100000.0

    def create_dummyparameters(self):
        super().create_dummyparameters()
        params = self.parameters
        icf = "/zhome/academic/HLRS/lha/zahapill/ics/ics_illustrisTNGboxes/L35n2160TNG/output/ICs"
        params["InitCondFile"] = icf

    def create_dummyparticles(self, lengths=None):
        super().create_dummyparticles()
        pdata = self.particles
        ngas = pdata["PartType0"]["Coordinates"].shape[0]
        extra_keys = ["StarFormationRate"]
        for key in extra_keys:
            pdata["PartType0"][key] = np.zeros((ngas,), dtype=float)


def write_gadget_testfile(path):
    # create a dummy Gadget snapshot
    dummy = DummyGadgetFile()
    dummy.write(path)
