import h5py
import numpy as np


def write_hdf5flat_testfile(path):
    # create a flat hdf5 file with a multiple datasets
    with h5py.File(path, "w") as hf:
        hf["field1"] = np.arange(10)
        hf["field2"] = np.arange(10)


def write_gadget_testfile(path):
    with h5py.File(path, "w") as hf:
        # a bunch of attributes/groups we let GitHub Copilot create
        hf.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, 0, 0]
        hf.attrs["NumPart_Total"] = [0, 0, 0, 0, 0, 0]
        hf.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        hf.attrs["MassTable"] = [0, 0, 0, 0, 0, 0]
        hf.attrs["Time"] = 0.0
        hf.attrs["Redshift"] = 0.0
        hf.attrs["BoxSize"] = 0.0
        hf.attrs["NumFilesPerSnapshot"] = 0
        hf.attrs["Omega0"] = 0.0
        hf.attrs["OmegaLambda"] = 0.0
        hf.attrs["HubbleParam"] = 0.0
        hf.attrs["Flag_Sfr"] = 0
        hf.attrs["Flag_Cooling"] = 0
        hf.attrs["Flag_StellarAge"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0
        hf.attrs["Flag_Metals"] = 0
        hf.attrs["Flag_Feedback"] = 0
        hf.attrs["Flag_DoublePrecision"] = 0
        hf.attrs["Flag_IC_Info"] = 0
        hf.attrs["Flag_LptInitCond"] = 0

        lens = []
        for i in range(6):
            n = 1
            if i == 2:
                n = 0
            gname = "PartType%i" % i
            hf.create_group(gname)
            hf[gname].create_dataset("Coordinates", (n, 3), dtype="f")
            hf[gname].create_dataset("ParticleIDs", (n,), dtype="i")
            hf[gname].create_dataset("Velocities", (n, 3), dtype="f")
            lens.append(n)

        hf.create_group("Header")
        hf["Header"].attrs["NumPart_ThisFile"] = lens
        hf["Header"].attrs["NumPart_Total"] = lens
