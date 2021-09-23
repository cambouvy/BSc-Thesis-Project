import os
import copy
import warnings
from warnings import warn
from collections import OrderedDict
from itertools import chain
import array
import gc
import re
import logging

import pandas as pd
import numpy as np
import shutil
import xarray as xr
""" 'Glueing' all the files outputted from QuaLiKiz together. The variables not needed for this project are dropped"""

if __name__ == "__main__":
    path = "/marconi_work/FUA35_GKNN/camille/edgerun8_netcdf/"
    files = os.listdir(path)

    allFiles = []

    for fileName in files:
        if fileName.endswith(".nc"):
            ds = xr.open_dataset(path + fileName)

            # Remove most dimn variables
            ds = ds.drop_vars(["Lcirce", "Lcirci", "Lecirce", "Lecirci", "Lepiege", "Lpiege", "Lpiegi", "Lvcirci", "Lvpiegi",
                               "rfdsol", "ifdsol", "rmodeshift", "imodeshift", "rmodewidth", "imodewidth", "rsol", "isol",
                               "rsolflu", "isolflu", "ntor", "distan", "kperp2", "kymaxETG", "kymaxITG", "Lcircgne", "Lcircgni",
                               "Lcircgte", "Lcircgti", "Lcircgui", "Lcircce", "Lcircci", "Lpiegce", "Lpiegci", "Lpieggte",
                               "Lpieggti", "Lpieggui", "Lpieggne", "Lpieggni", "Lepiegi"])


            #Remove SI variables
            ds = ds.drop_vars(["gam_SI", "ome_SI", "pfe_SI", "pfi_SI", "efe_SI", "efi_SI", "vfi_SI", "efeITG_SI", "efeTEM_SI",
                              "efiITG_SI", "efiTEM_SI", "pfeTEM_SI", "pfiITG_SI", "pfiTEM_SI", "vfiITG_SI", "vfiTEM_SI",
                              "dfe_SI", "dfi_SI", "vte_SI", "vti_SI", "vce_SI", "vci_SI", "vri_SI", "dfeITG_SI", "dfeTEM_SI",
                              "dfiITG_SI", "dfiTEM_SI", "vteITG_SI", "vteTEM_SI", "vriITG_SI", "vriTEM_SI", "vtiITG_SI",
                              "vtiTEM_SI", "vceITG_SI", "vceTEM_SI", "vciITG_SI", "vciTEM_SI"])



            # for var_name, value in ds.data_vars.items():
            #     if var_name.endswith("SI"):
            #         ds = ds.drop_vars(var_name)

            allFiles.append(ds)

    print("Went through all files, start concat")
    bigFile = xr.concat(allFiles, dim="dimx")
    bigFilePath = path + "glued_netcdf.nc"
    bigFile.to_netcdf(path=bigFilePath)





