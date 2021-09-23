import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" Returns statistics (mean/median) of the training set """

store = pd.HDFStore("/m100_work/FUAC5_GKNN/camille/results2/training_gen5_7D_pedformreg8_filter11.h5.1")
efeTEM_GB = store["/output/efeTEM_GB"].mean()
efiTEM_GB_div_efeTEM_GB_list = store["/output/efiTEM_GB_div_efeTEM_GB"]
efiTEM_GB_div_efeTEM_GB = store["/output/efiTEM_GB_div_efeTEM_GB"].mean()
pfeTEM_GB_div_efeTEM_GB = store["/output/pfeTEM_GB_div_efeTEM_GB"].mean()
efiITG_GB = store["/output/efiITG_GB"].mean()
efeITG_GB_div_efiITG_GB = store["/output/efeITG_GB_div_efiITG_GB"].mean()
pfeITG_GB_div_efiITG_GB = store["/output/pfeITG_GB_div_efiITG_GB"].mean()
efeETG_GB = store["/output/efeETG_GB"].mean()

efeTEM_GB_m = store["/output/efeTEM_GB"].median()
efiTEM_GB_div_efeTEM_GB_m = store["/output/efiTEM_GB_div_efeTEM_GB"].median()
pfeTEM_GB_div_efeTEM_GB_m = store["/output/pfeTEM_GB_div_efeTEM_GB"].median()
efiITG_GB_m = store["/output/efiITG_GB"].median()
efeITG_GB_div_efiITG_GB_m = store["/output/efeITG_GB_div_efiITG_GB"].median()
pfeITG_GB_div_efiITG_GB_m = store["/output/pfeITG_GB_div_efiITG_GB"].median()
efeETG_GB_m = store["/output/efeETG_GB"].median()

print("Mean efeTEM_GB: ", efeTEM_GB)
print("Mean efiTEM_GB_div_efeTEM_GB: ", efiTEM_GB_div_efeTEM_GB)
print("Mean pfeTEM_GB_div_efeTEM_GB: ", pfeTEM_GB_div_efeTEM_GB)
print("Mean efiITG_GB: ", efiITG_GB)
print("Mean efeITG_GB_div_efiITG_GB: ", efeITG_GB_div_efiITG_GB)
print("Mean pfeITG_GB_div_efiITG_GB: ", pfeITG_GB_div_efiITG_GB)
print("Mean efeETG_GB: ", efeETG_GB)

print("Median efeTEM_GB: ", efeTEM_GB_m)
print("Median efiTEM_GB_div_efeTEM_GB: ", efiTEM_GB_div_efeTEM_GB_m)
print("Median pfeTEM_GB_div_efeTEM_GB: ", pfeTEM_GB_div_efeTEM_GB_m)
print("Median efiITG_GB: ", efiITG_GB_m)
print("Median efeITG_GB_div_efiITG_GB: ", efeITG_GB_div_efiITG_GB_m)
print("Median pfeITG_GB_div_efiITG_GB: ", pfeITG_GB_div_efiITG_GB_m)
print("Median efeETG_GB: ", efeETG_GB_m)

#efiTEM_GB_div_efeTEM_GB_list.replace([np.inf, -np.inf], np.nan, inplace=True)
#efiTEM_GB_div_efeTEM_GB_list.dropna(inplace=True)
#efiTEM_GB_div_efeTEM_GB_list = efiTEM_GB_div_efeTEM_GB_list.reset_index()


plt.hist(efiTEM_GB_div_efeTEM_GB_list)
plt.show()

