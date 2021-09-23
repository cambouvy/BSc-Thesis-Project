import xarray as xr
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

"Plots a QuaLiKiz data slice "

ds = xr.open_dataset("/marconi_work/FUA35_GKNN/camille/edgerun8_netcdf/smag_1.00_dilution_0.20_Nustar_0.33_q_8.00.nc")
pointsToKeep = xr.DataArray(np.linspace(0, 4046, num=14), dims="points")
ds_slice = ds.sel(dimx=pointsToKeep, method="nearest")

interesting_variable_slice = ds.where((ds.Ate == 15) & (ds.Ane == 8), drop=True)

embed()

# Plotting section to add in the interactive shell 
plt.plot(interesting_variable_slice["Ati"], interesting_variable_slice["efiITG_GB"], 'o')
plt.title('QuaLiKiz slice')
plt.xlabel('Ati')
plt.ylabel('Ion Heat flux (GB)')
plt.show()



