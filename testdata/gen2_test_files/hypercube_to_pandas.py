import time
from itertools import product
import gc

import xarray as xr
import numpy as np
import pandas as pd
from IPython import embed

from qualikiz_tools.qualikiz_io.outputfiles import xarray_to_pandas


def metadatize(ds):
    scan_dims = [
        dim for dim in ds.dims if dim != "kthetarhos" and dim != "nions" and dim != "numsols"
    ]
    metadata = {}
    for name in ds:
        if (
            all([dim not in scan_dims for dim in ds[name].dims])
            and name != "kthetarhos"
            and name != "nions"
            and name != "numsols"
        ):
            metadata[name] = ds[name].values
            ds = ds.drop(name)
    ds.attrs = metadata
    return ds


def absambi(ds):
    n0, n1 = [
        -(ds["Zi"].sel(nions=1) - ds["Zeffx"])
        / (ds["Zi"].sel(nions=0) * (ds["Zi"].sel(nions=0) - ds["Zi"].sel(nions=1))),
        (ds["Zi"].sel(nions=0) - ds["Zeffx"])
        / (ds["Zi"].sel(nions=1) * (ds["Zi"].sel(nions=0) - ds["Zi"].sel(nions=1))),
    ]
    n0 = xr.DataArray(n0, coords={"Zeffx": n0["Zeffx"], "nions": 0}, name="n0")
    n1 = xr.DataArray(n1, coords={"Zeffx": n1["Zeffx"], "nions": 1}, name="n1")
    ds["n"] = xr.concat([n0, n1], dim="nions")
    ds["absambi"] = np.abs(((ds["pfi_GB"] * ds["n"] * ds["Zi"]).sum("nions") / ds["pfe_GB"]))
    if (ds["absambi"].isnull() & (ds["pfe_GB"] != 0)).sum() == 0:
        ds["absambi"] = ds["absambi"].fillna(1)
    else:
        raise Exception
    ds = ds.drop("n")
    return ds


def prep_totflux(ds):
    ds = absambi(ds)

    ds = metadatize(ds)

    gam = ds["gam_GB"]
    gam = gam.max(dim="numsols")
    gam_great = gam.where(gam.kthetarhos > 2, drop=True)
    ds["gam_great_GB"] = gam_great.max("kthetarhos")
    gam_leq = gam.where(gam.kthetarhos <= 2, drop=True)
    ds["gam_leq_GB"] = gam_leq.max("kthetarhos")

    ome = ds["ome_GB"]
    ome = ome.where(ome.kthetarhos <= 2, drop=True)
    ion_unstable = gam_leq != 0
    ds["TEM"] = (ion_unstable & (ome > 0).any(dim="numsols")).any(dim="kthetarhos")
    ds["ITG"] = (ion_unstable & (ome < 0).any(dim="numsols")).any(dim="kthetarhos")

    ds = ds.drop(["gam_GB", "ome_GB"])
    return ds


def sum_pf(df=None, vt=None, vr=0, vc=None, An=None):
    pf = df * An + vt + vr + vc
    return pf


def prep_sepflux(ds):
    ds = metadatize(ds)
    for spec, mode in product(["e", "i"], ["ITG", "TEM"]):
        fluxes = ["df", "vt", "vc"]
        if spec == "i":
            fluxes.append("vr")
        parts = {flux: flux + spec + mode + "_GB" for flux in fluxes}
        parts = {flux: ds[part] for flux, part in parts.items()}
        pf = sum_pf(**parts, An=ds["An"])
        pf.name = "pf" + spec + mode + "_GB"
        ds[pf.name] = pf
    return ds


starttime = time.time()
# Calculate synthetic and sanity-check variables
ds = xr.open_dataset("Zeffcombo.nc.1")
ds = prep_totflux(ds)
ds_sep = xr.open_dataset("Zeffcombo.sep.nc.1")
ds_sep = prep_sepflux(ds_sep)
ds_tot = ds.merge(ds_sep)
del ds
del ds_sep
gc.collect()
print("Datasets merged after", time.time() - starttime)
for value in ["vfiTEM_GB", "vfiITG_GB", "vriTEM_GB", "vriITG_GB"]:
    try:
        ds_tot = ds_tot.drop(value)
    except ValueError:
        print("{!s} already removed".format(value))
ds_tot.to_netcdf("Zeffcombo.combo.nc", format="NETCDF4", engine="netcdf4")
print("Checkpoint ds_tot saved after", time.time() - starttime)
ds_tot = ds_tot.sel(nions=0)
ds_tot.to_netcdf("Zeffcombo.combo.nions0.nc", format="NETCDF4", engine="netcdf4")
print("Checkpoint ds_tot nions0 saved after", time.time() - starttime)
# ds_tot = xr.open_dataset('Zeffcombo.combo.nions0.nc')

# Convert to pandas
print("Starting pandaization after", time.time() - starttime)
dfs = xarray_to_pandas(ds_tot)
print("Xarray pandaized after", time.time() - starttime)
del ds_tot
gc.collect()
store = pd.HDFStore("./gen3_9D_nions0_flat.h5")
dfs[("Zeffx", "Ati", "Ate", "An", "qx", "smag", "x", "Ti_Te", "Nustar")].reset_index(inplace=True)
dfs[("Zeffx", "Ati", "Ate", "An", "qx", "smag", "x", "Ti_Te", "Nustar")].index.name = "dimx"
print("Index reset after", time.time() - starttime)
store["/megarun1/input"] = dfs[
    ("Zeffx", "Ati", "Ate", "An", "qx", "smag", "x", "Ti_Te", "Nustar")
].iloc[:, :9]
print("Input stored after", time.time() - starttime)
store["/megarun1/flattened"] = dfs[
    ("Zeffx", "Ati", "Ate", "An", "qx", "smag", "x", "Ti_Te", "Nustar")
].iloc[:, 9:]
store["/megarun1/constants"] = dfs["constants"]
print("Done after", time.time() - starttime)
