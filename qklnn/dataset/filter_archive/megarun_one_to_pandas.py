"""
This script converts the original 'megarun_one' QuaLiKiz run to pandas.
In principle only the last 'generation' of the dataset can be generated,
but for just the pandaization all generations are probably the same. 
Hoever, the starting point is different

gen1-4s uses the original dataset, md5 hashes:
d703143e0771e4c3ff1623807962cf87  Zeffcombo.nc.1
3d2db88c5a92dc6250099398352b7794  Zeffcombo.sep.nc.1

gen5 uses the renormalized (GB normalization), md5 hashes:
ad40cf2f71887984d34bb0f588325d59  Zeffcombo.nc.1
155467ca36cbb5257473e8b436113b21  Zeffcombo.sep.nc.1
"""
import os
import time
import socket
import logging
import re

import xarray as xr
import numpy as np

from qlknn.misc.tools import dump_package_versions
from qlknn.dataset.hypercube_to_pandas import (
    metadatize,
    absambi,
    calculate_normni,
    calculate_rotdivs,
    determine_stability,
    sum_pf,
    sum_pinch,
    calculate_particle_sepfluxes,
    remove_rotation,
    open_with_disk_chunks,
    calculate_grow_vars,
    merge_gam_leq_great,
    compute_and_save_var,
    compute_and_save,
    save_prepared_ds,
    create_input_cache,
    input_hdf5_from_cache,
    data_hdf5_from_ds,
    dummy_var,
    save_attrs,
)

try:
    profile
except NameError:
    from qlknn.misc.tools import profile
root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)


@profile
def load_megarun1_ds(rootdir=".", dask=False):
    """Load the 'megarun1' data as xarray/dask dataset
    For the megarun1 dataset, the data is split in the 'total fluxes + growth rates'
    and 'TEM/ITG/ETG fluxes'. Open the former with `open_with_disk_chunks` and the
    latter with the same kwargs and merge them together.

    These two files were generated from the original

    Kwargs:
        starttime: Time the script was started. All debug timetraces will be
                   relative to this point. [Default: current time]

    Returns:
        ds:        Merged chunked xarray.Dataset ready for preparation.
    """
    ds, ds_kwargs = open_with_disk_chunks(os.path.join(rootdir, "Zeffcombo.nc.1"), dask=dask)
    ds_sep = xr.open_dataset(os.path.join(rootdir, "Zeffcombo.sep.nc.1"), **ds_kwargs)
    ds_tot = ds.merge(ds_sep.data_vars)
    return ds_tot, ds_kwargs


@profile
def prep_megarun_ds(
    prepared_ds_name,
    starttime=None,
    rootdir=".",
    use_gam_cache=False,
    ds_loader=load_megarun1_ds,
    dask=False,
    save_grow_ds=True,
):
    """Prepares a QuaLiKiz netCDF4 dataset for convertion to pandas
    This function was designed to use dask, but should work for
    pure xarray too. In this function it is assumed the chunks on disk,
    and the chunks for dask are the same (or at least aligned)

    Kwargs:
        starttime:      Time the script was started. All debug timetraces will be
                        relative to this point. [Default: current time]
        rootdir:        Path where all un-prepared datasets reside [Default: '.']
        use_disk_cache: Just load an already prepared dataset [Default: False]
        use_gam_cache:  Load the already prepared gam_leq/gam_great cache [Default: False]
        ds_loader:      Function that loads all non-prepared datasets and merges
                        them to one. See default for example [Default: load_megarun1_ds]

    Returns:
        ds:             Prepared xarray.Dataset (chunked if ds_loader returned chunked)
    """
    if starttime is None:
        starttime = time.time()

    prepared_ds_path = os.path.join(rootdir, prepared_ds_name)
    # Load the dataset
    ds, ds_kwargs = ds_loader(rootdir, dask=dask)
    logger.info("Datasets merging done")
    # Calculate gam_leq and gam_great and cache to disk
    ds = merge_gam_leq_great(
        ds,
        ds_kwargs=ds_kwargs,
        rootdir=rootdir,
        use_disk_cache=use_gam_cache,
        starttime=starttime,
    )
    logger.info("gam_[leq,great]_GB cache creation done")

    ds = determine_stability(ds)
    logger.info("[ITG|TEM|ETG] calculation done")

    if "normni" not in ds:
        ds = calculate_normni(ds)
        logger.info("normni calculation done")
    ds = absambi(ds)
    logger.info("absambi calculation done")

    ds = calculate_particle_sepfluxes(ds)
    logger.info("pf[i|e][ITG|TEM] calculation done")

    ds = sum_pinch(ds)
    logger.info("Total pinch calculation done")

    # Optionally save gam_GB and ome_GB in a separate dataset
    if save_grow_ds:
        prep_ds_name_parts = prepared_ds_name.split(".")
        grow_ds_name = ".".join([prep_ds_name_parts[0] + "_grow"] + prep_ds_name_parts[1:])
        grow_ds_path = os.path.join(rootdir, grow_ds_name)
        grow_ds = ds[["gam_GB", "ome_GB"]]
        grow_ds = metadatize(grow_ds)
        save_prepared_ds(grow_ds, grow_ds_path, starttime=starttime, ds_kwargs=None)
        logger.info("Saving grow_ds done")
        del grow_ds

    # Remove variables and coordinates we do not need for NN training
    ds = ds.drop(["gam_GB", "ome_GB"])
    # normni does not need to be saved for the 9D case.
    # TODO: Check for Aarons case!
    # if 'normni' in ds.data_vars:
    #    ds.attrs['normni'] = ds['normni']
    #    ds = ds.drop('normni')

    # Save all non-dimension coordinates as metadata
    ds = metadatize(ds)

    # Remove all but first ion
    # TODO: Check for Aarons case!
    ds = ds.sel(nions=0)
    ds.attrs["nions"] = ds["nions"].values
    ds = ds.drop("nions")
    logger.info("Bookkeeping done")

    logger.info("Pre-disk write dataset preparation done")
    return ds, ds_kwargs


def prepare_megarun1(rootdir, use_disk_cache=False, use_gam_cache=False, dask=False):
    starttime = time.time()
    store_name = os.path.join(rootdir, "gen5_9D_nions0_flat_filter10.h5.1")
    prep_ds_name = "Zeffcombo_prepared.nc.1"
    prep_ds_path = os.path.join(rootdir, prep_ds_name)
    if use_disk_cache and not os.path.isfile(prep_ds_path):
        logger.warning(
            "Use of disk cache requested, but {!s} does not exist. Creating cache!".format(
                prep_ds_path
            )
        )
        use_disk_cache = False
    if use_disk_cache:
        ds, ds_kwargs = open_with_disk_chunks(prep_ds_path, dask=dask)
    else:
        ds_loader = load_megarun1_ds
        ds, ds_kwargs = prep_megarun_ds(
            prep_ds_name,
            starttime=starttime,
            rootdir=rootdir,
            ds_loader=ds_loader,
            use_gam_cache=use_gam_cache,
            dask=dask,
        )
        ds = remove_rotation(ds)
        save_prepared_ds(ds, prep_ds_path, starttime=starttime, ds_kwargs=None)
    logger.info("Preparing dataset done")
    return ds, store_name


if __name__ == "__main__":
    # client = Client()
    # This script keeps three on-disk 'caches' to start somewhere inbetween. By default, do not use them
    use_prep_disk_cache = True
    use_input_cache = True
    use_gam_cache = True
    use_dask = False
    dump_package_versions(log_func=logger.info)
    if use_dask:
        import dask

        dask_scheduler = "threads"  # Fastest on Marconi by a negligible amount
        # dask_scheduler = 'distributed-threads'
        # dask_scheduler = 'distributed-processes'
        # dask_scheduler = 'single-threaded'
        # dask_scheduler = 'synchronous' #Debugging scheduler
        if dask_scheduler == "distributed-threads":
            from dask.distributed import Client

            client = Client(processes=False)
        elif dask_scheduler == "distributed-processes":
            from dask.distributed import Client

            client = Client(processes=True)
        else:
            dask.config.set(scheduler=dask_scheduler)
        logger.info("Using dask scheduler '{!s}'".format(dask_scheduler))
    else:
        logger.info("Using xarray")

    system_name = socket.gethostname()
    if system_name in ["karel-differ"]:
        rootdir = "../../../../qlk_data/gen5"
        logger.info("Detected {!s}, setting rootdir to {!s}".format(system_name, rootdir))
        if not use_dask:
            raise Exception("System {!s} cannot run without dask, aborting".format(system_name))
    elif re.match("\D\d{3}\D\d{2}\D\d{2}", system_name) is not None:
        rootdir = "/marconi_scratch/userexternal/kvandepl"
        logger.info("Detected {!s}, setting rootdir to {!s}".format("marconi", rootdir))
    else:
        root_dir = "."
        logger.warning(
            "Unknown system {!s}. Setting rootdir to '{!s}'".format(system_name, root_dir)
        )

    ds, store_name = prepare_megarun1(
        rootdir,
        use_disk_cache=use_prep_disk_cache,
        use_gam_cache=use_gam_cache,
        dask=use_dask,
    )

    # Convert to pandas
    # Remove all variables with more dims than our cube
    logger.info("Dropping all variables not in hypercube")
    non_drop_dims = list(ds[dummy_var].dims)
    for name, var in ds.variables.items():
        if len(set(var.dims) - set(non_drop_dims)) != 0:
            ds = ds.drop(name)

    # dummy_var = next(ds.data_vars.keys().__iter__())
    logger.info("Setting monotionicly increasing index dimx")
    ds["dimx"] = (
        ds[dummy_var].dims,
        np.arange(0, ds[dummy_var].size).reshape(ds[dummy_var].shape),
    )
    cachedir = os.path.join(rootdir, "cache")
    logger.info("Creating input cache")
    create_input_cache(ds, cachedir, use_dask=use_dask)

    input_hdf5_from_cache(
        store_name, cachedir, columns=non_drop_dims, mode="w", use_dask=use_dask
    )
    save_attrs(ds.attrs, store_name)

    data_hdf5_from_ds(ds, store_name)
    logger.info("Conversion to pandas done")
