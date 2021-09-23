import time
import gc
import os
import logging
import warnings
import pandas as pd
import xarray as xr


import numpy as np

from qualikiz_tools.qualikiz_io.outputfiles import xarray_to_pandas, qualikiz_folder_to_xarray
from qlknn.misc.tools import dump_package_versions
from qlknn.dataset.data_io import save_to_store, load_from_store
from qlknn.dataset.filtering import (
    create_divsum,
    sanity_filter,
    generate_test_train_index,
    split_test_train,
    stability_filter,
    div_filter,
    temperature_gradient_breakdown_filter,
)
from qlknn.dataset.hypercube_to_pandas import (
    open_with_disk_chunks,
    save_prepared_ds,
    remove_rotation,
    dummy_var,
    absambi,
    determine_stability,
    determine_At
)
from qlknn.dataset.filter_archive.megarun_one_to_pandas import prep_megarun_ds

root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)

try:
    import dask.dataframe as dd

    has_dask = True
except ModuleNotFoundError:
    logger.warning("No dask installed, falling back to xarray")


def load_ds(rootdir=".", dask=False):
    ds, ds_kwargs = open_with_disk_chunks(
        os.path.join(rootdir, iden + ".nc.1"), dask=dask
    )
    return ds, ds_kwargs


""" Filtering and preprocessing of the (glued and folded) dataset.
    6 files are saved after this run: the general file containing the dataset converted to pandas, the "sane" dataset 
    which is filtered out, the training and test set and the unstable training and test set """


if __name__ == "__main__":
    dump_package_versions()

    dim = 7
    gen = 5
    filter_num = 11

    iden = "pedformreg8"
    rootdir = "/m100_work/FUAC5_GKNN/camille/results/"
    use_cache = True
    basename = (
        "gen" + str(gen) + "_" + str(dim) + "D_" + iden + "_filter" + str(filter_num)
    )
    suffix = ".h5.1"
    store_name = basename + suffix
    store_path = os.path.join(rootdir, store_name)

    prep_ds_name = "folded_netcdf7.nc"
    starttime = time.time()
    prepared_ds_path = os.path.join(rootdir, prep_ds_name)

    if use_cache:
        print("Trying to use cache")
        if os.path.isfile(prepared_ds_path):
            ds, ds_kwargs = open_with_disk_chunks(prepared_ds_path, dask=False)
        else:
            print(f"Trying to use cache, but '{prepared_ds_path}' does not exist. Not using cache!")
            use_cache = False

    if not use_cache:
        ds, ds_kwargs = prep_megarun_ds(
            prep_ds_name,
            starttime=starttime,
            rootdir=rootdir,
            ds_loader=load_ds,
            save_grow_ds=False,
        )
        # Dims are now (Ate, Nustar, q, smag, Ati, An, dilution)

        # Remove rotation as not considered in L-Mode
        ds = remove_rotation(ds)

        print("Saving prepared dataset to", prepared_ds_path)
        save_prepared_ds(ds, prepared_ds_path, starttime=starttime, ds_kwargs=ds_kwargs)

        # Create new "dimx" variable to keep track of what's what
        # This dimx will be used for all HDF5 files and further derived files
        ds["dimx"] = (
            ds[dummy_var].dims,
            np.arange(0, ds[dummy_var].size).reshape(ds[dummy_var].shape),
        )


    print("Preparing dataset done")


    # Dimensions considered in the dataset
    features = ('Ate', 'Nustar', 'q', 'smag', 'Ati', 'An', 'dilution')
    features2 = ('Ate', 'Nustar', 'q', 'smag', 'Ati', 'An','dilution', 'nions')



    # Determine instability mode of each point
    # ds = determine_stability(ds)

    # Calculate ambipolarity
    ds = absambi(ds)

    # Dropping non-hypercube variables
    # to_drop = []
    # for var in ds.data_vars:
    #     if ds[var].dims != features and ds[var].dims != features2:
    #         to_drop.append(var)
    # ds = ds.drop(to_drop)


    pandas_all = xarray_to_pandas(ds)
    all_vars1 = pandas_all[features].reset_index()
    all_vars2 = pandas_all[features2].reset_index()
    all_vars = pd.concat([all_vars1, all_vars2], axis=1).drop_duplicates().reset_index(drop=True)
    all_vars = all_vars.loc[:, ~all_vars.columns.duplicated()]

    outp = all_vars.loc[:, [col for col in all_vars if col not in features]]
    inp = all_vars.loc[:, features]

    save_to_store(inp, outp, pandas_all["constants"], store_path, style="sep")
    del inp, outp, pandas_all
    gc.collect()

    input, data, const = load_from_store(os.path.join(rootdir, store_name), dask=False)

    # Create the targets for 'divsum' networks (i.e. the 'physics-informed' ones)
    print("Creating divsum")
    create_divsum(data)

    data.dropna(inplace=True)

    with warnings.catch_warnings():
        data = sanity_filter(
            data,
            septot_factor=1.5,
            ambi_bound=1.5,
            femto_bound=1e-4,  # (everything under this value will be clipped to 0)
            startlen=len(data)

        )

    # Reset the indices after some rows were removed by the filtering
    data.reset_index(inplace=True)

    # Attempt to run the temperature gradient breadkdown filter, not carried out in this work due to a bug in the method

    # Add At to the input for the temperature gradient breakdown filter
    # tgb_filter_input = determine_At(data, input)
    #
    # #Split the data into smaller datasets to avoid memory error when filtering with temperature gradient breakdown
    # data_split = [data.iloc[:1000000, :], data.iloc[1000001:2000000, :], data.iloc[2000001:3000000, :], data.iloc[3000001:4000000, :],
    #               data.iloc[4000001:5000000, :], data.iloc[5000001:6000000, :], data.iloc[6000001:7000000, :], data.iloc[7000001:8000000, :],
    #               data.iloc[8000001:9000000, :], data.iloc[10000001:11000000, :], data.iloc[11000001:12000000, :], data.iloc[12000001:13000000, :],
    #               data.iloc[13000001:14000000, :], data.iloc[14000001:15000000, :], data.iloc[15000001:16000000, :], data.iloc[16000001:17000000, :],
    #               data.iloc[17000001:18000000, :], data.iloc[18000001:, :]]
    #
    #
    #
    # tgb_filter_input_split = [tgb_filter_input.iloc[:1000000, :],tgb_filter_input.iloc[1000001:2000000, :], tgb_filter_input.iloc[2000001:3000000, :],
    #                           tgb_filter_input.iloc[3000001:4000000, :],
    #                           tgb_filter_input.iloc[4000001:5000000, :], tgb_filter_input.iloc[5000001:6000000, :], tgb_filter_input.iloc[6000001:7000000, :],
    #                           tgb_filter_input.iloc[7000001:8000000, :],
    #                           tgb_filter_input.iloc[8000001:9000000, :], tgb_filter_input.iloc[10000001:11000000, :], tgb_filter_input.iloc[11000001:12000000, :],
    #                           tgb_filter_input.iloc[12000001:13000000, :],
    #                           tgb_filter_input.iloc[13000001:14000000, :], tgb_filter_input.iloc[14000001:15000000, :], tgb_filter_input.iloc[15000001:16000000, :],
    #                           tgb_filter_input.iloc[16000001:17000000, :],
    #                           tgb_filter_input.iloc[17000001:18000000, :], tgb_filter_input.iloc[18000001:, :]]
    #
    # for i in range(len(data_split)):
    #     # Determine indexes of fluxes to drop because QuaLiKiz breaks down at high gradients.
    #     data_split[i] = temperature_gradient_breakdown_filter(tgb_filter_input_split[i], data_split[i], "ITG", patience=6)
    #     data_split[i] = temperature_gradient_breakdown_filter(tgb_filter_input_split[i], data_split[i], "TEM", patience=6)
    #     data_split[i] = temperature_gradient_breakdown_filter(tgb_filter_input_split[i], data_split[i], "ETG", patience=6)
    #
    # data = pd.concat(data_split)

    # # Calculate At and add it to an input used for tgb_filter
    # tgb_filter_input = input
    # tgb_filter_input = determine_At(data, tgb_filter_input)
    #
    # # Determine indexes of fluxes to drop because QuaLiKiz breaks down at high gradients.
    # data = temperature_gradient_breakdown_filter(tgb_filter_input, data, "ITG", patience=6)
    # data = temperature_gradient_breakdown_filter(tgb_filter_input, data, "TEM", patience=6)
    # data = temperature_gradient_breakdown_filter(tgb_filter_input, data, "ETG", patience=6)



    print("filter done")
    gc.collect()
    input = input.loc[data.index]
    filter_name = basename
    sane_store_name = os.path.join(rootdir, "sane_" + basename + ".h5.1")
    save_to_store(input, data, const, sane_store_name)
    print("Filtering dataset done")

    stable_pts = 0
    for index, row in data.iterrows():
        if not (row["ETG"]) and not (row["ITG"]) and not (row["TEM"]):
            stable_pts += 1
    print("Stable points: ", 100 * stable_pts / len(data), "%")


    print("Splitting dataset in test-train")
    generate_test_train_index(input, data, const)
    split_test_train(input, data, const, filter_name, rootdir=rootdir)
    del data, input, const
    gc.collect()

    for set in ["test", "training"]:
        print(dim, set)
        basename = "".join(
            [
                set,
                "_gen",
                str(gen),
                "_",
                str(dim),
                "D_",
                iden,
                "_filter",
                str(filter_num),
                ".h5.1",
            ]
        )
        input, data, const = load_from_store(os.path.join(rootdir, basename))

        stable_pts = 0
        itg_pts = 0
        tem_pts = 0
        etg_pts = 0
        etg_itg_pts = 0
        etg_tem_pts = 0
        itg_tem_pts = 0
        itg_tem_etg_pts = 0
        for index, row in data.iterrows():
            if not (row["ETG"]) and not (row["ITG"]) and not (row["TEM"]):
                stable_pts += 1
            elif not (row["ETG"]) and (row["ITG"]) and not(row["TEM"]):
                itg_pts += 1
            elif not (row["ETG"]) and not (row["ITG"]) and (row["TEM"]):
                tem_pts += 1
            elif (row["ETG"]) and not (row["ITG"]) and not(row["TEM"]):
                etg_pts += 1
            elif (row["ETG"]) and (row["ITG"]) and not(row["TEM"]):
                etg_itg_pts += 1
            elif (row["ETG"]) and not (row["ITG"]) and (row["TEM"]):
                etg_tem_pts += 1
            elif not (row["ETG"]) and (row["ITG"]) and (row["TEM"]):
                itg_tem_pts += 1
            else:
                itg_tem_etg_pts += 1

        print("Total rows: ", len(data))
        print("Stable points: ", 100 * stable_pts / len(data), "%")
        print("ITG unstable points: ", 100 * itg_pts / len(data), "%")
        print("ETG unstable points: ", 100 * etg_pts / len(data), "%")
        print("TEM unstable points: ", 100 * tem_pts / len(data), "%")
        print("ITG-ETG unstable points: ", 100 * etg_itg_pts / len(data), "%")
        print("ETG-TEM unstable points: ", 100 * etg_tem_pts / len(data), "%")
        print("ITG-TEM unstable points: ", 100 * itg_tem_pts / len(data), "%")
        print("Completely unstable points: ", 100 * itg_tem_etg_pts / len(data), "%")

        data = stability_filter(data)
        save_to_store(input, data, const, os.path.join(rootdir, "unstable_" + basename))

    print("Filtering done")
