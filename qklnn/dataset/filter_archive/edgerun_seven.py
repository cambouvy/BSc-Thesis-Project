import time
import gc
import os
import logging
import warnings

import numpy as np
from IPython import embed

from qualikiz_tools.qualikiz_io.outputfiles import xarray_to_pandas
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
)
from qlknn.dataset.filter_archive.megarun_one_to_pandas import prep_megarun_ds

try:
    import dask.dataframe as dd

    has_dask = True
except ModuleNotFoundError:
    logger.warning("No dask installed, falling back to xarray")

root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)


def load_ds(rootdir=".", dask=False):
    ds, ds_kwargs = open_with_disk_chunks(os.path.join(rootdir, iden + ".nc.1"), dask=dask)
    return ds, ds_kwargs


if __name__ == "__main__":
    dump_package_versions()
    div_bounds = {
        "efeITG_GB_div_efiITG_GB": (0.05, 2.5),
        "pfeITG_GB_div_efiITG_GB": (0.02, 0.6),
        "efiTEM_GB_div_efeTEM_GB": (0.05, 2.0),
        "pfeTEM_GB_div_efeTEM_GB": (0.03, 0.8),
    }

    dim = 6
    gen = 5
    filter_num = 14

    iden = "pedformreg7"
    rootdir = "/mnt/ssd2/qlk_data/" + iden
    use_cache = True
    basename = "gen" + str(gen) + "_" + str(dim) + "D_" + iden + "_filter" + str(filter_num)
    suffix = ".h5.1"
    store_name = basename + suffix
    store_path = os.path.join(rootdir, store_name)

    # Prepare the dataset. Copy the way we do it for the megarun_ds
    prep_ds_name = "prepared.nc.1"
    starttime = time.time()
    prepared_ds_path = os.path.join(rootdir, prep_ds_name)

    if use_cache:
        if os.path.isfile(prepared_ds_path):
            ds, ds_kwargs = open_with_disk_chunks(prepared_ds_path, dask=False)
        else:
            use_cache = False
            logger.warning(
                "Use cache is True, but {!s} does not exist. Re-creating".format(prepared_ds_path)
            )
    if not use_cache:
        ds, ds_kwargs = prep_megarun_ds(
            prep_ds_name,
            starttime=starttime,
            rootdir=rootdir,
            ds_loader=load_ds,
            save_grow_ds=False,
        )

        # Drop SI variables
        for name, var in ds.variables.items():
            if name.endswith("_SI"):
                ds = ds.drop(name)

        ds = remove_rotation(ds)
        save_prepared_ds(ds, prepared_ds_path, starttime=starttime, ds_kwargs=ds_kwargs)

    logger.info("Preparing dataset done")
    to_drop = []
    features = ("Nustar", "Zeff", "q", "smag", "An", "At")
    for var in ds.data_vars:
        if ds[var].dims != features:
            to_drop.append(var)
    # Dropping non-hypercube variables
    ds = ds.drop(to_drop)
    ds["dimx"] = (
        ds[dummy_var].dims,
        np.arange(0, ds[dummy_var].size).reshape(ds[dummy_var].shape),
    )

    pandas_all = xarray_to_pandas(ds)
    all_vars = pandas_all[features].reset_index()
    outp = all_vars.loc[:, [col for col in all_vars if col not in features]]
    inp = all_vars.loc[:, features]

    save_to_store(inp, outp, pandas_all["constants"], store_path, style="sep")
    del inp, outp, pandas_all
    gc.collect()

    input, data, const = load_from_store(os.path.join(rootdir, store_name), dask=False)
    if not isinstance(data, dd.DataFrame):
        startlen = len(data)
    else:
        startlen = None

    create_divsum(data)

    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", FutureWarning)
        data = sanity_filter(
            data,
            50,
            10,
            1.5,
            1e-4,
            cke_bound=50,
            cki_bound=50,
            ambipolar_filter_version=0,
            startlen=startlen,
        )
        # data = regime_filter(data, 0, 2000)

    # Determine indexes of fluxes to drop because QuaLiKiz breaks down at high gradients.
    data = temperature_gradient_breakdown_filter(input, data, "ITG", patience=6)
    data = temperature_gradient_breakdown_filter(input, data, "TEM", patience=6)
    data = temperature_gradient_breakdown_filter(input, data, "ETG", patience=6)

    print("filter done")
    gc.collect()
    input = input.loc[data.index]
    if startlen is not None:
        print("After filter {!s:<13} {:.2f}% left".format("regime", 100 * len(data) / startlen))
    filter_name = basename
    sane_store_name = os.path.join(rootdir, "sane_" + basename + ".h5.1")
    save_to_store(input, data, const, sane_store_name)
    generate_test_train_index(input, data, const)
    split_test_train(input, data, const, filter_name, rootdir=rootdir)
    del data, input, const
    gc.collect()
    print("Filtering dataset done")

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

        data = stability_filter(data)
        data = div_filter(data, div_bounds)
        save_to_store(input, data, const, os.path.join(rootdir, "unstable_" + basename))
    logger.info("Filtering done")
