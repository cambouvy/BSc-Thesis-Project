import gc
import os
import re
import logging
import socket
from itertools import product

import numpy as np
import pandas as pd

root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)

from qlknn.misc.tools import dump_package_versions
from qlknn.dataset.data_io import save_to_store, load_from_store
from qlknn.dataset.filtering import (
    create_stored_filter,
    create_divsum,
    split_dims,
    load_stored_filter,
    sanity_filter,
    regime_filter,
    generate_test_train_index,
    split_test_train,
    stability_filter,
    div_filter,
    negative_filter,
    cke_filter,
    cki_filter,
    septot_filter,
    ambipolar_filter,
    femtoflux_filter,
)

# Set up some defaults
filter_functions = {
    "negative": negative_filter,
    "cke": cke_filter,
    "cki": cki_filter,
    "septot": septot_filter,
    "ambipolar": ambipolar_filter,
    "femtoflux": femtoflux_filter,
}

filter_defaults = {
    "div": {
        "efeITG_GB_div_efiITG_GB": (0.05, 1.5),
        "pfeITG_GB_div_efiITG_GB": (0.02, 0.6),
        "efiTEM_GB_div_efeTEM_GB": (0.05, 2.0),
        "pfeTEM_GB_div_efeTEM_GB": (0.03, 0.8),
        "dfeITG_GB_div_efiITG_GB": (0.02, np.inf),
        "dfiITG_GB_div_efiITG_GB": (0.15, np.inf),
        "vceITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "vciITG_GB_div_efiITG_GB": (0.1, np.inf),
        "vteITG_GB_div_efiITG_GB": (0.02, np.inf),
        "vtiITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "dfeTEM_GB_div_efeTEM_GB": (0.10, np.inf),
        "dfiTEM_GB_div_efeTEM_GB": (0.05, np.inf),
        "vceTEM_GB_div_efeTEM_GB": (0.07, np.inf),
        "vciTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
        "vteTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
        "vtiTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
    },
    "negative": None,
    "cke": 50,
    "cki": 50,
    "septot": 1.5,
    "ambipolar": 1.5,
    "femtoflux": 1e-4,
}

div_names_base = [
    "efeITG_GB_div_efiITG_GB",
    "pfeITG_GB_div_efiITG_GB",
    "efiTEM_GB_div_efeTEM_GB",
    "pfeTEM_GB_div_efeTEM_GB",
]

div_names_dv = [
    "dfeITG_GB_div_efiITG_GB",
    "dfiITG_GB_div_efiITG_GB",
    "vceITG_GB_div_efiITG_GB",
    "vciITG_GB_div_efiITG_GB",
    "vteITG_GB_div_efiITG_GB",
    "vtiITG_GB_div_efiITG_GB",
    "dfeTEM_GB_div_efeTEM_GB",
    "dfiTEM_GB_div_efeTEM_GB",
    "vceTEM_GB_div_efeTEM_GB",
    "vciTEM_GB_div_efeTEM_GB",
    "vteTEM_GB_div_efeTEM_GB",
    "vtiTEM_GB_div_efeTEM_GB",
]
div_names = div_names_base + div_names_dv

dump_package_versions(log_func=logger.info)

system_name = socket.gethostname()
if system_name in ["karel-differ"]:
    rootdir = "../../../../qlk_data/gen5"
    logger.info("Detected {!s}, setting rootdir to {!s}".format(system_name, rootdir))
    if not use_dask:
        raise Exception("System {!s} cannot run without dask, aborting".format(system_name))
elif re.match("\D\d{3}\D\d{2}\D\d{2}", system_name) is not None:
    rootdir = "/marconi_scratch/userexternal/kvandepl"
    logger.info("Detected {!s}, setting rootdir to {!s}".format("marconi", rootdir))
elif re.match("davide", system_name) is not None:
    rootdir = "/davide_scratch/userexternal/kvandepl"
    logger.info("Detected {!s}, setting rootdir to {!s}".format("davide", rootdir))
else:
    rootdir = "."
    logger.warning("Unknown system {!s}. Setting rootdir to '{!s}'".format(system_name, rootdir))

dim = 9
gen = 5
filter_num = 10

basename = "gen" + str(gen) + "_" + str(dim) + "D_nions0_flat_filter" + str(filter_num)
store_name = basename + ".h5.1"

input, data, const = load_from_store(store_name)
# Create all filters as on-disk in-dataset cache
# store_filters = False
# if store_filters:
#    with pd.HDFStore(store_name) as store:
#        for filter_name in filter_functions.keys():
#            logger.info('Creating stored filter {!s}'.format(filter_name))
#            create_stored_filter(store, data, filter_name, filter_defaults[filter_name])
logger.info("Creating divsum")
create_divsum(data, divnames=div_names)
logger.info("Splitting dims")
split_dims(input, data, const, gen, filter_num=str(filter_num))

startlen = len(data)

# As the 9D dataset is too big for memory, we have saved the septot filter seperately (Fits on DAVIDE ~250GB RAM)
filters = {}
# with pd.HDFStore(store_name) as store:
#    for filter_name in filter_functions.keys():
#        name = ''.join(['stored_', filter_name, '_filter'])
#        filters[name] = load_stored_filter(store, filter_name, filter_defaults[filter_name])

logger.info("Applying sanity filter")
data = sanity_filter(
    data,
    filter_defaults["septot"],
    filter_defaults["ambipolar"],
    filter_defaults["femtoflux"],
    cke_bound=filter_defaults["cke"],
    cki_bound=filter_defaults["cki"],
    startlen=startlen,
    ambipolar_filter_version=0,
    **filters,
)
logger.info("After filter {!s:<13} {:.2f}% left".format("sanity", 100 * len(data) / startlen))
logger.info("Applying regime filter")
data = regime_filter(data, 0, 100 / 3)
logger.info("After filter {!s:<13} {:.2f}% left".format("regime", 100 * len(data) / startlen))
gc.collect()
logger.info("Slicing input")
input = input.loc[data.index]
sane_store_name = os.path.join(rootdir, "sane_" + basename + ".h5.1")
logger.info("Saving sane dataset to {!s}".format(sane_store_name))
save_to_store(input, data, const, sane_store_name, compress=True)

logger.info("Splitting sane dataset in smaller dimensions")
split_dims(input, data, const, gen, prefix="sane_", filter_num=str(filter_num), compress=True)
# input, data, const = load_from_store(sane_store_name)
del data, input, const
gc.collect()


logger.info("Splitting sets in test-train")
for dim in [4, 7, 9]:
    basename = "gen" + str(gen) + "_" + str(dim) + "D_nions0_flat_filter" + str(filter_num)
    store_name = basename + ".h5.1"
    logger.info("Starting test-train split for dim={!s}, store={!s}".format(dim, store_name))
    input, data, const = load_from_store(store_name)
    generate_test_train_index(input, data, const, frac=0.1)
    split_test_train(input, data, const, basename, rootdir=rootdir, compress=True)
    del input, data, const
    gc.collect()

logger.info("Applying stability and div filters")
for dim, setname in product([4, 7, 9], ["test", "training"]):
    basename = "gen" + str(gen) + "_" + str(dim) + "D_nions0_flat_filter" + str(filter_num)
    store_name = setname + "_" + basename + ".h5.1"
    logger.info(
        "Starting stability and div filters for dim={!s}, set={!s}, store={!s}".format(
            dim, setname, store_name
        )
    )
    input, data, const = load_from_store(store_name)

    data = stability_filter(data)
    # create_divsum(data)
    data = div_filter(data, filter_bounds=filter_defaults["div"])
    save_to_store(input, data, const, "unstable_" + store_name, compress=True)
    del input, data, const
    gc.collect()

logger.info("Filtering dataset done")
