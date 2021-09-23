import gc

import pandas as pd

from qlknn.dataset.filtering import *

try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError
try:
    from distributed import Client

    use_dask = True
except ModuleNotFoundError:
    print("No dask installed, falling back to xarray")


def filter_rot():
    dim = 8
    gen = 4
    filter_num = 10

    root_dir = "../../../qlk_data"
    iden = "rot_three"
    basename = "".join(["gen", str(gen), "_", str(dim), "D_", iden])
    suffix = ".h5.1"
    store_name = basename + suffix
    input, data, const = load_from_store(os.path.join(root_dir, store_name), dask=False)
    if not isinstance(data, dd.DataFrame):
        startlen = len(data)
    else:
        startlen = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        data = sanity_filter(
            data,
            filter_defaults["ck"],
            filter_defaults["septot"],
            filter_defaults["ambipolar"],
            filter_defaults["femtoflux"],
            startlen=startlen,
        )
        data = regime_filter(data, 0, 100)
    print("filter done")
    gc.collect()
    input = input.loc[data.index]
    if startlen is not None:
        print("After filter {!s:<13} {:.2f}% left".format("regime", 100 * len(data) / startlen))
    filter_name = basename + "_filter" + str(filter_num)
    sane_store_name = os.path.join(root_dir, "sane_" + filter_name + ".h5")
    input, data = create_rotdiv(input, data)
    save_to_store(input, data, const, sane_store_name)
    generate_test_train_index(input, data, const)
    split_test_train(input, data, const, filter_name, root_dir=root_dir)
    del data, input, const
    gc.collect()

    for dim, set in product([8], ["test", "training"]):
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
                ".h5",
            ]
        )
        input, data, const = load_from_store(os.path.join(root_dir, basename))

        data = stability_filter(data)
        # create_divsum(data)
        data = div_filter(data)
        save_to_store(input, data, const, os.path.join(root_dir, "unstable_" + basename))
    # separate_to_store(input, data, '../filtered_' + store_name + '_filter6')
