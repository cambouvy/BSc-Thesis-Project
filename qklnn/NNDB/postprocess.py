import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from IPython import embed

from qlknn.NNDB.model import (
    Network,
    NetworkJSON,
    PostprocessSlice,
    Postprocess,
    Filter,
    db,
)
from qlknn.models.ffnn import QuaLiKizNDNN
from qlknn.plots.quickslicer import get_similar_not_in_table
from qlknn.dataset.filtering import regime_filter, stability_filter
from qlknn.dataset.data_io import sep_prefix, load_from_store
from qlknn.plots.load_data import load_data, load_nn, prettify_df
from qlknn.misc.tools import parse_dataset_name


def nns_from_nndb(max=20):
    db.connect()
    non_processed = get_similar_not_in_table(
        Postprocess,
        max,
        only_sep=False,
        no_particle=False,
        no_divsum=False,
        no_mixed=False,
        no_gam=False,
    )

    dataset = non_processed[0].pure_network_params.get().dataset
    filter_id = non_processed[0].pure_network_params.get().filter_id
    nns = OrderedDict()
    for dbnn in non_processed:
        nn = dbnn.to_QuaLiKizNN()
        nn.label = "_".join([str(el) for el in [dbnn.__class__.__name__, dbnn.id]])
        nns[nn.label] = nn
    db.close()
    return nns, dataset, filter_id


def process_nns(nns, root_path, set, dataset, filter, leq_bound, less_bound):
    # store = pd.HDFStore('../filtered_gen2_7D_nions0_flat_filter6.h5')
    nn0 = list(nns.values())[0]
    target_names = nn0._target_names
    feature_names = nn0._feature_names

    dim = len(feature_names)
    filter_name = set + "_" + str(dim) + "D_" + dataset + "_filter" + str(filter) + ".h5"
    filter_path_name = os.path.join(root_path, filter_name)
    if not os.path.isfile(filter_path_name):
        filter_path_name += ".1"

    __, regime, __ = load_from_store(
        store_name=filter_path_name, columns=["efe_GB", "efi_GB"], load_input=False
    )
    regime = regime_filter(regime, leq_bound, less_bound).index
    input, target, __ = load_from_store(store_name=filter_path_name, columns=target_names)
    target = target.loc[regime]
    print("target loaded")
    target.columns = pd.MultiIndex.from_product([["target"], target.columns])

    try:
        input["logNustar"] = np.log10(input["Nustar"])
        del input["Nustar"]
    except KeyError:
        print("No Nustar in dataset")
    input = input.loc[target.index]
    input = input[feature_names]
    input.index.name = "dimx"
    input.reset_index(inplace=True)
    dimx = input["dimx"]
    input.drop("dimx", inplace=True, axis="columns")
    target.reset_index(inplace=True, drop=True)

    print("Dataset prepared")
    results = pd.DataFrame()
    for label, nn in nns.items():
        print("Starting on {!s}".format(label))
        out = nn.get_output(input, safe=False)
        out.columns = pd.MultiIndex.from_product([[label], out.columns])
        print("Done! Merging")
        results = pd.concat([results, out], axis="columns")
    diff = results.stack().sub(target.stack().squeeze(), axis=0).unstack()
    rms = diff.pow(2).mean().pow(0.5)

    db.connect()
    for col in rms.index.levels[0]:
        dict_ = {}
        dict_["leq_bound"] = leq_bound
        dict_["less_bound"] = less_bound
        dict_["rms"] = rms[col]
        dict_["filter"] = filter
        cls, id = col.split("_")
        dbnn = Network.get_by_id(int(id))
        dict_ = {"network": dbnn}
        post = Postprocess(**dict_)
        post.save()
    db.close()

    return rms


if __name__ == "__main__":
    # filter_path_name = '../filtered_7D_nions0_flat_filter5.h5'
    root_path = "../../../qlk_data/"
    set = "unstable_test_gen4"
    leq_bound = 0
    less_bound = 10
    nns, dataset, filter_id = nns_from_nndb(100)
    rms = process_nns(nns, root_path, set, dataset, filter_id, leq_bound, less_bound)

# results = pd.DataFrame([], index=pd.MultiIndex.from_product([['target'] + list(nns.keys()), target_names]))
