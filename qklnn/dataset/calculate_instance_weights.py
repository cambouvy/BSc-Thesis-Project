import os
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
from qlknn.models.kerasmodel import NDHornNet
from qlknn.misc.analyse_names import determine_special_input
from qlknn.dataset.mapping import get_ID
from qlknn.dataset.mapping import all_inputs
from IPython import embed


def combine_inp_thresh(sep_store, comb_store, turb):
    """Combine quicklsicer inputs with leading flux thresholds and write to HDF

    Args:
      - sep_store: String where to find seperate input and output of a quickslicer run
      - comb_store: String where to store combined inputs and thresholds
      - turb: String specifying the turbulence type ('ITG, 'TEM', 'ETG')
    """
    store = pd.HDFStore(sep_store, "r")
    inp = store["/input"]
    if turb == "ITG":
        lead_fl = "efi"
    else:
        lead_fl = "efe"
    thresh = store["/stats"]["QLK_" + lead_fl + turb + "_GB"]["thresh"]
    store.close()
    inp = inp.loc[thresh.index, :]
    combined = pd.concat([inp, thresh], axis=1)
    kwargs = {}
    kwargs["complib"] = "zlib"
    kwargs["complevel"] = 1
    combined.to_hdf(comb_store, "/" + turb, format="fixed", **kwargs)


def sort_thresholds(unsorted_thresh, special_inp, inp_index):
    """Sort thresholds to global IDs and reindex with given input indices

    Args:
      - unsorted_thresh: DataFrame of inputs with thresholds
      - special_inp: String special_input not in the inputs of unsorted_thresh
      - inp_index: all global IDs of datapoints for which a threshold is needed
    """
    unsorted_thresh = unsorted_thresh.loc[unsorted_thresh["thresh"].notnull(), :]
    len_inputs = len(unsorted_thresh)
    unsorted_thresh = pd.DataFrame(
        np.repeat(unsorted_thresh.values, len(all_inputs[special_inp]), axis=0),
        columns=unsorted_thresh.columns,
    )
    unsorted_thresh[special_inp] = np.tile(all_inputs[special_inp], len_inputs)
    unsorted_thresh.index = get_ID(unsorted_thresh)
    unsorted_thresh = unsorted_thresh["thresh"]
    return unsorted_thresh.reindex(inp_index)


def instace_weights_from_thresholds(inp, special_dim, thresholds, width, base_top_ratio):
    """Calculate instance weights for given inputs from thresholds

    weight = (base_top_ratio-1) * e^(-(inp-threshold)^2 / (2*width^2)) + 1

    Args:
      - inp: DataFrame with all the inputs for which a instance weight is desired
      - special_dim: input dimension in which the threshold is given
      - thresholds: DataFrame of thresholds for all inputs
      - width: width of the gaussian for the weight calculation
      - base_top_ratio: ratio of the lowest weights (normally 1.0) to the highest
    """
    thresholds = thresholds.flatten()
    thresholds = np.clip(thresholds, -2, 16)
    gaussian = norm(thresholds, width)
    weights = gaussian.pdf(inp[special_dim].values)
    weights = pd.Series(weights, index=inp.index, dtype="float32")
    weights = weights * math.sqrt(2 * math.pi) * width * (base_top_ratio - 1) + 1
    weights[weights.isnull()] = 1.0
    return weights


if __name__ == "__main__":
    width = 7
    base_top_ratio = 10
    data_store = "../../../../Data/training_gen5_7D_nions0_flat_filter10.h5.1"
    weights_store = "../../../../Data/test.h5.1"
    thresholds_from_NNs = False
    nn_root = "../../../../NeuralNets/7DNets"
    QLK_thresholds_store = "../../../../Data/all_thresholds.h5.1"
    kwargs = {}
    kwargs["complib"] = "zlib"
    kwargs["complevel"] = 1

    store = pd.HDFStore(data_store, "r")
    inp = store["/input"]
    store.close()

    print("Collect thresholds", flush=True)
    if thresholds_from_NNs:
        print("Calculate thresholds from HornNets", flush=True)
        HornNets = {
            "ETG": NDHornNet(os.path.join(nn_root, "ETG", "nn.json")),
            "ITG": NDHornNet(os.path.join(nn_root, "ITG", "nn.json")),
            "TEM": NDHornNet(os.path.join(nn_root, "TEM", "nn.json")),
        }
        batch_size = math.ceil(len(inp) / 1000)
        all_thresh = {
            turb: HornNets[turb].get_threshold(inp, batch_size=batch_size)
            for turb in HornNets.keys()
        }
    else:
        print("Load thresholds from store", flush=True)
        store = pd.HDFStore(QLK_thresholds_store, "r")
        all_thresh = {}
        all_thresh_unsorted = {}
        for turb in ["ITG", "TEM", "ETG"]:
            if "/" + turb + "_sorted" in store.keys():
                all_thresh[turb] = store["/" + turb + "_sorted"]
            elif "/" + turb in store.keys():
                all_thresh_unsorted[turb] = store["/" + turb]
        store.close()

        for turb in all_thresh_unsorted.keys():
            print(
                "Sort "
                + turb
                + " thresholds from quickslicer ID to unique global ID and fill in with NaN",
                flush=True,
            )
            all_thresh[turb] = sort_thresholds(
                all_thresh_unsorted[turb], determine_special_input(turb)[0], inp.index
            )
            all_thresh[turb].to_hdf(
                QLK_thresholds_store, "/" + turb + "_sorted", format="fixed", **kwargs
            )

        for turb in all_thresh.keys():
            all_thresh[turb] = all_thresh[turb].values

    for turb in all_thresh.keys():
        print("Calculate weights for " + turb + " type tubulence", flush=True)
        spec_dim = determine_special_input(turb)[0]
        weights = instace_weights_from_thresholds(
            inp, spec_dim, all_thresh[turb], width, base_top_ratio
        )
        weights.to_hdf(weights_store, "/output/" + turb + "weights", format="fixed", **kwargs)
