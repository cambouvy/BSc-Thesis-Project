""" Info"""
import os
import copy
import tarfile
from pathlib import Path
from collections import OrderedDict

import requests
import pandas as pd
import numpy as np

from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN
from qlknn.models.clipping import LeadingFluxNN
from qlknn.models.victor_rule import VictorNN
from qlknn.misc.analyse_names import (
    is_pure,
    is_transport,
    split_parts,
)


def get_weights_and_biases():
    """ Pull weights and biases from the gitlab.com repository """
    resulting_folder = Path("qlknn-hyper-master").resolve()
    if not resulting_folder.exists():
        tarball_name = "qlknn-hyper-master.tar.bz2"
        url = f"https://gitlab.com/qualikiz-group/qlknn-hyper/-/archive/master/{tarball_name}"
        r = requests.get(url, allow_redirects=True)
        open(tarball_name, "wb").write(r.content)
        tar = tarfile.open(tarball_name, "r:bz2")
        tar.extractall()
        tar.close()
        os.remove(tarball_name)

    if not resulting_folder.exists():
        raise RuntimeError("Failed to download weights and biases")

    return resulting_folder


def stacker(*args):
    """ Stack 1D networks together as a 2D array """
    return np.hstack(args)


def parse_jsons(folder):
    """ Parse the JSONs in the given folder """
    networks = OrderedDict()
    jsons = folder.glob("*.json")
    # Walk over all JSONs and try to load it into a QuaLiKizNDNN. Only 1D out here!
    for path in jsons:
        nn = QuaLiKizNDNN.from_json(path)
        if len(nn._target_names) > 1:
            raise RuntimeError("Multi-target NN! Not sure what to do..")
        networks[nn._target_names[0]] = nn
    networks = {name: net for name, net in networks.items()}
    if len(networks) == 0:
        raise RuntimeError(
            f"Did not find any networks in '{nn_folder}'! Does it contain the JSONs?"
        )
    return networks


def combine_divnet_with_leading(networks):
    """ Multiply nets of the form flux_div_leading with leading """
    networks = copy.deepcopy(networks)
    for target_name in list(networks.keys()):
        if is_transport(target_name) and not is_pure(target_name):
            target, op, leading = split_parts(target_name)
            if op != "_div_":
                raise RuntimeError("Did not expect other operations than _div_")
            nn_norot = QuaLiKizComboNN(
                pd.Series(target),
                [networks.pop(target_name), networks[leading]],
                lambda x, y: x * y,
            )
            networks[target] = nn_norot
    return networks


def create_layered_nets(networks):
    networks = copy.deepcopy(networks)
    """ Create networks in layers ~as they are implemented in JETTO/RAPTOR """

    if "gam_leq_GB" in networks:
        gam_name = "gam_leq_GB"
    else:
        raise RuntimeError("No gam network found! Needed for VictorRule")

    gam = networks.pop(gam_name)
    flux_names = [
        "efeETG_GB",  # 1
        "efeITG_GB",  # 2
        "efeTEM_GB",  # 3
        "efiITG_GB",  # 4
        "efiTEM_GB",  # 5
        "pfeITG_GB",  # 6
        "pfeTEM_GB",  # 7
        "dfeITG_GB",  # 8
        "dfeTEM_GB",  # 9
        "vteITG_GB",  # 10
        "vteTEM_GB",  # 11
        "vceITG_GB",  # 12
        "vceTEM_GB",  # 13
        "dfiITG_GB",  # 14
        "dfiTEM_GB",  # 15
        "vtiITG_GB",  # 16
        "vtiTEM_GB",  # 17
        "vciITG_GB",  # 18
        "vciTEM_GB",  # 19
        "gam_leq_GB",
    ]  # As defined in QLKNN-fortran
    nets = [networks[name] for name in flux_names[:-1]]

    combo_nn = QuaLiKizComboNN(pd.Series(flux_names[:-1]), nets, stacker)
    leading_nn = LeadingFluxNN.add_leading_flux_clipping(combo_nn)
    vic_nn = VictorNN(leading_nn, gam)

    return combo_nn, leading_nn, vic_nn


low_bound = None
high_bound = None

nn_folder = get_weights_and_biases()
networks = parse_jsons(nn_folder)
networks = combine_divnet_with_leading(networks)
combo_nn, leading_nn, vic_nn = create_layered_nets(networks)

leading_nn.label = "QLKNN-10D-hyper"
# Define for external imports
nns = {
    leading_nn.label: leading_nn,
}
slicedim = "Ati"
style = "mono"

if __name__ == "__main__":
    # Create a pd.DataFrame containing a superset of QLKNN inputs
    scann = 24
    input = pd.DataFrame()
    input["Ati"] = np.array(np.linspace(2, 13, scann))
    input["Ti_Te"] = np.full_like(input["Ati"], 1.0)
    input["Te"] = np.full_like(input["Ati"], 1.0)
    input["Zeff"] = np.full_like(input["Ati"], 1.0)
    input["An"] = np.full_like(input["Ati"], 2.0)
    input["Ate"] = np.full_like(input["Ati"], 5.0)
    input["q"] = np.full_like(input["Ati"], 0.660156)
    input["smag"] = np.full_like(input["Ati"], 0.399902)
    input["logNustar"] = np.full_like(input["Ati"], np.log10(0.009995))
    input["x"] = np.full_like(input["Ati"], 0.449951)
    input["gammaE_QLK"] = np.full_like(input["Ati"], 0.4)

    # Do not clip fluxes on the Python side
    # low_bound = np.array([[0 if ('ef' in name) and (not 'div' in name) else -np.inf for name in nn._target_names]]).T
    # low_bound = pd.DataFrame(index=nn._target_names, data=low_bound)

    print("Combo NN")
    print(
        combo_nn.get_output(
            input,
            safe=True,
            clip_high=False,
            clip_low=False,
            high_bound=high_bound,
            low_bound=low_bound,
        )
    )

    print("Leading NN")
    print(
        leading_nn.get_output(
            input,
            safe=True,
            clip_high=False,
            clip_low=False,
            high_bound=high_bound,
            low_bound=low_bound,
        )
    )

    print("Vic NN")
    print(
        vic_nn.get_output(
            input,
            safe=True,
            clip_high=False,
            clip_low=False,
            high_bound=high_bound,
            low_bound=low_bound,
        )
    )
