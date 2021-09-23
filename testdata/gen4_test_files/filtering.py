from __future__ import division
import re
from itertools import product
import gc
import os
import warnings
from collections import OrderedDict
import copy

from IPython import embed
import pandas as pd
import numpy as np
import dask.dataframe as dd

from qlknn.dataset.data_io import (
    put_to_store_or_df,
    save_to_store,
    load_from_store,
    sep_prefix,
)
from qlknn.misc.analyse_names import (
    heat_vars,
    particle_vars,
    particle_diffusion_vars,
    momentum_vars,
    is_partial_diffusion,
    is_partial_particle,
    is_pure_heat,
    split_name,
    is_transport,
)
from qlknn.misc.tools import profile


@profile
def regime_filter(data, geq, less):
    """Filter the dataset based on the total ion/electron heat flux
    This filter is used to constain the dataset to experimentally relevant
    fluxes. We have an estimate for the expected total heat flux, and
    filter out the full datapoint.

    Args:
        data: The dataset to filter. Usually `pandas.DataFrame`. Needs to
              contain 'efi_GB' and 'efe_GB'.
        geq:  Lower bound on ef[i|e] heat flux. Inclusive bound.
        less: Upper bound on ef[i|e] heat flux. Exclusive bound.
    """
    within = pd.Series(np.full(len(data), True, dtype="bool"), index=data.index)
    within &= (data["efe_GB"] < less) & (data["efi_GB"] < less)
    within &= (data["efe_GB"] >= geq) & (data["efi_GB"] >= geq)
    data = data.loc[within]
    return data


@profile
def div_filter(store, filter_bounds=None):
    """Filter flux_div_flux variables based on bounds
    We know from experience the maximum relative difference in flux between
    the ions and electrons of different modes. As such we remove the fluxpoint
    if it falls outside the given bounds.

    For heat fluxes:     low_bound < flux_div_flux < high_bound.
    For particle fluxes: low_bound < abs(flux_div_flux) < high_bound

    Note: Technically, as heat fluxes are non-negative anyway, we could use the same
    bounds.

    Args:
        store:         The store name or `pd.HDFStore` to apply the filter to

    Kwargs:
        filter_bounds: A dictionary with as keys the flux_div_flux variable to filter
                       and values a tuple with (low_bound, high_bound). For defaults,
                       see `filter_defaults['div']`
    """
    all_filter_bounds = copy.deepcopy(filter_defaults["div"])
    if filter_bounds is None:
        filter_bounds = {}
    all_filter_bounds.update(filter_bounds)

    for group in store:
        if isinstance(store, pd.HDFStore):
            name = group.lstrip(sep_prefix)
        else:
            name = group

        # Read filter bound. If not in dict, skip this variable
        if name in all_filter_bounds:
            low, high = all_filter_bounds[name]
        else:
            continue

        # Load the variable from store, save the name
        se = store[group]
        se.name = name
        # And save the pre-filter instances for our debugging print
        pre = np.sum(~se.isnull())

        if is_partial_particle(name):
            se = se.abs()

        # Apply the filter and save to store/dataframe
        put_to_store_or_df(store, se.name, store[group].loc[(low < se) & (se < high)])
        print(
            "{:5.2f}% of sane unstable {!s:<9} points inside div bounds".format(
                np.sum(~store[group].isnull()) / pre * 100, group
            )
        )
    return store


@profile
def stability_filter(data):
    """Filter out the stable points based on growth rate

    QuaLiKiz gives us growth rate information, so we can use this to filter
    out all stable points from the dataset. Of course, we only check the
    stability of the relevant mode: TEM for TEM and ITG for ITG, 'elec' for
    ETG (legacy, sorry). multi-scale (e.g. electron and ion-scale) for
    electron-heat-flux like vars (efe, dfe and family) and ion-scale
    otherwise (e.g. efi, pfe, pfi). TEM and ITG-stabilty are defined in
    `hypercube_to_pandas`. We define electron-unstable if we have a nonzero
    growthrate for kthetarhos <= 2, ion-unstable if we have a nonzero
    growthrate for kthetarhos > 2, and multiscale-unstable if electron-unstable
    or ion-unstable.

    Args:
        data: `pd.DataFrame` containing the data to be filtered, and `TEM`, `ITG`,
              `gam_leq_GB`, `gam_great_GB`

    """
    for col in data.columns:
        splitted = re.compile("(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)").split(col)
        try:
            transp, species, mode, norm = split_name(col)
        except ValueError:
            print("skipping {!s}".format(col))

        if not is_transport(col):
            print("skipping {!s}".format(col))
            continue
        # First check for which regime this variable should be filtered
        if mode == "TEM":
            gam_filter = "tem"
        elif mode == "ITG":
            gam_filter = "itg"
        elif mode == "ETG":
            gam_filter = "elec"
        elif transp in heat_vars and species == "e":
            gam_filter = "multi"
        else:
            gam_filter = "ion"

        pre = np.sum(~data[col].isnull())
        print(col, gam_filter)
        # Now apply a filter based on the regime of the variable
        if gam_filter == "ion":
            data[col] = data[col].loc[data["gam_leq_GB"] != 0]
        elif gam_filter == "elec":
            data[col] = data[col].loc[data["gam_great_GB"] != 0]
        elif gam_filter == "multi":
            if "gam_great_GB" in data.columns:
                data[col] = data[col].loc[(data["gam_leq_GB"] != 0) | (data["gam_great_GB"] != 0)]
            else:
                # If gam_great_GB is not there, the QuaLiKiz run was not with electron scale
                data[col] = data[col].loc[(data["gam_leq_GB"] != 0)]
        elif gam_filter == "tem":
            data[col] = data[col].loc[data["TEM"] == True]
        elif gam_filter == "itg":
            data[col] = data[col].loc[data["ITG"] == True]
        print(
            "{:5.2f}% of sane {!s:<9} points unstable at {!s:<5} scale".format(
                np.sum(~data[col].isnull()) / pre * 100, col, gam_filter
            )
        )
    return data


@profile
def negative_filter(data):
    """Check if none of the heat-flux variables is negative

    Only checks on `heat_vars` e.g. efe_GB, efiTEM_GB etc.

    Args:
        data to perform the 'negative check' on

    Returns:
        Per-element `True` if none of the checked heat-flux variables is negative
    """
    filter_functions = {
        "negative": negative_filter,
        "ck": ck_filter,
        "septot": septot_filter,
        "ambipolar": ambipolar_filter,
        "femtoflux": femtoflux_filter,
    }
    # anyisneg = pd.Series(np.full(len(data), True, dtype='bool'), index=data.index)
    heat_cols = [col for col in data.columns if is_pure_heat(col)]
    nonnegative = (data[heat_cols] >= 0).all(axis=1)
    return nonnegative


@profile
def ck_filter(data, bound):
    """ Check if convergence checks cki and cki are within bounds"""
    ck = (data[["cki", "cke"]].abs() < bound).all(axis=1)
    return ck


@profile
def septot_filter(data, septot_factor, startlen=None):
    """ Check if ITG/TEM/ETG heat flux !>> total_flux"""
    if startlen is None and not isinstance(data, dd.DataFrame):
        startlen = len(data)
    difference_okay = pd.Series(np.full(len(data), True, dtype="bool"), index=data.index)
    sepnames = []
    for type, spec in product(heat_vars, ["i", "e"]):
        totname = type + spec + "_GB"
        if spec == "i":  # no ETG
            seps = ["ITG", "TEM"]
        else:  # All modes
            seps = ["ETG", "ITG", "TEM"]
        sepnames = [
            type + spec + sep + "_GB" for sep in seps if type + spec + sep + "_GB" in data.columns
        ]
        print("Checking {!s}".format(sepnames))
        difference_okay &= (
            data[sepnames].abs().le(septot_factor * data[totname].abs(), axis=0).all(axis=1)
        )
        if startlen is not None:
            print(
                "After filter {!s:<6} {!s:<6} {:.2f}% left".format(
                    "septot", totname, 100 * difference_okay.sum() / startlen
                )
            )
    return difference_okay


@profile
def ambipolar_filter(data, bound):
    """ Check if ambipolarity is conserved """
    return (data["absambi"] < bound) & (data["absambi"] > 1 / bound)


@profile
def femtoflux_filter(data, bound):
    """ Check if flux is no 'femto_flux', a very small non-zero flux"""
    fluxes = [
        col
        for col in data
        if len(re.compile("(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)").split(col)) == 5
        if re.compile("(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)").split(col)[0]
        in particle_vars + heat_vars + momentum_vars
    ]
    no_femto = pd.Series(np.full(len(data), True, dtype="bool"), index=data.index)
    absflux = data[fluxes].abs()
    no_femto &= ~((absflux < bound) & (absflux != 0)).any(axis=1)
    return no_femto


def sanity_filter(
    data,
    ck_bound,
    septot_factor,
    ambi_bound,
    femto_bound,
    stored_negative_filter=None,
    stored_ck_filter=None,
    stored_ambipolar_filter=None,
    stored_septot_filter=None,
    stored_femtoflux_filter=None,
    startlen=None,
):
    """Filter out insane points

    There are points where we do not trust QuaLiKiz. These points are
    filtered out by functions defined in this module. Currently:

        negative_filter:  Points with negative heat flux
        ck_filter:        Points with too high convergence errors
        septot_factor:    Points where sep flux >> total flux
        ambipolar_filter: Points that don't conserve ambipolarity
        femtoflux_filter: Point with very tiny fluxes

    Optionally one can provide a earlier-stored filter in the form
    of a list of indices to remove from the dataset
    Args:
        ck_bound:        Maximum ck[i/e]
        septot_factor:   Maximum factor between tot flux and sep flux
        ambi_bound:      Maximum factor between dq_i/dt and dq_e/dt
        femto_bound:     Maximum value of what is defined as femtoflux

    Kwargs:
        stored_[filter]: List of indices contained in filter [Default: None]
        starlen:         Total amount of points at start of function. By
                         default all points in dataset.
    """
    if startlen is None and not isinstance(data, dd.DataFrame):
        startlen = len(data)
    # Throw away point if negative heat flux
    if stored_negative_filter is None:
        data = data.loc[negative_filter(data)]
    else:
        data = data.reindex(index=data.index.difference(stored_negative_filter), copy=False)
    if startlen is not None:
        print("After filter {!s:<13} {:.2f}% left".format("negative", 100 * len(data) / startlen))
    else:
        print("filter {!s:<13} done".format("negative"))
    gc.collect()

    # Throw away point if cke or cki too high
    if stored_ck_filter is None:
        data = data.loc[ck_filter(data, ck_bound)]
    else:
        data = data.reindex(index=data.index.difference(stored_ck_filter), copy=False)
    if startlen is not None:
        print("After filter {!s:<13} {:.2f}% left".format("ck", 100 * len(data) / startlen))
    else:
        print("filter {!s:<13} done".format("ck"))
    gc.collect()

    # Throw away point if sep flux is way higher than tot flux
    if stored_septot_filter is None:
        data = data.loc[septot_filter(data, septot_factor, startlen=startlen)]
    else:
        data = data.reindex(index=data.index.difference(stored_septot_filter), copy=False)
    if startlen is not None:
        print("After filter {!s:<13} {:.2f}% left".format("septot", 100 * len(data) / startlen))
    else:
        print("filter {!s:<13} done".format("septot"))
    gc.collect()

    # Throw away point if ambipolarity is not conserved
    if stored_ambipolar_filter is None:
        data = data.loc[ambipolar_filter(data, ambi_bound)]
    else:
        data = data.reindex(index=data.index.difference(stored_ambipolar_filter), copy=False)
    if startlen is not None:
        print(
            "After filter {!s:<13} {:.2f}% left".format("ambipolar", 100 * len(data) / startlen)
        )
    else:
        print("filter {!s:<13} done".format("ambipolar"))
    gc.collect()

    # Throw away point if it is a femtoflux
    if stored_femtoflux_filter is None:
        data = data.loc[femtoflux_filter(data, femto_bound)]
    else:
        data = data.reindex(index=data.index.difference(stored_femtoflux_filter), copy=False)
    if startlen is not None:
        print(
            "After filter {!s:<13} {:.2f}% left".format("femtoflux", 100 * len(data) / startlen)
        )
    else:
        print("filter {!s:<13} done".format("femtoflux"))
    gc.collect()

    # Alternatively:
    # data = data.loc[filter_negative(data) & filter_ck(data, ck_bound) & filter_septot(data, septot_factor)]

    return data


filter_functions = {
    "negative": negative_filter,
    "ck": ck_filter,
    "septot": septot_filter,
    "ambipolar": ambipolar_filter,
    "femtoflux": femtoflux_filter,
}

filter_defaults = {
    "div": {
        "pfeTEM_GB": (0.02, 20),
        "pfeITG_GB": (0.02, 10),
        "efiTEM_GB": (0.05, np.inf),
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
    "ck": 50,
    "septot": 1.5,
    "ambipolar": 1.5,
    "femtoflux": 1e-4,
}


def create_stored_filter(store, data, filter_name, filter_setting):
    """Create filter index from filter function

    This function applies the filter specified by `filter_name` and
    finds the indices of the data that would be filtered out from
    `data` and saves it in `store[filter/filter_name]`

    Args:
        store:          `pandas.HDFStore` to save filter indices in
        data:           Data to apply filter to
        filter_name:    Name of filter to apply
        filter_setting: Filter-specific settings. Given to filter function
    """
    filter_func = filter_functions[filter_name]
    name = "".join(["/filter/", filter_name])
    if filter_setting is not None:
        name = "_".join([name, str(filter_setting)])
        var = data.index[~filter_func(data, filter_setting)].to_series()
    else:
        var = data.index[~filter_func(data)].to_series()
    store.put(name, var)


def load_stored_filter(store, filter_name, filter_setting):
    """ Load saved filter by `create_stored_filter` from disk"""
    name = "".join(["/filter/", filter_name])
    try:
        if filter_setting is not None:
            name = "".join([name, "_", str(filter_setting)])
            filter = store.get(name)
        else:
            filter = store.get(name)
    except KeyError:
        filter = None
    return filter


gen3_div_names_base = [
    "efeITG_GB_div_efiITG_GB",
    "pfeITG_GB_div_efiITG_GB",
    "efiTEM_GB_div_efeTEM_GB",
    "pfeTEM_GB_div_efeTEM_GB",
]

gen3_div_names_dv = [
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
gen3_div_names = gen3_div_names_base + gen3_div_names_dv


@profile
def create_divsum(store, divnames=gen3_div_names):
    """Create individual targets needed vor divsum-style networks

    This function takes a list of div-style targets, for example
    'efeITG_GB_div_efiITG_GB', and creates this variable from its separate
    parts 'efeITG' and 'efiITG'

    Args:
        store:    The `pd.HDFStore` where to store the divsums

    Kwargs:
        divnames: A list of 'flux_div_flux' strings. By default creates the
                  targets needed to train gen3/4 networks. [Default: gen3_div_names]
    """
    if isinstance(store, pd.HDFStore):
        prefix = sep_prefix
    else:
        prefix = ""

    for name in divnames:
        one, two = re.compile("_div_").split(name)
        one, two = store[prefix + one], store[prefix + two]
        res = (one / two).dropna()
        put_to_store_or_df(store, name, res)


def create_divsum_legacy(store):
    for group in store:
        if isinstance(store, pd.HDFStore):
            group = group[1:]
        splitted = re.compile("(?=.*)(.)(|ITG|ETG|TEM)(_GB|SI|cm)").split(group)
        if splitted[0] in heat_vars and splitted[1] == "i" and len(splitted) == 5:
            group2 = splitted[0] + "e" + "".join(splitted[2:])
            sets = [
                ("_".join([group, "plus", group2]), store[group] + store[group2]),
                ("_".join([group, "div", group2]), store[group] / store[group2]),
                ("_".join([group2, "div", group]), store[group2] / store[group]),
            ]
        elif splitted[0] == "pf" and splitted[1] == "e" and len(splitted) == 5:
            group2 = "efi" + "".join(splitted[2:])
            group3 = "efe" + "".join(splitted[2:])
            sets = [
                (
                    "_".join([group, "plus", group2, "plus", group3]),
                    store[group] + store[group2] + store[group3],
                ),
                ("_".join([group, "div", group2]), store[group] / store[group2]),
                ("_".join([group, "div", group3]), store[group] / store[group3]),
            ]

        else:
            continue

        for name, set in sets:
            set.name = name
            put_to_store_or_df(store, set.name, set)
    return store


@profile
def create_rotdiv(input, data, rotvar="Machtor", rot_over_norot=True):
    rotvar = "Machtor"
    transp = [col for col in data.columns if is_transport(col)]
    total = pd.concat([data, input], axis=1)
    total = total.set_index(list(input.columns))
    # rotdiv_cols = tuple(name + '_rot0_div_' + name for name in transp)
    # total = pd.concat([total, pd.DataFrame(np.full((len(total), len(transp)), np.NaN), index=total.index, columns=rotdiv_cols)], axis=1)
    new_order = list(range(len(total.index.names)))
    rot_idx = total.index.names.index(rotvar)
    new_order.remove(rot_idx)
    new_order.insert(0, rot_idx)
    total = total.reorder_levels(new_order)
    dfs = {}
    for val in total.index.get_level_values(rotvar).unique():
        if rot_over_norot:
            df = total.loc[val, transp] / total.loc[0, transp]
            df = df.where(total.loc[0, transp] != 0, 0)
            df.columns = [col + "_div_" + col + "_rot0" for col in df.columns]
        else:
            df = total.loc[0, transp] / total.loc[val, transp]
            df = df.where(total.loc[val, transp] != 0, 0)
            df.columns = [col + "_rot0_div_" + col for col in df.columns]
        dfs[val] = df
    rotdivs = pd.concat(dfs, names=[rotvar] + df.index.names)
    data = pd.concat([total, rotdivs], axis=1)
    data.reset_index(inplace=True)
    input = data.loc[:, list(input.columns)]
    for col in input.columns:
        del data[col]
    return input, data


@profile
def filter_Zeff_Nustar(input, Zeff=1, Nustar=1e-3):
    """ Filter out Zeff and Nustar """
    idx = input.index[
        (
            np.isclose(input["Zeff"], Zeff, atol=1e-5, rtol=1e-3)
            & np.isclose(input["Nustar"], Nustar, atol=1e-5, rtol=1e-3)
        )
    ]
    return idx


@profile
def filter_Ate_An_x(input, Ate=6.5, An=2, x=0.45):
    """ Filter out Ate, An and x"""

    idx = input.index[
        (
            np.isclose(input["Ate"], Ate, atol=1e-5, rtol=1e-3)
            & np.isclose(input["An"], An, atol=1e-5, rtol=1e-3)
            & np.isclose(input["x"], x, atol=1e-5, rtol=1e-3)
        )
    ]
    return idx


@profile
def split_karel9D_input(input, const):
    """Split karel-style 9D input data in 9, 7 and 4D

    The karel-style 9D dataset has as input/features:
    [Zeff, Ati, Ate, An, q, smag, x, Ti_Te, Nustar]
    We split this dataset in a 7D one by choosing Zeff and Nustar = const,
    and further split this dataset to 4D by choosing Ate, An, and x = const

    Args:
        input:   The dataframe containing at least Zeff, Nustar, Ate, An, and x
        const:   Series containing the variables constant for this dataset

    Returns:
        idx:     Dict with indexes for the 9D, 7D, and 4D dataset
        inputs:  Dict with already-split (input) dataframes
        consts:  Dict with constant variables for the given dataset
    """
    idx = {}
    consts = {9: const.copy(), 7: const.copy(), 4: const.copy()}
    idx[7] = filter_Zeff_Nustar(input)

    inputs = {9: input}
    idx[9] = input.index
    inputs[7] = input.loc[idx[7]]
    for name in ["Zeff", "Nustar"]:
        consts[7][name] = float(inputs[7].head(1)[name].values)
    inputs[7].drop(["Zeff", "Nustar"], axis="columns", inplace=True)

    idx[4] = filter_Ate_An_x(inputs[7])
    inputs[4] = inputs[7].loc[idx[4]]
    for name in ["Ate", "An", "x"]:
        consts[4][name] = float(inputs[4].head(1)[name].values)
    inputs[4].drop(["Ate", "An", "x"], axis="columns", inplace=True)

    return idx, inputs, consts


@profile
def split_dims(input, data, const, gen, prefix="", split_func=split_karel9D_input):
    """Split full dataset in lower-D subsets and save to store


    Args:
        input:      Dataframe containing input/feature variables
        data:       Dataframe containing output/target variables
        const:      Series containing constants for this dataset
        gen:        Generation indicator. Needed to generate store name

    Kwargs:
        prefix:     Prefix to store name. [Default: '']
        split_func: Function use to split the dataset. Signature should match
                    the default function. [Default: `split_karel9D_input`]
    """
    idx, inputs, consts = split_func(input, const)
    subdims = list(dx.keys())
    subdims.remove(max(idx.keys()))
    for dim in sorted(subdims):
        print("splitting", dim)
        store_name = (
            prefix
            + "gen"
            + str(gen)
            + "_"
            + str(dim)
            + "D_nions0_flat"
            + "_filter"
            + str(filter_num)
            + ".h5"
        )
        save_to_store(inputs[dim], data.loc[idx[dim]], consts[dim], store_name)


@profile
def generate_test_train_index(input, data, const, frac=0.1):
    """ Randomly split full dataset in 'test' and 'training' and save to store """
    rand_index = pd.Int64Index(np.random.permutation(input.index.copy(deep=True)))
    sep_index = int(frac * len(rand_index))
    # idx = {}
    # idx['test'] = rand_index[:sep_index]
    # idx['training'] = rand_index[sep_index:]
    data["is_test"] = False
    data.loc[rand_index[:sep_index], "is_test"] = True


@profile
def split_test_train(input, data, const, filter_name, root_dir="."):
    print("Splitting subsets")
    save_to_store(
        input.loc[data["is_test"], :],
        data.loc[data["is_test"], :],
        const,
        os.path.join(root_dir, "test_" + filter_name + ".h5"),
    )
    save_to_store(
        input.loc[~data["is_test"], :],
        data.loc[~data["is_test"], :],
        const,
        os.path.join(root_dir, "training_" + filter_name + ".h5"),
    )
