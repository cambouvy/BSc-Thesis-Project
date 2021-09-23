from __future__ import division
import re
import itertools
from itertools import product
import gc
import os
import copy
import logging
import warnings

from IPython import embed
import pandas as pd
import numpy as np

root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)

has_dask = False
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError
try:
    from dask.diagnostics import visualize, ProgressBar
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    import dask.dataframe as dd

    has_dask = True
except ModuleNotFoundError:
    logger.warning("No dask installed, falling back to xarray")

from qlknn.dataset.data_io import (
    put_to_store_or_df,
    save_to_store,
    load_from_store,
    sep_prefix,
)
from qlknn.misc.analyse_names import (
    heat_vars,
    heat_flux,
    particle_vars,
    particle_diffusion_vars,
    momentum_vars,
    is_partial_particle,
    is_partial_momentum,
    is_pure_heat,
    split_name,
    is_transport,
    is_ion_scale,
    is_electron_scale,
    is_multi_scale,
    is_mode_scale,
    is_pure_flux,
    is_flux,
)
from qlknn.misc.tools import profile



@profile
def temperature_gradient_breakdown_filter(input, data, mode, debug_plot=False, patience=6):

    """
    Filter out fluxes where QuaLiKiz breaks down and underestimates the flux for large temperature gradients.

    Args:
        input (pd.DataFrame)     : Input DataFrame.
        data (pd.DataFrame)      : Data DataFrame.
        target (str)             : Target flux to filter. e.g. efiITG_GB, efeTEM_GB, efeETG_GB, etc.
        sensitivity (float)      : Sensitivity to minimum required flux increase.
    Returns:
        data (pd.DataFrame)      : Output DataFrame where 'wrong' fluxes have been replaced by NaN.

    """
    if mode == "ITG":
        target = "efiITG_GB"
    if mode == "TEM":
        target = "efeTEM_GB"
    elif mode == "ETG":
        target = "efeETG_GB"
    logger.info("Starting temperature gradient breakdown filter for {!s}".format(target))


    df = input.join(data)

    # Get list of input features and use them as index
    features = list(input.columns)
    df = df.set_index(features)

    # Group slices for multi-indexing.
    grouped = df.unstack("At")

    # Loop over slices to determine which indexes to drop from the DataFrame
    num_slices = len(grouped)
    drop_dimxs = []
    if debug_plot:
        import matplotlib.pyplot as plt
    for i, slc in enumerate(grouped.iterrows(), 1):
        slc_df = slc[1]
        slc_var = slc_df[target]
        # Skip all-zero slices
        if all(slc_var == 0):
            continue
        # Remove zeros from slice
        slc_var = slc_var.where(slc_var != 0)
        # Find point with negative gradient
        slc_ratios = np.diff(slc_var) / np.diff(slc_var.index)
        # Ignore warnings becausse of NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            neg_idx = np.flatnonzero(slc_ratios < 0)
        # Check if 'breakdown' happens, e.g. if there is no point higher than the point before negative gradient
        is_sick = [
            np.all(slc_var.iloc[idx + 1 : idx + 1 + patience] < slc_var.iloc[idx])
            for idx in neg_idx
        ]
        if not any(is_sick):
            continue
        # Get index of the first sick point, and mark all points after it for removal
        sick_idx = neg_idx[np.argmax(is_sick)] + 1
        sick_dimx = slc_df["dimx"].iloc[sick_idx:]
        if debug_plot:
            plt.scatter(slc_df[target].index, slc_df[target], color="green")
            plt.scatter(slc_var.index, slc_var, color="blue")
            plt.scatter(sick_dimx.index, slc_var.loc[sick_dimx.index], color="red")
            plt.show()
        drop_dimxs.extend(sick_dimx.dropna().astype("int").tolist())
        if logger.getEffectiveLevel() >= logging.INFO:
            print("%i / %i" % (i, num_slices), end="\r")

    logger.info("Dropping {:.4f}% of mode {!s}".format(len(drop_dimxs) / len(df) * 100, mode))

    # Determine column names of output targets to be filtered
    drop_columns = [col for col in data.columns if is_mode_scale(col, mode)]
    data[drop_columns] = data.loc[drop_dimxs, drop_columns]

    return data


@profile
def input_range_filter(inp, data, filter_ranges=None):
    """Filter the dataset based on an acceptable range of values from another
    dataset.

    Args:
        inp: Dataset containing values which determine acceptability by range.
             Needs to contain 'x'.
        data: The dataset to filter. Usually `pandas.DataFrame`.
        filter_ranges: Dict of lower and upper bounds, inclusive, of accepted range
                       for desired parameters in inp.
    """
    if filter_ranges is None:
        filter_ranges = {}
    within = pd.Series(np.full(len(inp), True, dtype="bool"), index=data.index)
    within &= ~inp["x"].isnull()
    pre = np.sum(within)
    for key in filter_ranges:
        low, high = filter_ranges[key]
        filt = (inp[key] >= low) & (inp[key] <= high)
        within &= filt
        logger.info(
            "{:5.2f}% of finite input points inside {!s:<9} range bounds".format(
                np.sum(filt) / pre * 100, key
            )
        )
    data = data.loc[within]
    return data


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
    For momentum fluxes: low_bound < abs(flux_div_flux) < high_bound

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

        # TODO: Check if this is still needed
        # if is_partial_particle(name) or is_partial_momentum(name):
        #    se = se.abs()

        # Apply the filter and save to store/dataframe
        filt = (low < se) & (se < high)
        if low > 0.0 or high < 0.0:
            filt = filt | (se == 0.0)
        put_to_store_or_df(store, se.name, store[group].loc[filt])
        logger.info(
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
    stability of the relevant mode: TEM for TEM and ITG for ITG, ETG for
    ETG. multi-scale (e.g. electron and ion-scale) for electron-heat-flux like
    vars (efe, dfe and family) and ion-scale otherwise (e.g. efi, pfe, pfi).
    TEM and ITG-stabilty are defined in`hypercube_to_pandas`. We define
    electron-unstable if we have a nonzero growthrate for kthetarhos <= 2,
    ion-unstable if we have a nonzero growthrate for kthetarhos > 2,
    and multiscale-unstable if electron-unstable or ion-unstable.

    Args:
        data: `pd.DataFrame` containing the data to be filtered, and `TEM`, `ITG` and `ETG`

    """
    for col in data.columns:
        if not is_transport(col):
            logger.debug("{!s} is not a transport coefficient, skipping..".format(col))
            continue

        # First check for which regime this variable should be filtered
        if is_mode_scale(col, "TEM"):
            gam_filter = "tem"
        elif is_mode_scale(col, "ITG"):
            gam_filter = "itg"
        elif is_mode_scale(col, "ETG"):
            gam_filter = "etg"
        elif is_ion_scale(col):
            gam_filter = "ion"
        elif is_multi_scale(col):
            gam_filter = "multi"
        else:
            logger.warning("Unable to determine scale of {!s}, skipping..".format(col))
            continue

        pre = np.sum(~data[col].isnull())
        logger.debug("Variable {!s} has gam filter {!s}".format(col, gam_filter))
        # Now apply a filter based on the regime of the variable
        if gam_filter == "tem":
            data[col] = data[col].loc[data["TEM"] == True]
        elif gam_filter == "itg":
            data[col] = data[col].loc[data["ITG"] == True]
        elif gam_filter == "etg":
            data[col] = data[col].loc[data["ETG"] == True]
        elif gam_filter == "ion":
            data[col] = data[col].loc[data.loc[:, ("ITG", "TEM")].any(axis=1)]
        elif gam_filter == "multi":
            data[col] = data[col].loc[data.loc[:, ("ITG", "TEM", "ETG")].any(axis=1)]
        logger.info(
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
    heat_cols = [col for col in data.columns if is_pure_heat(col) and is_pure_flux(col)]
    nonnegative = (data[heat_cols] >= 0).all(axis=1)
    return nonnegative

@profile
def negative_filter_to_zero(data):
    """Check if none of the heat-flux variables is negative

    Only checks on `heat_vars` e.g. efe_GB, efiTEM_GB etc.

    Args:
        data to perform the 'negative check' on

    Returns:
        Per-element `True` if the checked heat-flux variables is negative
    """
#     heat_cols = [col for col in data.columns if is_pure_heat(col) and is_pure_flux(col)]
    flux_cols = [col for col in data.columns if is_pure_heat(col) and is_flux(col)]
    nonnegative = (data[flux_cols] >= 0).all(axis=1)
    negative = ~nonnegative

    return negative, flux_cols


@profile
def ck_filter(data, bound):
    """ Check if convergence checks cke and cki are within bounds"""
    cke = cke_filter(data, bound)
    cki = cki_filter(data, bound)
    ck = cke & cki
    return ck


@profile
def cek_filter(data, bound):
    """ Check if convergence checks ceke and ceki are within bounds"""
    ceke = ceke_filter(data, bound)
    ceki = ceki_filter(data, bound)
    cek = ceke & ceki
    return cek


@profile
def cke_filter(data, bound):
    """ Check if convergence check cke is within bounds"""
    cke = data["cke"].abs() < bound
    return cke


@profile
def cki_filter(data, bound):
    """ Check if convergence checks cki are within bounds"""
    cki = data["cki"].abs() < bound
    return cki


@profile
def ceke_filter(data, bound):
    """ Check if convergence check ceke is within bounds"""
    ceke = data["ceke"].abs() < bound
    return ceke


@profile
def ceki_filter(data, bound):
    """ Check if convergence checks ceki are within bounds"""
    ceki = data["ceki"].abs() < bound
    return ceki


@profile
def septot_filter(data, septot_factor, startlen=None):
    """ Check if ITG/TEM/ETG heat flux !>> total_flux"""
    if startlen is None and has_dask and not isinstance(data, dd.DataFrame):
        startlen = len(data)
    difference_okay = pd.Series(np.full(len(data), True, dtype="bool"), index=data.index)
    sepnames = []
    for type, spec in product(heat_flux, ["i", "e"]):
        totname = type + spec + "_GB"
        if spec == "i":  # no ETG
            seps = ["ITG", "TEM"]
        else:  # All modes
            seps = ["ETG", "ITG", "TEM"]
        sepnames = [
            type + spec + sep + "_GB" for sep in seps if type + spec + sep + "_GB" in data.columns
        ]
        logger.debug("Checking {!s}".format(sepnames))
        difference_okay &= (
            data[sepnames].abs().le(septot_factor * data[totname].abs(), axis=0).all(axis=1)
        )
        if startlen is not None:
            logger.info(
                "After filter {!s:<6} {!s:<6} {:.2f}% left".format(
                    "septot", totname, 100 * difference_okay.sum() / startlen
                )
            )
    return difference_okay


@profile
def ambipolar_filter(data, bound, version=1):
    """ Check if ambipolarity is conserved """
    if version == 0:
        ambi_failed = (data["absambi"] < bound) & (data["absambi"] > 1 / bound)
    elif version == 1:
        ambi_failed = ((data["absambi"] - 1.0) < bound) & ((data["absambi"] - 1.0) > -bound)
    else:
        raise Exception("Unknown ambipolar_filter version passed!")
    return ambi_failed


@profile
def femtoflux_filter(data, bound):
    """ Check if transport coefficient is no 'femto_flux', a very small non-zero flux"""
    transport_coeffs = [col for col in data if is_transport(col) and is_pure_flux(col)]
    no_femto = pd.Series(np.full(len(data), True, dtype="bool"), index=data.index)
    abstransport = data[transport_coeffs].abs()
    no_femto &= ~((abstransport < bound) & (abstransport != 0)).any(axis=1)
    return no_femto


def sanity_filter(
    data,
    septot_factor,
    ambi_bound,
    femto_bound,
    cke_bound=np.inf,
    cki_bound=np.inf,
    ceke_bound=np.inf,
    ceki_bound=np.inf,
    stored_negative_filter=None,
    stored_ck_filter=None,
    stored_cke_filter=None,
    stored_cki_filter=None,
    stored_cek_filter=None,
    stored_ceke_filter=None,
    stored_ceki_filter=None,
    stored_ambipolar_filter=None,
    stored_septot_filter=None,
    stored_femtoflux_filter=None,
    startlen=None,
    ambipolar_filter_version=1,
):
    """Filter out insane points

    There are points where we do not trust QuaLiKiz. These points are
    filtered out by functions defined in this module. Currently:

        negative_filter:  Points with negative heat flux
        ck_filter:        Points with too high convergence errors
        cke_filter:       Points with too high convergence errors
        cki_filter:       Points with too high convergence errors
        cek_filter:       Points with too high convergence errors
        ceke_filter:      Points with too high convergence errors
        ceki_filter:      Points with too high convergence errors
        septot_factor:    Points where sep flux >> total flux
        ambipolar_filter: Points that don't conserve ambipolarity
        femtoflux_filter: Point with very tiny transport

    Optionally one can provide a earlier-stored filter in the form
    of a list of indices to remove from the dataset
    Args:
        ck_bound:        Maximum ck[i/e]
        cke_bound:       Maximum cke
        cki_bound:       Maximum cki
        cek_bound:       Maximum cek[i/e]
        ceke_bound:      Maximum ceke
        ceki_bound:      Maximum ceki
        septot_factor:   Maximum factor between tot flux and sep flux
        ambi_bound:      Maximum factor between dq_i/dt and dq_e/dt
        femto_bound:     Maximum value of what is defined as femtoflux

    Kwargs:
        stored_[filter]: List of indices contained in filter [Default: None]
        starlen:         Total amount of points at start of function. By
                         default all points in dataset.
    """
    if startlen is None and has_dask and not isinstance(data, dd.DataFrame):
        startlen = len(data)

    if np.isfinite(cke_bound):
        # Throw away point if cke too high
        if stored_cke_filter is None:
            logger.debug("Applying functional filter")
            data = data.loc[cke_filter(data, cke_bound)]
        else:
            logger.debug("Applying stored filter")
            data = data.reindex(index=data.index.difference(stored_cke_filter), copy=False)
        if startlen is not None:
            logger.info(
                "After filter {!s:<13} {:.f}% left".format("cke", 100 * len(data) / startlen)
            )
        else:
            logger.info("filter {!s:<13} done".format("cke"))
        gc.collect()

    if np.isfinite(cki_bound):
        # Throw away point if cki too high
        if stored_cki_filter is None:
            logger.debug("Applying functional filter")
            data = data.loc[cki_filter(data, cki_bound)]
        else:
            logger.debug("Applying stored filter")
            data = data.reindex(index=data.index.difference(stored_cki_filter), copy=False)
        if startlen is not None:
            logger.info(
                "After filter {!s:<13} {:.2f}% left".format("cki", 100 * len(data) / startlen)
            )
        else:
            logger.info("filter {!s:<13} done".format("cki"))
        gc.collect()

    if np.isfinite(ceke_bound):
        # Throw away point if ceke too high
        if stored_ceke_filter is None:
            logger.debug("Applying functional filter")
            data = data.loc[ceke_filter(data, ceke_bound)]
        else:
            logger.debug("Applying stored filter")
            data = data.reindex(index=data.index.difference(stored_ceke_filter), copy=False)
        if startlen is not None:
            logger.info(
                "After filter {!s:<13} {:.2f}% left".format("ceke", 100 * len(data) / startlen)
            )
        else:
            logger.info("filter {!s:<13} done".format("ceke"))
        gc.collect()

    if np.isfinite(ceki_bound):
        # Throw away point if ceki too high
        if stored_ceki_filter is None:
            logger.debug("Applying functional filter")
            data = data.loc[ceki_filter(data, ceki_bound)]
        else:
            logger.debug("Applying stored filter")
            data = data.reindex(index=data.index.difference(stored_ceki_filter), copy=False)
        if startlen is not None:
            logger.info(
                "After filter {!s:<13} {:.2f}% left".format("ceki", 100 * len(data) / startlen)
            )
        else:
            logger.info("filter {!s:<13} done".format("ceki"))
        gc.collect()

    # Throw away point if negative heat flux
    # if stored_negative_filter is None:
    #     logger.debug("Applying functional filter")
    #     data = data.loc[negative_filter(data)]
    # else:
    #     logger.debug("Applying stored filter")
    #     data = data.reindex(index=data.index.difference(stored_negative_filter), copy=False)
    # if startlen is not None:
    #     logger.info(
    #         "After filter {!s:<13} {:.4f}% left".format("negative", 100 * len(data) / startlen)
    #     )
    # else:
    #     logger.info("filter {!s:<13} done".format("negative"))
    # gc.collect()

    # Clip negative heat flux to zero
    if stored_negative_filter is None:
        logger.debug("Applying functional filter")
        neg_filter, flux_cols = negative_filter_to_zero(data)
        data.loc[neg_filter, flux_cols] = 0
    else:
        logger.debug("Applying stored filter")
        data = data.reindex(index=data.index.difference(stored_negative_filter), copy=False)
    if startlen is not None:
        logger.info(
            "After filter {!s:<13} {:.4f}% left".format("negative", 100 * len(data) / startlen)
        )
    else:
        logger.info("filter {!s:<13} done".format("negative"))
    gc.collect()

    # Throw away point if sep flux is way higher than tot flux
    if stored_septot_filter is None:
        logger.debug("Applying functional filter")
        data = data.loc[septot_filter(data, septot_factor, startlen=startlen)]
    else:
        logger.debug("Applying stored filter")
        data = data.reindex(index=data.index.difference(stored_septot_filter), copy=False)
    if startlen is not None:
        logger.info(
            "After filter {!s:<13} {:.4f}% left".format("septot", 100 * len(data) / startlen)
        )
    else:
        logger.info("filter {!s:<13} done".format("septot"))
    gc.collect()

    # Throw away point if ambipolarity is not conserved
    if stored_ambipolar_filter is None:
        logger.debug("Applying functional filter")
        data = data.loc[ambipolar_filter(data, ambi_bound, version=ambipolar_filter_version)]
    else:
        logger.debug("Applying stored filter")
        data = data.reindex(index=data.index.difference(stored_ambipolar_filter), copy=False)
    if startlen is not None:
        logger.info(
            "After filter {!s:<13} {:.4f}% left".format("ambipolar", 100 * len(data) / startlen)
        )
    else:
        logger.info("filter {!s:<13} done".format("ambipolar"))
    gc.collect()

    # Throw away point if it is a femtoflux
    if stored_femtoflux_filter is None:
        logger.debug("Applying functional filter")
        data = data.loc[femtoflux_filter(data, femto_bound)]
    else:
        logger.debug("Applying stored filter")
        data = data.reindex(index=data.index.difference(stored_femtoflux_filter), copy=False)
    if startlen is not None:
        logger.info(
            "After filter {!s:<13} {:.4f}% left".format("femtoflux", 100 * len(data) / startlen)
        )
    else:
        logger.info("filter {!s:<13} done".format("femtoflux"))
    gc.collect()

    # Alternatively:
    # data = data.loc[filter_negative(data) & filter_ck(data, ck_bound) & filter_septot(data, septot_factor)]

    return data


filter_functions = {
    "negative": negative_filter,
    "ck": ck_filter,
    "cke": cke_filter,
    "cki": cki_filter,
    "cek": cek_filter,
    "ceke": ceke_filter,
    "ceki": ceki_filter,
    "septot": septot_filter,
    "ambipolar": ambipolar_filter,
    "femtoflux": femtoflux_filter,
}

filter_defaults = {
    "div": {
        "efeITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "pfeITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "efiTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
        "pfeTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
        "dfeITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "dfiITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "vceITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "vciITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "vteITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "vtiITG_GB_div_efiITG_GB": (-np.inf, np.inf),
        "dfeTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
        "dfiTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
        "vceTEM_GB_div_efeTEM_GB": (-np.inf, np.inf),
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
        logger.info("Calculated {!s}".format(prefix + name))


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
def filter_nans(input):
    input.dropna(inplace=True)


@profile
def filter_Zeff_Nustar(input, Zeff=1, Nustar=1e-3):
    """ Filter out Zeff and Nustar """
    if "Nustar" in input.columns:
        sel = np.isclose(input["Zeff"], Zeff, atol=1e-5, rtol=1e-3) & np.isclose(
            input["Nustar"], Nustar, atol=1e-5, rtol=1e-3
        )
    elif "logNustar" in input.columns:
        sel = np.isclose(input["Zeff"], Zeff, atol=1e-5, rtol=1e-3) & np.isclose(
            input["logNustar"], np.log10(Nustar), atol=1e-5, rtol=1e-3
        )
    else:
        raise Exception("No Nustar-like variable in dataset")
    idx = input.index[sel]
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
    for name in ["Zeff", "logNustar"]:
        consts[7][name] = float(inputs[7].head(1)[name].values)
    inputs[7].drop(["Zeff", "logNustar"], axis="columns", inplace=True)

    idx[4] = filter_Ate_An_x(inputs[7])
    inputs[4] = inputs[7].loc[idx[4]]
    for name in ["Ate", "An", "x"]:
        consts[4][name] = float(inputs[4].head(1)[name].values)
    inputs[4].drop(["Ate", "An", "x"], axis="columns", inplace=True)

    return idx, inputs, consts


@profile
def split_dims(
    input,
    data,
    const,
    gen,
    prefix="",
    filter_num="",
    split_func=split_karel9D_input,
    compress=True,
):
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
    if compress:
        suffix = ".1"
    else:
        suffix = ""
    idx, inputs, consts = split_func(input, const)
    subdims = list(idx.keys())
    subdims.remove(max(idx.keys()))
    for dim in sorted(subdims):
        logger.info("splitting {!s}".format(dim))
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
            + suffix
        )
        save_to_store(inputs[dim], data.loc[idx[dim]], consts[dim], store_name, compress=compress)


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
def split_test_train(input, data, const, filter_name, rootdir=".", compress=True):
    logger.info("Splitting subsets")
    if compress:
        suffix = ".1"
    else:
        suffix = ""

    save_to_store(
        input.loc[data["is_test"], :],
        data.loc[data["is_test"], :],
        const,
        os.path.join(rootdir, "test_" + filter_name + ".h5" + suffix),
        compress=compress,
    )
    save_to_store(
        input.loc[~data["is_test"], :],
        data.loc[~data["is_test"], :],
        const,
        os.path.join(rootdir, "training_" + filter_name + ".h5" + suffix),
        compress=compress,
    )
