import gc
from collections import OrderedDict
import warnings
import re
import logging
from typing import Optional, Mapping, Sequence

root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)

import pandas as pd
import numpy as np
from IPython import embed

try:
    import dask.dataframe as dd

    has_dask = True
except ImportError:
    warnings.warn("Dask not found")
    has_dask = False

try:
    profile
except NameError:
    from qlknn.misc.tools import profile

from qlknn.misc.analyse_names import (
    heat_vars,
    particle_vars,
    particle_diffusion_vars,
    momentum_vars,
    is_flux,
    is_growth,
)
from qlknn.misc.tools import first

store_format = "fixed"
sep_prefix = "/output/"


@profile
def convert_nustar(input_df):
    # Nustar relates to the targets with a log
    try:
        input_df["logNustar"] = np.log10(input_df["Nustar"])
        del input_df["Nustar"]
    except KeyError:
        logger.warn("No Nustar in dataset")
    return input_df


def put_to_store_or_df(store_or_df, name, var, store_prefix=sep_prefix):
    if isinstance(store_or_df, pd.HDFStore):
        store_or_df.put("".join([store_prefix, name]), var, format=store_format)
    else:
        store_or_df[name] = var


from qlknn.misc.analyse_names import split_parts, extract_operations, extract_part_names

_op_convert = {
    "_div_": "/",
    "_plus_": "+",
}


# This function is used as "prototype Sphinx autofunction" template
# In Sphinx speak, this will end up as .. autofunction:: combine_vars
# from the sphinx.ext.autodoc module
#
# If sphinx-autodoc-typehints cannot find the type of the optional argument
# in the Keyword Args list, it will generate an ugly stub in the Args list
# This depends on docs/source/conf.py
#
# Easiest is to use Pythons type checking features, see
# https://docs.python.org/3.6/library/typing.html
# Using the ``Optional`` class for input variables is optional
# Optional[str] is just shorthand for "``str`` or ``None``"
# For output variables it's required for the parser to know what you mean
#
# Max linelength is determined by Black, check pyproject.toml.
# For source code, try to stay within 98 characters. For docstrings, stay within
# 72 characters. See
# black: https://github.com/psf/black#pyprojecttoml
# PEP8:  https://www.python.org/dev/peps/pep-0008/#maximum-line-length
def combine_vars(
    df: pd.DataFrame, target_var_name: str, new_varname: str = "", inplace: bool = True
) -> Optional[pd.DataFrame]:
    """Combines variables in a DataFrame

    Target name is disassebled and checked if it can be created from the
    given :py:class:`pandas:pandas.DataFrame`. The variable is then create by
    calling :py:func:`pandas:pandas.eval`.

    Args:
        df: Object to pull data from and when ``inplace=True`` save result in
        target_var_name: Name to be parsed and created from ``df``
        new_varname: stuff
        inplace: Modify the passed ``df`` directly. Mirrors the
            :py:func:`pandas:pandas.eval` API.

    Returns:
        When ``inplace=True`` returns None, but the passed ``df`` object will
        contain the requested ``target_var_name`` variable.
        If ``inplace=False`` returns a copy of passed ``df`` object with
        the new variable.
    """
    if new_varname == "":
        new_varname = target_var_name
    parts = split_parts(target_var_name)
    coefs = extract_part_names(parts)
    ops = extract_operations(parts)
    for coef in coefs:
        if coef not in df:
            raise KeyError(coef)

    # Target variable names define left-to-right operations
    # Convert the "human names" of target_var_name to pandas eval expression
    # See https://pandas.pydata.org/docs/user_guide/enhancingperf.html?highlight=inplace#expression-evaluation-via-eval
    numexpr_ops = []
    for op, next_var in zip(ops, coefs[1:]):
        if op in _op_convert:
            numexpr_ops.append(_op_convert[op])
        else:
            raise NotImplementedError(f"Operation {op}")

    assert len(numexpr_ops) + 1 == len(
        coefs
    ), "Internal error when building pandas expression, open issue"
    # Combine in single expression. Add spaces for readability
    rest_exprs = map(
        " ".join, zip(numexpr_ops, coefs[1:])
    )  # Expression grouped like "op space coef"
    expr = f"{new_varname} = {coefs[0]} {' '.join(rest_exprs)}"
    df_new = df.eval(expr, inplace=inplace)
    return df_new


def separate_to_store(
    data,
    store,
    save_flux=True,
    save_growth=True,
    save_all=False,
    verbose=False,
    dropna=True,
    **put_kwargs,
):
    for col in data:
        key = "".join([sep_prefix, col])
        splitted = re.compile("(?=.*)(.)(|ITG|ETG|TEM)_(GB|SI|cm)").split(col)
        if (is_flux(col) and save_flux) or (is_growth(col) and save_growth) or save_all:
            if verbose:
                print("Saving", col)
            var = data[col]
            if dropna:
                var = var.dropna()
            var.to_hdf(store, key, format=store_format, **put_kwargs)
        else:
            if verbose:
                print("Do not save", col)


def save_to_store(input, data, const, store, style="both", compress=True, prefix="/"):
    kwargs = {}
    if isinstance(store, pd.HDFStore):
        store_name = store.filename
    else:
        store_name = store
    if compress is True:
        kwargs["complib"] = "zlib"
        kwargs["complevel"] = 1
        if not store_name.endswith(".1"):
            logger.warning(
                "Applying compression level 1, but store_name does not end with .1 as is convention"
            )
    if style == "sep" or style == "both":
        separate_to_store(data, store, save_all=True, **kwargs)
    if style == "flat" or style == "both":
        if len(data) > 0:
            data.to_hdf(store, prefix + "flattened", format=store_format, **kwargs)
        else:
            data.to_hdf(store, prefix + "flattened", format="fixed", **kwargs)

    input.to_hdf(store, prefix + "input", format=store_format, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        const.to_hdf(store, prefix + "constants", format="fixed")
    if isinstance(store, pd.HDFStore):
        store.close()


def return_all(columns):
    """ Check if all columns should be returned """
    return columns is None


def return_no(columns):
    """ Check if no columns should be returned """
    return columns is False


def determine_column_map(store: pd.HDFStore, prefix: str) -> Mapping[str, str]:
    """Map HDF5 keys to pretty shorthand names

    Args:
        store: Store to look in for columns
        prefix: Common prefix for all columns to be looked for

    Returns:
        A mapping with the HDF5 name as keys and the pretty names as values
    """
    # Associate 'nice' name with 'ugly' HDF5 node path
    name_dict = OrderedDict(
        (name, name.replace(prefix + sep_prefix, "", 1))
        for name in store.keys()
        if (("input" not in name) and ("constants" not in name) and ("flattened" not in name))
    )
    return name_dict


def filter_column_map(name_dict: Mapping, columns: Sequence[str]) -> Mapping[str, str]:
    """Filter out mapped columns not in store

    Args:
        name_dict: Map of HDF5 keys to pretty names
        columns: List of columns to search for

    Returns:
        A mapping with the HDF5 name as keys and the pretty names as values
    """
    if not return_all(columns) and not return_no(columns):
        req_columns = OrderedDict(
            (varname, name) for (varname, name) in name_dict.items() if name in columns
        )
    else:
        req_columns = name_dict
    return req_columns


@profile
def load_from_store(
    store_name=None,
    store=None,
    fast=True,
    mode="bare",
    how="left",
    columns=None,
    prefix="",
    load_input=True,
    load_const=True,
    nustar_to_lognustar=True,
    dask=False,
    verbosity_level=None,
):
    """Load a QLKNN-style dataset from HDF5 store

    Can load HDF5 stores since gen1 (2015), and thus contains many tricks
    Takes either a string path or pandas.HDFStore. Modern layout is a
    HDFStore with the ND hyperrectangle stored in the following groups:
    /input: A 2D array with the NN input, e.g. Ate, Ati, etc.
    /output/varname: 1D arrays with NN output 'varname', e.g. efeETG_GB, efi_GB
    /constants: The constants of the hypercube, e.g. relacc1, Te (if constant)
    /flattened: Optional: All outputs together in a 2D array for quick loading

    Kwargs:
      - store_name: String with path to HDFStore
      - store: HDFStore instance
      - fast: Enable fast mode; heuristically the fastest way to load from disk
      - mode: Mode of disk loading. Use non-fast methods to load from disk
              Inactive if 'fast' is True
      - how: How to join DataFrames in non-fast mode.
      - columns: Which columns to grab from HDFStore. Will be glued together
                 to a 2D array and returned. None for all columns.
      - prefix: Prefix to the HDF5 groups in the store
      - load_input: Also load and return the input from the HDFStore
      - load_const: Also load and return the constants from the HDFStore
      - nustar_to_lognustar: Convert \\nu^* to \\ln{\\nu^*} on the fly
      - dask: Use dask to do a parallel load from disk
      - verbosity_level: Verbosity of this function. Uses python logger

    Returns:
      - input: DataFrame with input in the HDFStore.
               Empty DataFrame if load_input=False
      - data: DataFrame with loaded columns from the HDFStore.
      - const: Series with constants from the HDFStore.
               Empty DataFrame if load_const=False
    """
    if verbosity_level is not None:
        logger.setLevel(verbosity_level)
    logger.debug("Reading user settings")
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, pd.Series):
        columns = columns.values
    if store_name is not None and store is not None:
        raise Exception("Specified both store and store name!")

    if dask and not has_dask:
        raise Exception("Requested dask, but dask import failed")

    if store is None:
        store = pd.HDFStore(store_name, "r")
    elif dask:
        raise ValueError("store cannot be passed if dask=True")
    logger.info("Loading from {!s} at {!s}".format(type(store), store.filename))

    def is_legacy(store):
        return all(["megarun" in name for name in store.keys()])

    if is_legacy(store):
        warnings.warn("Using legacy datafile!")
        prefix = "/megarun1/"

    def has_flattened(store):
        """ Check if store has a 'flattened' group """
        return any(["flattened" in group for group in store.keys()])

    def have_sep(columns):
        """ Check if store has separate groups for output variables """
        # TODO: Needs the columns in the df, now implicitly passed
        return columns is None or (len(req_columns) == len(columns))

    name_dict = determine_column_map(store, prefix)
    req_columns = filter_column_map(name_dict, columns)
    logger.debug("Detected the user wants to load %s", req_columns)

    # Now that we know what kind of dataset we are dealing with
    # and have guessed what the user wants, start
    # Loading input and constants
    logger.info("Load input from disk")
    if load_input:
        if dask:
            store.close()
            input = dd.read_hdf(store_name, prefix + "input")
        else:
            input = store[prefix + "input"]
            store.close()
        if nustar_to_lognustar and "Nustar" in input:
            input = convert_nustar(input)
    else:
        input = pd.DataFrame()
    logger.info("Input loaded, load constants from disk")
    if load_const:
        try:
            store.open("r")
            const = store[prefix + "constants"]
            store.close()
        except ValueError:
            # If pickled with a too new version
            # old python version cannot read it
            warnings.warn("Could not load const.. Skipping for now")
            const = pd.Series()
    else:
        const = pd.Series()

    if not return_no(columns):
        logger.info("Constants loaded, load output from disk")
        # Constants and input are loading, now load the brunt of data; the output
        store.open("r")
        # If we have a /flattened group or we do not have separately saved columns
        if has_flattened(store) and (return_all(columns) or not have_sep(columns)):
            if return_all(columns):
                # Take the "old" code path, this is fast if more columns than not
                # need to be loaded
                if dask:
                    data = dd.read_hdf(store_name, prefix + "flattened", chunksize=8192 * 10)
                else:
                    data = store.select(prefix + "flattened")
            else:
                # Partial load. Try to be smart, which pandas functions are not.
                # New code path is faster, but this is faster than pandas internals
                # when this was tested
                if dask:
                    data = dd.read_hdf(store_name, prefix + "flattened", columns=columns)
                else:
                    storer = store.get_storer(prefix + "flattened")
                    if storer.format_type == "fixed":
                        data = store.select(prefix + "flattened")
                        not_in_flattened = [col not in data.columns for col in columns]
                        if any(not_in_flattened):
                            raise Exception(
                                "Could not find {!s} in store {!s}".format(
                                    [
                                        col
                                        for not_in, col in zip(not_in_flattened, columns)
                                        if not_in
                                    ],
                                    store,
                                )
                            )
                        else:
                            print(
                                "Not implemented yet, but shouldn't happen anyway.. Contact Karel"
                            )
                            from IPython import embed

                            embed()
                    else:
                        data = store.select(prefix + "flattened", columns=columns)
        else:  # If no flattened
            # Taking "new" code path
            if not have_sep(columns):
                raise Exception("Could not find {!s} in store {!s}".format(columns, store))
            if dask:
                data = dd.read_hdf(store_name, "/output/*", columns=columns, chunksize=8192 * 10)
            elif fast:
                # Do a fast load as tested when this was written
                output = []
                for varname, name in req_columns.items():
                    var = store[varname]
                    var.name = name
                    output.append(var)
                data = pd.concat(output, axis=1)
                del output
            else:
                # You are on your own here. Good luck!
                if (mode != "update") and (mode != "bare"):
                    data = store[first(req_columns)[0]].to_frame()
                elif mode == "update":
                    df = store[first(req_columns)[0]]
                    data = pd.DataFrame(columns=req_columns.values(), index=df.index)
                    df.name = first(req_columns)[1]
                    data.update(df, raise_conflict=True)
                elif mode == "bare":
                    if not load_input:
                        raise Exception("Need to load input for mode {!s}".format(mode))
                    raw_data = np.empty([len(input), len(req_columns)])
                    ii = 0
                    varname = first(req_columns)[0]
                    df = store[varname]
                    if df.index.equals(input.index):
                        raw_data[:, ii] = df.values
                    else:
                        raise Exception("Nonmatching index on {!s}!".format(varname))
                for ii, (varname, name) in enumerate(req_columns.items()):
                    if ii == 0:
                        continue
                    if ("input" not in varname) and ("constants" not in varname):
                        if mode == "join":
                            data = data.join(store[varname], how=how)
                        elif mode == "concat":
                            data = pd.concat(
                                [data, store[varname]], axis=1, join="outer", copy=False
                            )
                        elif mode == "merge":
                            data = data.merge(
                                store[varname].to_frame(),
                                left_index=True,
                                right_index=True,
                                how=how,
                                copy=False,
                            )
                        elif mode == "assign":
                            data = data.assign(**{name: store[varname]})
                        elif mode == "update":
                            df = store[varname]
                            df.name = name
                            data.update(df, raise_conflict=True)
                        elif mode == "bare":
                            df = store[varname].reindex(index=input.index)
                            if df.index.equals(input.index):
                                raw_data[:, ii] = df.values
                            else:
                                raise Exception("Nonmatching index on {!s}!".format(varname))
                            del df
                    gc.collect()
                if mode == "bare":
                    data = pd.DataFrame(raw_data, columns=req_columns.values(), index=input.index)
    else:  # Don't return any data
        data = pd.DataFrame(index=input.index)
    logger.info("Output loaded.")
    store.close()
    gc.collect()
    return input, data, const
