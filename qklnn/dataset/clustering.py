import os
import sys
import re
import copy
import logging
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from matplotlib.pyplot import cm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Any

dask_available = True
try:
    import dask_ml.cluster
except ImportError:
    dask_available = False

from qlknn.dataset.data_io import load_from_store

root_logger = logging.getLogger("qlknn")
logger = root_logger
logger.setLevel(logging.INFO)


def k_expression_bart(density, npts, rfmin, tau, zeta=1.0):
    k_base = float(npts) / float(rfmin)
    k_adjustment = 1.0 - np.exp(-float(tau) * float(density))
    k_cluster = int(
        np.floor(
            k_base * (1.0 - float(zeta))
            + float(zeta) * (k_base * (1.0 - k_adjustment) + k_adjustment)
        )
    )
    if k_cluster < 1:
        k_cluster = 1
    return k_cluster


def rescale_data(dataframe):

    data = dataframe.to_numpy()

    ### Re-scale data to unit variance and zero mean
    scaler = StandardScaler()
    scaler.fit(data)
    data_sigma = scaler.scale_
    data_stdv = scaler.transform(data)
    data_mean = scaler.mean_
    data_scaled = np.zeros(data.shape)
    for i in range(0, data_scaled.shape[1]):
        # Seems unnecessarily complex but ok
        data_scaled[:, i] = 1 / (1 / data_sigma[i]) * data_stdv[:, i]

    return pd.DataFrame(data_scaled, index=dataframe.index, columns=dataframe.columns)


def run_dbscan(dataframe, eps=1.0, reduce_next=False, **kwargs):

    reduction_threshold = kwargs.get("reduction_threshold", None)
    min_samples = kwargs.get("min_samples", 1)
    min_cluster_size = kwargs.get("min_cluster_size", 2)

    data = dataframe.to_numpy()

    # NOTE: Data may be scaled, eps should be adjusted relative to data scale!
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    logger.info("DBSCAN step complete!")
    labels = db.labels_
    # Mask for detecting outliers (-1 in labels)
    outliers_samples_mask = labels == -1
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers_ = list(labels).count(-1)
    logger.info("Estimated number of clusters: %d" % n_clusters_)
    logger.info("Estimated number of outliers points: %d" % n_outliers_)

    indices = None
    if n_clusters_ != 0:
        # Cluster number stored in 'unique' and number of points stored in 'counts'
        unique, counts = np.unique(labels, return_counts=True)
        sorter = np.argsort(-counts)  # Sort by number of points in ascending order
        counts = counts[sorter]
        unique = unique[sorter]
        # Mask for DBSCAN outputs which are not outliers
        remove_outliers_mask = unique != -1
        # Mask for clusters tagged for reduction
        reduction_cluster_mask = counts > min_cluster_size
        # Mask for clusters which are larger than requested reduction threshold (RT)
        counts_mask = counts > reduction_threshold + 1

        # Outliers removed
        cluster_indices_toss = unique[~remove_outliers_mask]
        # Outliers and clusters with npts < RT removed
        cluster_indices_next = unique[reduction_cluster_mask & remove_outliers_mask & counts_mask]
        # Outliers and clusters with npts > RT and npts < min_size removed
        cluster_indices_reduce = unique[
            reduction_cluster_mask & remove_outliers_mask & ~counts_mask
        ]
        # Outliers and clusters with npts > min_size removed
        cluster_indices_keep = unique[~reduction_cluster_mask & remove_outliers_mask]

        indices = {
            "toss": dataframe.loc[np.isin(labels, cluster_indices_toss).tolist()].index.to_list(),
            "next": dataframe.loc[np.isin(labels, cluster_indices_next).tolist()].index.to_list(),
            "reduce": [],
            "keep": dataframe.loc[np.isin(labels, cluster_indices_keep).tolist()].index.to_list(),
        }
        for k in range(0, len(cluster_indices_reduce)):  # Loop over clusters
            cluster_mask = labels == cluster_indices_reduce[k]
            indices["reduce"].append(dataframe.loc[cluster_mask].index.to_list())
        if reduce_next:
            for k in range(0, len(cluster_indices_next)):
                cluster_mask = labels == cluster_indices_next[k]
                indices["reduce"].append(dataframe.loc[cluster_mask].index.to_list())

    return indices


def run_kmeans(dataframe, radius=1.0, **kwargs):

    num_dim = kwargs.get("num_dim", 1)
    min_reduction = kwargs.get("min_reduction", 1)
    tau = kwargs.get("tau", 1.0)
    zeta = kwargs.get("zeta", 1.0)
    dask_flag = kwargs.get("kmeans_dask", False)

    data = dataframe.to_numpy()
    index = dataframe.index.to_list()

    # NOTE: Data may be scaled, radius should be adjusted relative to data scale!
    neigh = NearestNeighbors(radius=radius)
    neigh.fit(data)
    distance_neighbours, index_neighbours = neigh.radius_neighbors()
    point_density = np.array([])
    for k in range(data.shape[0]):  # Loop over the points within a given cluster
        result = index_neighbours[k].shape[0] / (radius ** num_dim)
        point_density = np.append(point_density, result)
    number_points = data.shape[0]
    density_local = np.nanmax(point_density)
    k_cluster = k_expression_bart(density_local, number_points, min_reduction, tau, zeta=zeta)
    if not (dask_available and dask_flag):
        kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(data)
    else:
        kmeans = dask_ml.cluster.KMeans(n_clusters=k_cluster, random_state=0).fit(data)
    cluster_labels = kmeans.labels_
    cluster_center = kmeans.cluster_centers_
    # Data point closest to kmean centroid is flagged for keeping
    closest, _ = pairwise_distances_argmin_min(cluster_center, data)
    result_index = dataframe.iloc[closest].index
    index_of_closest = int(result_index[0]) if len(result_index) > 0 else None

    return index_of_closest


def two_step_clustering(data, **settings):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("DataFrame expected as input! Aborting!")

    ### Extract required settings from settings list
    num_dim = settings.get("num_dim", 1)
    reduction_threshold = settings.get("reduction_threshold", None)
    initial_eps = settings.get("initial_eps", 1.0)  # Initial value for scaling eps
    min_eps = settings.get("min_eps", None)  # Minimum value for scaling eps
    verbose = settings.get("verbose", 0)
    fdebug = settings.get("debug_mode", False)

    ### Determine the maximum density resolvable based on inputs and settings
    if verbose > 0:
        logger.info("Dataset size: %d" % (len(data)))
        density_max = reduction_threshold / (min_eps ** num_dim)
        logger.info("Maximum density resolvable: %f / %d-D unit volume" % (density_max, num_dim))

    logger.info("Start scaling data")
    data_scaled = rescale_data(data)

    ### Initial DBSCAN, used to remove outliers => unclustered points in first DBSCAN iteration
    logger.info("Start initial DBSCAN-kmeans run")
    index_vectors = run_dbscan(data_scaled, eps=1.0, **settings)
    if index_vectors is None:
        logger.info("No clusters discovered in initial pass, reconfigure settings.")
        return
    # Data to keep within final reduced list - only the initial pass throws away the noise
    index_keep = index_vectors["keep"]
    if verbose > 0:
        logger.info("  %d points passed to next iteration." % (len(index_vectors["next"])))
        logger.info("  %d points identified as noise (thrown)." % (len(index_vectors["toss"])))
    if verbose > 1:
        logger.info("  %d points inserted directly in final set." % (len(index_vectors["keep"])))
    # Apply kmeans to clusters tagged for reduction as is
    if len(index_vectors["reduce"]) != 0:
        if verbose > 1:
            logger.info("  %d clusters identified." % (len(index_vectors["reduce"])))
            np_clusters = 0
            for j in range(0, len(index_vectors["reduce"])):
                np_clusters += len(index_vectors["reduce"][j])
            logger.info("  %d points to be reduced in this iteration." % (np_clusters))
        for k in range(0, len(index_vectors["reduce"])):  # Loop over clusters
            if verbose > 2:
                logger.info("    Cluster %5d: %d points." % (k, len(index_vectors["reduce"][k])))
            data_cluster = data_scaled.loc[index_vectors["reduce"][k]]
            index_scalar = run_kmeans(data_cluster, radius=initial_eps, **settings)
            index_keep.append(index_scalar)
    # Keep track of number of points in final dataset selected by this iteration
    iteration_counter = np.array([len(index_keep)])
    # Data to pass onto next iteration
    data_scaled = data_scaled.loc[index_vectors["next"]]

    logger.info("Start DBSCAN iterations")
    scaling_eps = initial_eps
    eps_counter = np.array([scaling_eps])
    ### TODO: This should be changed to a while loop
    max_iter = 10000 if not fdebug else 1
    for i in range(0, max_iter):
        # Perform secondary DBSCAN on the scaled data after removing outliers
        logger.info("DBSCAN iteration number: %d" % (i + 1))
        index_vectors = run_dbscan(data_scaled, eps=scaling_eps, **settings)
        # Exit loop if all clusters tagged for reduction have been reduced
        if index_vectors is None:
            logger.info("No more clusters found by DBSCAN")
            break

        # Data to keep within final reduced list - all loop iterations keep outliers
        index_keep.extend(index_vectors["toss"])
        index_keep.extend(index_vectors["keep"])
        if verbose > 0:
            logger.info("  %d points passed to next iteration." % (len(index_vectors["next"])))
        if verbose > 1:
            logger.info("  %d points identified as noise (kept)." % (len(index_vectors["toss"])))
            logger.info(
                "  %d points inserted directly in final set."
                % (len(index_vectors["toss"]) + len(index_vectors["keep"]))
            )
        # Apply kmeans to clusters tagged for reduction as is
        if len(index_vectors["reduce"]) != 0:
            if verbose > 1:
                logger.info("  %d clusters identified." % (len(index_vectors["reduce"])))
                np_clusters = 0
                for j in range(0, len(index_vectors["reduce"])):
                    np_clusters += len(index_vectors["reduce"][j])
                logger.info("  %d points to be reduced in this iteration." % (np_clusters))
            for k in range(0, len(index_vectors["reduce"])):  # Loop over clusters
                if verbose > 2:
                    logger.info(
                        "    Cluster %5d: %d points." % (k, len(index_vectors["reduce"][k]))
                    )
                data_cluster = data_scaled.loc[index_vectors["reduce"][k]]
                index_scalar = run_kmeans(data_cluster, radius=scaling_eps, **settings)
                index_keep.append(index_scalar)
        # Keep track of number of points in final dataset selected by this iteration
        iteration_counter = np.hstack((iteration_counter, len(index_keep)))
        # Data to pass onto next iteration
        data_scaled = (
            data_scaled.loc[index_vectors["next"]]
            if len(index_vectors["next"]) != 0
            else pd.DataFrame()
        )
        logger.info("Number of points to pass forward: %d" % (len(index_vectors["next"])))

        if scaling_eps < min_eps:
            logger.info("Minimum radius is reached, stopping iterations")
            break
        if len(data_scaled) == 0:
            logger.info("No more large clusters to reduce, stopping iterations")
            break

        # Reducing the radius vector for each iteration
        scaling_eps = scaling_eps - 0.05
        eps_counter = np.hstack((eps_counter, scaling_eps))  # Saving the new radius
        logger.info("New eps: %f." % (scaling_eps))

    # One final pass is done to reduce remaining data, in case of minimum radius exit condition
    if len(data_scaled) != 0:
        index_vectors = run_dbscan(data_scaled, eps=scaling_eps, reduce_next=True, **settings)
        if verbose > 3:
            # These should be zero if printed
            logger.info("  %d points identified as noise (kept)." % (len(index_vectors["toss"])))
            logger.info(
                "  %d points inserted directly in final set."
                % (len(index_vectors["toss"]) + len(index_vectors["keep"]))
            )
        if verbose > 2:
            logger.info("  %d clusters identified." % (len(index_vectors["reduce"])))
        if verbose > 0:
            np_clusters = 0
            for j in range(0, len(index_vectors["reduce"])):
                np_clusters += len(index_vectors["reduce"][j])
            logger.info("  %d points to be reduced in the final iteration." % (np_clusters))
        for k in range(0, len(index_vectors["reduce"])):  # Loop over clusters
            if verbose > 0:
                logger.info("    Cluster %5d: %d points." % (k, len(index_vectors["reduce"][k])))
            data_cluster = data_scaled.loc[index_vectors["reduce"][k]]
            index_scalar = run_kmeans(data_cluster, radius=scaling_eps, **settings)
            index_keep.append(index_scalar)
        iteration_counter = np.hstack((iteration_counter, len(index_keep)))

    data_final = data.loc[index_keep]  # Output data is not ordered identically to input data
    logger.info("Final number of points per dimension: %d" % data_final.shape[0])
    # logger.info("size of the iteration counter: %d" % iteration_counter.shape[0])

    return data_final


def run_clustering(settingspath, verbose=0, fdebug=False):

    spath = Path(settingspath)
    if not spath.is_file():
        raise IOError("Settings file %s was not found! Aborting!" % str(spath))

    ### Load settings
    logger.info("Reading JSON settings file")
    settings = dict()
    with open(str(spath.resolve())) as file_:
        settings = json.load(file_)
    settings["verbose"] = verbose
    settings["debug_mode"] = fdebug
    if verbose > 1 or fdebug:
        logger.info(repr(settings))
        logger.info("Dask available in system: %s" % repr(dask_available))

    opath = Path(settings.get("output_name", None))
    if opath.exists() and not opath.is_file():
        raise IOError(
            "Output file %s already exists and cannot be overwritten! Aborting!"
            % (str(opath.resolve()))
        )
    dpath = Path(settings.get("dataset_path", None))
    if not dpath.is_file():
        raise IOError("Data file %s was not found! Aborting!" % str(dpath))

    ### Load data
    logger.info("Reading input data file")
    dims = settings.pop("dims", [])
    input_list = []
    data_input = None
    if ".h5" in dpath.suffixes:
        (data_input, f_outp, f_const) = load_from_store(str(dpath.resolve()))
        drop_list = []
        output_list = []
        for var in data_input.columns:
            if var not in dims:
                drop_list.append(var)
            else:
                input_list.append(var)
        for var in f_outp.columns:
            if var in dims:
                output_list.append(var)
        if len(drop_list) > 0:
            data_input = data_input.drop(drop_list, axis=1)
        if len(output_list) > 0:
            ofilt = np.all(np.isfinite(f_outp[output_list]), axis=1)
            t_outp = f_outp.loc[ofilt]
            ifilt = data_input.index.isin(t_outp.index)
            data_input = data_input.loc[ifilt]
            for var in output_list:
                data_input[var] = t_outp[var]
        if "num_dim" not in settings:
            settings["num_dim"] = len(data_input.columns)
    elif ".nc" in dpath.suffixes:  # Not expected to ever start from netCDF file
        ds = xr.open_dataset(str(dpath.resolve()))
        coord_list = []
        drop_list = []
        for var in ds.coords:
            coord_list.append(var)
            if var not in dims:
                drop_list.append(var)
            else:
                input_list.append(var)
        for var in ds.variables:
            if var not in dims:
                drop_list.append(var)
        if len(coord_list) > 0:
            ds = ds.reset_coords(coord_list)
        if len(drop_list) > 0:
            ds = ds.drop_vars(drop_list)
        data_input = ds.to_dataframe()
        if "num_dim" not in settings:
            settings["num_dim"] = len(data_input.columns)
    else:
        logger.warning("This input format is not supported!")

    data_output = two_step_clustering(data_input, **settings)

    # logger.info("Start saving data to pickle")
    # df = pd.DataFrame(data_output, columns=settings['dims'])
    # df['iteration_counter'] = iteration_counter
    # pickle_name = settings.get("pickle_name", None)
    # df.to_pickle(pickle_name)
    # df['iteration_counter'] = df['iteration_counter'].astype(np.int64)

    logger.info("Saving data to HDF5 file")
    data_output[input_list].to_hdf(str(opath.resolve()), "input", format="table")
    for name in data_output.columns:
        if name not in input_list:
            data_output[name].to_hdf(str(opath.resolve()), "output/" + name, format="table")
    store = pd.HDFStore(str(opath.resolve()))
    store["constants"] = pd.DataFrame()
    store.close()

    logger.info("DBSCAN-kmeans reduction algorithm completed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch database clustering algorithm")
    parser.add_argument("--settings", default=None, type=str, help="Optional settings JSON file")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--debug", default=False, action="store_true", help="Toggle debug mode")
    args = parser.parse_args()
    run_clustering(args.settings, verbose=args.verbose, fdebug=args.debug)


if __name__ == "__main__":
    main()
