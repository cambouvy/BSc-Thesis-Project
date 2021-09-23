from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os
import io
import subprocess
import json
from itertools import product
from shutil import copyfile
from collections import OrderedDict
import signal

if not (sys.version_info > (3, 0)):
    range = xrange

import argparse
import tensorflow as tf
from tensorflow.contrib import opt
from tensorflow.python.client import timeline
import numpy as np
import pandas as pd
from IPython import embed


from qlknn.misc.tools import profile
from qlknn.training.datasets import (
    Dataset,
    Datasets,
    convert_panda,
    split_panda,
    shuffle_panda,
)
from qlknn.training.nn_primitives import (
    model_to_json,
    weight_variable,
    bias_variable,
    variable_summaries,
    nn_layer,
    normab,
    normsm,
    descale_panda,
    scale_panda,
)
from qlknn.training.profiling import TimeLiner
from qlknn.dataset.data_io import load_from_store
from qlknn.misc.to_precision import to_precision
from qlknn.misc.tools import ordered_dict_prepend

FLAGS = None


def timediff(start, event):
    print("{:35} {:5.0f}s".format(event + " after", time.time() - start))


def print_last_row(df, header=False):
    print(
        df.iloc[[-1]].to_string(
            header=header,
            float_format=lambda x: "{:.2f}".format(x),
            col_space=12,
            justify="left",
        )
    )


def timeout(signum, frame):
    global stopping_timeout
    print("Received signal: {!s} frame: {!s}".format(signum, frame))
    # mutating a mutable object doesn't require changing what the variable name points to
    # Python2/3 compat (ugly) workaround
    stopping_timeout[0] = True


@profile
def drop_outliers(target_df, settings):
    if settings["drop_outlier_above"] < 1:
        target_df = target_df[target_df < target_df.quantile(settings["drop_outlier_above"])]
    if settings["drop_outlier_below"] > 0:
        target_df = target_df[target_df > target_df.quantile(settings["drop_outlier_below"])]
    return target_df


@profile
def drop_nans(target_df):
    # Remove NaNs
    target_df.dropna(axis=0, inplace=True)
    return target_df


@profile
def filter_input(input_df, target_df):
    # Use only samples in the feature set that are in the target set
    # input_df = input_df.reindex(target_df.index, copy=False)
    data_df = pd.concat((input_df, target_df), join="inner", copy=False, axis=1)
    return data_df


@profile
def convert_dtype(data_df, settings):
    # Convert to dtype in settings file. Usually float32 or float64
    data_df = data_df.astype(settings["dtype"])
    return data_df


def prep_dataset(settings):
    train_dims = settings["train_dims"]
    # Open HDF store. This is usually a soft link to our filtered dataset
    input_df, target_df, const = load_from_store(settings["dataset_path"], columns=train_dims)

    try:
        del input_df["nions"]  # Delete leftover artifact from dataset split
    except KeyError:
        pass

    target_df = drop_outliers(target_df, settings)
    target_df = drop_nans(target_df)

    data_df = filter_input(input_df, target_df)
    del input_df, target_df
    data_df = convert_dtype(data_df, settings)

    return data_df


@profile
def calc_standardization(data_df, settings, warm_start_nn=None):
    if warm_start_nn is None:
        if settings["standardization"].startswith("minmax"):
            min = float(settings["standardization"].split("_")[-2])
            max = float(settings["standardization"].split("_")[-1])
            scale_factor, scale_bias = normab(data_df, min, max)

        if settings["standardization"].startswith("normsm"):
            s_t = float(settings["standardization"].split("_")[-2])
            m_t = float(settings["standardization"].split("_")[-1])
            scale_factor, scale_bias = normsm(data_df, s_t, m_t)
    else:
        scale_factor = pd.concat(
            [
                warm_start_nn._feature_prescale_factor,
                warm_start_nn._target_prescale_factor,
            ]
        )
        scale_bias = pd.concat(
            [warm_start_nn._feature_prescale_bias, warm_start_nn._target_prescale_bias]
        )

    return scale_factor, scale_bias


class QLKNet:
    def __init__(self, x, num_target_dims, settings, debug=False, warm_start_nn=None):
        self.x = x
        self.NUM_TARGET_DIMS = num_target_dims
        self.SETTINGS = settings
        self.DEBUG = debug
        self.WARM_START_NN = warm_start_nn
        self.create()

    def create(self):
        x = self.x
        settings = self.SETTINGS
        debug = self.DEBUG
        warm_start_nn = self.WARM_START_NN
        num_target_dims = self.NUM_TARGET_DIMS

        layers = [x]
        if settings["drop_chance"] != 0:
            drop_prob = tf.constant(settings["drop_chance"], dtype=x.dtype)
        self.is_train = tf.placeholder(tf.bool)
        for ii, (activation, neurons) in enumerate(
            zip(settings["hidden_activation"], settings["hidden_neurons"]), start=1
        ):
            if warm_start_nn is None:
                weight_init = settings["weight_init"]
                bias_init = settings["bias_init"]
            else:
                if (
                    warm_start_nn.layers[ii - 1]._activation == activation
                    and warm_start_nn.layers[ii - 1]._weights.shape[1] == neurons
                ):
                    weight_init = warm_start_nn.layers[ii - 1]._weights
                    bias_init = warm_start_nn.layers[ii - 1]._biases
                    activation = warm_start_nn.layers[ii - 1]._activation
                else:
                    raise Exception("Settings file layer shape does not match warm_start_nn")

            if activation == "tanh":
                act = tf.tanh
            elif activation == "relu":
                act = tf.nn.relu
            elif activation == "none":
                act = None

            layer = nn_layer(
                layers[-1],
                neurons,
                "layer" + str(ii),
                dtype=x.dtype,
                act=act,
                debug=debug,
                bias_init=bias_init,
                weight_init=weight_init,
            )
            if settings["drop_chance"] != 0:
                dropout = tf.layers.dropout(layer, drop_prob, training=self.is_train)
                if debug:
                    tf.summary.histogram("post_dropout_layer_" + str(ii), dropout)
                layers.append(dropout)
            else:
                layers.append(layer)

        # Last layer (output layer) usually has no activation
        activation = settings["output_activation"]
        if warm_start_nn is None:
            weight_init = bias_init = settings["standardization"]
        else:
            weight_init = warm_start_nn.layers[-1]._weights
            bias_init = warm_start_nn.layers[-1]._biases
            activation = warm_start_nn.layers[-1]._activation

        if activation == "tanh":
            act = tf.tanh
        elif activation == "relu":
            act = tf.nn.relu
        elif activation == "none":
            act = None
        self.y = nn_layer(
            layers[-1],
            num_target_dims,
            "layer" + str(len(layers)),
            dtype=x.dtype,
            act=act,
            debug=debug,
            bias_init=bias_init,
            weight_init=weight_init,
        )


def train(settings, warm_start_nn=None, restore_old_checkpoint=False):
    tf.reset_default_graph()
    start = time.time()

    data_df = prep_dataset(settings)
    target_names = settings["train_dims"]
    feature_names = list(data_df.columns)
    for dim in target_names:
        feature_names.remove(dim)

    if settings["calc_standardization_on_nonzero"]:
        any_nonzero = (data_df[target_names] != 0).any(axis=1)
        data_df_nonzero = data_df.loc[any_nonzero, :]
        data_df_zero = data_df.loc[~any_nonzero, :]
        scale_factor, scale_bias = calc_standardization(
            data_df_nonzero, settings, warm_start_nn=warm_start_nn
        )
        data_df = scale_panda(data_df, scale_factor, scale_bias)
    else:
        scale_factor, scale_bias = calc_standardization(
            data_df, settings, warm_start_nn=warm_start_nn
        )
        data_df = scale_panda(data_df, scale_factor, scale_bias)

    # Standardize input
    timediff(start, "Scaling defined")

    datasets = convert_panda(
        data_df,
        feature_names,
        target_names,
        settings["validation_fraction"],
        settings["test_fraction"],
    )

    # Start tensorflow session
    if "NUM_INTER_THREADS" in os.environ:
        NUM_INTER_THREADS = int(os.environ["NUM_INTER_THREADS"])
    else:
        NUM_INTER_THREADS = None
    if "NUM_INTRA_THREADS" in os.environ:
        NUM_INTRA_THREADS = int(os.environ["NUM_INTRA_THREADS"])
    else:
        NUM_INTRA_THREADS = None
    config = tf.ConfigProto(
        inter_op_parallelism_threads=NUM_INTER_THREADS,
        intra_op_parallelism_threads=NUM_INTRA_THREADS,
    )
    # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
    #                    allow_soft_placement=True, device_count = {'CPU': 1})
    sess = tf.Session(config=config)

    # Input placeholders
    with tf.name_scope("input"):
        x = tf.placeholder(
            datasets.train._target.dtype, [None, len(feature_names)], name="x-input"
        )
        y_ds = tf.placeholder(x.dtype, [None, len(target_names)], name="y-input")

    net = QLKNet(x, len(target_names), settings, warm_start_nn=warm_start_nn)
    y = net.y
    y_descale = (net.y - scale_bias[target_names].values) / scale_factor[target_names].values
    y_ds_descale = (y_ds - scale_bias[target_names].values) / scale_factor[target_names].values
    is_train = net.is_train

    timediff(start, "NN defined")
    if settings["goodness_only_on_unstable"] is True:
        orig_is_stable = tf.less_equal(y_ds_descale, 0)

    # Define loss functions
    with tf.name_scope("Loss"):
        with tf.name_scope("mse"):
            if settings["goodness_only_on_unstable"]:
                mse = tf.losses.mean_squared_error(
                    y_ds, y, weights=tf.logical_not(orig_is_stable)
                )
                mse_descale = tf.losses.mean_squared_error(
                    y_ds_descale, y_descale, weights=tf.logical_not(orig_is_stable)
                )
            else:
                mse = tf.losses.mean_squared_error(y_ds, y)
                mse_descale = tf.losses.mean_squared_error(y_ds_descale, y_descale)
            tf.summary.scalar("MSE", mse)
        with tf.name_scope("mabse"):
            if settings["goodness_only_on_unstable"]:
                mabse = tf.losses.absolute_difference(
                    y_ds, y, weights=tf.logical_not(orig_is_stable)
                )
            else:
                mabse = tf.losses.absolute_difference(y_ds, y)
            tf.summary.scalar("MABSE", mabse)
        with tf.name_scope("l2"):
            l2_scale = tf.Variable(settings["cost_l2_scale"], dtype=x.dtype, trainable=False)
            # l2_norm = tf.reduce_sum(tf.square())
            # l2_norm = tf.to_double(tf.add_n([tf.nn.l2_loss(var)
            #                        for var in tf.trainable_variables()
            #                        if 'weights' in var.name]))
            l2_norm = tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables() if "weights" in var.name]
            )
            # mse = tf.losses.mean_squared_error(y_, y)
            # TODO: Check normalization
            l2_loss = l2_scale * l2_norm
            # tf.summary.scalar('l2_norm', l2_norm)
            # tf.summary.scalar('l2_scale', l2_scale)
            tf.summary.scalar("l2_loss", l2_loss)
        with tf.name_scope("l1"):
            l1_scale = tf.Variable(settings["cost_l1_scale"], dtype=x.dtype, trainable=False)
            # l1_norm = tf.to_double(tf.add_n([tf.reduce_sum(tf.abs(var))
            #                        for var in tf.trainable_variables()
            #                        if 'weights' in var.name]))
            l1_norm = tf.add_n(
                [
                    tf.reduce_sum(tf.abs(var))
                    for var in tf.trainable_variables()
                    if "weights" in var.name
                ]
            )
            # TODO: Check normalization
            l1_loss = l1_scale * l1_norm
            # tf.summary.scalar('l1_norm', l1_norm)
            # tf.summary.scalar('l1_scale', l1_scale)
            tf.summary.scalar("l1_loss", l1_loss)
        with tf.name_scope("stable_positive"):
            stable_positive_scale = tf.Variable(
                settings["cost_stable_positive_scale"], dtype=x.dtype, trainable=False
            )
            stable_positive_offset = tf.Variable(
                settings["cost_stable_positive_offset"], dtype=x.dtype, trainable=False
            )
            nn_pred_above_offset = tf.greater(y, settings["cost_stable_positive_offset"])
            punish_unstable_pred = tf.logical_and(orig_is_stable, nn_pred_above_offset)
            if settings["cost_stable_positive_function"] == "block":
                stable_positive_loss = tf.reduce_mean(
                    stable_positive_scale
                    * (y - stable_positive_offset)
                    * tf.cast(punish_unstable_pred, x.dtype)
                )
            elif settings["cost_stable_positive_function"] == "barrier":
                stable_positive_loss = tf.reduce_mean(
                    tf.cast(orig_is_stable, x.dtype)
                    * tf.exp(stable_positive_scale * (y - stable_positive_offset))
                )
            tf.summary.scalar("stable_positive_loss", stable_positive_loss)

        if settings["goodness"] == "mse":
            goodness_loss = mse
        elif settings["goodness"] == "mabse":
            goodness_loss = mabse
        loss = goodness_loss
        if settings["cost_stable_positive_scale"] != 0:
            loss += stable_positive_loss
            stable_positive_loss_importance = stable_positive_loss / goodness_loss
        if settings["cost_l1_scale"] != 0:
            loss += l1_loss
            rel_l1_loss_importance = l1_loss / goodness_loss
        if settings["cost_l2_scale"] != 0:
            loss += l2_loss
            rel_l2_loss_importance = l2_loss / goodness_loss
        tf.summary.scalar("total loss", loss)

    with tf.name_scope("goodness"):
        tf.summary.scalar("RMS descaled", tf.sqrt(mse_descale))

    with tf.name_scope("importance"):
        try:
            tf.summary.scalar("rel_l2_loss_importance", rel_l2_loss_importance)
        except:
            pass
        try:
            tf.summary.scalar("rel_l1_loss_importance", rel_l1_loss_importance)
        except:
            pass
        try:
            tf.summary.scalar("stable_positive_loss_importance", stable_positive_loss_importance)
        except:
            pass

    optimizer = None
    train_step = None
    # Define optimizer algorithm.
    with tf.name_scope("train"):
        lr = settings["learning_rate"]
        if settings["optimizer"] == "adam":
            beta1 = settings["adam_beta1"]
            beta2 = settings["adam_beta2"]
            train_step = tf.train.AdamOptimizer(
                lr,
                beta1,
                beta2,
            ).minimize(loss)
        elif settings["optimizer"] == "adadelta":
            rho = settings["adadelta_rho"]
            train_step = tf.train.AdadeltaOptimizer(
                lr,
                rho,
            ).minimize(loss)
        elif settings["optimizer"] == "rmsprop":
            decay = settings["rmsprop_decay"]
            momentum = settings["rmsprop_momentum"]
            train_step = tf.train.RMSPropOptimizer(lr, decay, momentum).minimize(loss)
        elif settings["optimizer"] == "grad":
            train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        elif settings["optimizer"] == "lbfgs":
            optimizer = opt.ScipyOptimizerInterface(
                loss,
                options={
                    "maxiter": settings["lbfgs_maxiter"],
                    "maxfun": settings["lbfgs_maxfun"],
                    "maxls": settings["lbfgs_maxls"],
                },
            )
        # tf.logging.set_verbosity(tf.logging.INFO)

    cur_train_time = tf.Variable(0.0, name="cur_train_time", dtype="float64", trainable=False)
    global_step = tf.Variable(0, name="global_step", dtype="int64", trainable=False)
    epoch = tf.Variable(0, name="epoch", dtype="int64", trainable=False)
    # Merge all the summaries
    merged = tf.summary.merge_all()

    # Initialze writers, variables and logdir
    log_dir = "tf_logs"
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    validation_writer = tf.summary.FileWriter(log_dir + "/validation", sess.graph)
    tf.global_variables_initializer().run(session=sess)
    timediff(start, "Variables initialized")

    # Save checkpoints of training to restore for early-stopping
    saver = tf.train.Saver(max_to_keep=settings["early_stop_after"] + 1)
    checkpoint_dir = "checkpoints"
    if restore_old_checkpoint:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        tf.gfile.MkDir(checkpoint_dir)

    # Define log files
    train_log = pd.DataFrame(
        columns=[
            "epoch",
            "walltime",
            "loss",
            "mse",
            "mabse",
            "l1_norm",
            "l2_norm",
            "stable_positive_loss",
        ]
    )
    validation_log = pd.DataFrame(
        columns=[
            "epoch",
            "walltime",
            "loss",
            "mse",
            "mabse",
            "l1_norm",
            "l2_norm",
            "stable_positive_loss",
        ]
    )

    # Split dataset in minibatches
    minibatches = settings["minibatches"]
    batch_size = int(np.floor(datasets.train.num_examples / minibatches))

    timediff(start, "Starting loss calculation")
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    summary, lo, meanse, meanabse, l1norm, l2norm, stab_pos = sess.run(
        [merged, loss, mse, mabse, l1_norm, l2_norm, stable_positive_loss],
        feed_dict=feed_dict,
    )
    train_log.loc[0] = (
        epoch.eval(session=sess),
        0,
        lo,
        meanse,
        meanabse,
        l1norm,
        l2norm,
        stab_pos,
    )
    validation_log.loc[0] = (
        epoch.eval(session=sess),
        0,
        lo,
        meanse,
        meanabse,
        l1norm,
        l2norm,
        stab_pos,
    )

    # Define variables for early stopping
    not_improved = 0
    best_early_measure = np.inf
    early_measure = np.inf

    max_epoch = settings.get("max_epoch") or sys.maxsize

    # Set debugging parameters
    setting = lambda x, default: default if x is None else x
    steps_per_report = setting(settings.get("steps_per_report"), np.inf)
    epochs_per_report = setting(settings.get("epochs_per_report"), np.inf)
    save_checkpoint_networks = setting(settings.get("save_checkpoint_networks"), False)
    save_best_networks = setting(settings.get("save_best_networks"), False)
    track_training_time = setting(settings.get("track_training_time"), False)
    full_timeline = TimeLiner()

    # Set up log files
    train_log_file = open("train_log.csv", "a", 1)
    train_log_file.truncate(0)
    colwidth = max(len(name) for name in validation_log.columns[:-1])
    colwidth = 12
    ffmt = "% -{width}.3f".format(width=colwidth)
    header = ["{:<{width}}".format(colname, width=colwidth) for colname in train_log.columns]
    train_log.to_csv(train_log_file, float_format=ffmt, header=header)
    validation_log_file = open("validation_log.csv", "a", 1)
    validation_log_file.truncate(0)
    validation_log.to_csv(validation_log_file, float_format=ffmt, header=header)

    # Set flag to stop training because of timeout
    global stopping_timeout
    stopping_timeout = [False]

    timediff(start, "Training started")
    train_start = time.time()
    prev_train_time = cur_train_time.eval(session=sess)
    prev_epochs = epoch.eval(session=sess)
    stop_reason = "undefined"
    signal.signal(signal.SIGUSR1, timeout)
    final_nn_file = "nn.json"
    try:
        for epoch_sess in range(max_epoch):
            epoch.load(prev_epochs + epoch_sess, sess)
            epoch_idx = epoch.eval(session=sess)
            for step in range(minibatches):
                # Extra debugging every steps_per_report
                if not step % steps_per_report and steps_per_report != np.inf:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                xs, ys = datasets.train.next_batch(batch_size, shuffle=True)
                feed_dict = {x: xs, y_ds: ys, is_train: True}
                # If we have a scipy-style optimizer
                if optimizer:
                    # optimizer.minimize(sess, feed_dict=feed_dict)
                    optimizer.minimize(
                        sess,
                        feed_dict=feed_dict,
                        #                   options=run_options,
                        #                   run_metadata=run_metadata)
                    )
                    lo = loss.eval(feed_dict=feed_dict)
                    meanse = mse.eval(feed_dict=feed_dict)
                    meanabse = mabse.eval(feed_dict=feed_dict)
                    l1norm = l1_norm.eval(feed_dict=feed_dict)
                    l2norm = l2_norm.eval(feed_dict=feed_dict)
                    summary = merged.eval(feed_dict=feed_dict)
                else:  # If we have a TensorFlow-style optimizer
                    (summary, lo, meanse, meanabse, l1norm, l2norm, stab_pos, _,) = sess.run(
                        [
                            merged,
                            loss,
                            mse,
                            mabse,
                            l1_norm,
                            l2_norm,
                            stable_positive_loss,
                            train_step,
                        ],
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata,
                    )
                train_writer.add_summary(summary, global_step.eval(session=sess))

                # Extra debugging every steps_per_report
                if not step % steps_per_report and steps_per_report != np.inf:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    # with open('timeline_run.json', 'w') as f:
                    #    f.write(ctf)
                    full_timeline.update_timeline(ctf)

                    train_writer.add_run_metadata(
                        run_metadata, "epoch%d step%d" % (epoch_idx, step)
                    )
                # Add to CSV log buffer
                cur_train_time.load(time.time() - train_start + prev_train_time, sess)
                if track_training_time:
                    idx = epoch_idx * minibatches + step
                    train_log.loc[idx] = (
                        epoch_idx,
                        cur_train_time.eval(session=sess),
                        lo,
                        meanse,
                        meanabse,
                        l1norm,
                        l2norm,
                        stab_pos,
                    )

                global_step.load(global_step.eval(session=sess) + 1, sess)
                # Stop on timeout (USR1)
                if stopping_timeout[0]:
                    break

            # Stop on timeout (USR1)
            if stopping_timeout[0]:
                print("stopping timeout..")
                stop_reason = "timeout"
                final_nn_file = "nn_timeout.json"
                break

            ########
            # After-epoch stuff
            ########

            if track_training_time is True:
                step_start = time.time()
            assert epoch_sess == datasets.train.epochs_completed
            xs, ys = datasets.validation.next_batch(-1, shuffle=False)
            feed_dict = {x: xs, y_ds: ys, is_train: False}
            # Run with full trace every epochs_per_report Gives full runtime information
            if not epoch_idx % epochs_per_report and epochs_per_report != np.inf:
                print("epoch_debug!", epoch_idx)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            # Calculate all variables with the validation set
            summary, lo, meanse, meanabse, l1norm, l2norm, stab_pos = sess.run(
                [merged, loss, mse, mabse, l1_norm, l2_norm, stable_positive_loss],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata,
            )

            validation_writer.add_summary(summary, global_step.eval(session=sess))
            # More debugging every epochs_per_report
            if not epoch_idx % epochs_per_report and epochs_per_report != np.inf:
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                full_timeline.update_timeline(ctf)
                # with open('timeline.json', 'w') as f:
                #    f.write(ctf)

                validation_writer.add_run_metadata(run_metadata, "epoch%d" % epoch_idx)

            # Save checkpoint
            save_path = saver.save(
                sess,
                os.path.join(checkpoint_dir, "model.ckpt"),
                global_step=global_step,
                write_meta_graph=False,
            )

            # Update CSV logs
            cur_train_time.load(time.time() - train_start + prev_train_time, session=sess)
            if track_training_time:
                validation_log.loc[epoch_idx] = (
                    epoch_idx,
                    cur_train_time.eval(session=sess),
                    lo,
                    meanse,
                    meanabse,
                    l1norm,
                    l2norm,
                    stab_pos,
                )

                validation_log.loc[epoch_idx:].to_csv(
                    validation_log_file, header=False, float_format=ffmt
                )
                validation_log = validation_log[0:0]  # Flush validation log
                train_log.loc[epoch_idx * minibatches :].to_csv(
                    train_log_file, header=False, float_format=ffmt
                )
                train_log = train_log[0:0]  # Flush train_log

            # Determine early-stopping criterion
            if settings["early_stop_measure"] == "mse":
                early_measure = meanse
            elif settings["early_stop_measure"] == "loss":
                early_measure = lo
            elif settings["early_stop_measure"] == "none":
                early_measure = np.nan

            # Early stopping, check if measure is better
            if early_measure < best_early_measure:
                best_early_measure = early_measure
                if save_best_networks:
                    nn_best_file = os.path.join(
                        checkpoint_dir, "nn_checkpoint_" + str(epoch_idx) + ".json"
                    )
                    trainable = {
                        x.name: tf.to_double(x).eval(session=sess).tolist()
                        for x in tf.trainable_variables()
                    }
                    model_to_json(
                        nn_best_file,
                        trainable=trainable,
                        feature_names=feature_names,
                        target_names=target_names,
                        train_set=datasets.train,
                        scale_factor=scale_factor,
                        scale_bias=scale_bias,
                        settings=settings,
                    )
                not_improved = 0
            else:  # If early measure is not better
                not_improved += 1
            # If not improved in 'early_stop' epoch, stop
            if (
                settings["early_stop_measure"] != "none"
                and not_improved >= settings["early_stop_after"]
            ):
                if save_checkpoint_networks:
                    nn_checkpoint_file = os.path.join(
                        checkpoint_dir, "nn_checkpoint_" + str(epoch_idx) + ".json"
                    )
                    trainable = {
                        x.name: tf.to_double(x).eval(session=sess).tolist()
                        for x in tf.trainable_variables()
                    }
                    model_to_json(
                        nn_checkpoint_file,
                        trainable=trainable,
                        feature_names=feature_names,
                        target_names=target_names,
                        train_set=datasets.train,
                        scale_factor=scale_factor,
                        scale_bias=scale_bias,
                        settings=settings,
                    )

                print("Not improved for %s epochs, stopping.." % (not_improved))
                stop_reason = "early_stopping"
                break

            # Stop if loss is nan or inf
            if np.isnan(lo) or np.isinf(lo):
                print("Loss is {}! Stopping..".format(lo))
                stop_reason = "NaN or inf loss"
                break

        # Clean break from train loop without a reason. Probably epochs ran out
        if stop_reason == "undefined":
            stop_reason = "max epochs"

    # Stop on Ctrl-C
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
        print("KeyboardInterrupt Stopping..")

    train_writer.close()
    validation_writer.close()

    # Restore checkpoint with best epoch
    try:
        best_epoch = epoch_idx - not_improved
        saver.restore(sess, saver.last_checkpoints[best_epoch - epoch_idx])
    except IndexError:
        print("Can't restore old checkpoint, just saving current values")
        best_epoch = epoch.eval(session=sess)

    cur_train_time.load(time.time() - train_start + prev_train_time, session=sess)
    validation_log.loc[epoch_idx] = (
        epoch_idx,
        cur_train_time.eval(session=sess),
        lo,
        meanse,
        meanabse,
        l1norm,
        l2norm,
        stab_pos,
    )
    train_log.loc[epoch_idx * minibatches + step] = (
        epoch_idx,
        cur_train_time.eval(session=sess),
        lo,
        meanse,
        meanabse,
        l1norm,
        l2norm,
        stab_pos,
    )
    validation_log.loc[epoch_idx:].to_csv(validation_log_file, header=False, float_format=ffmt)
    train_log.loc[epoch_idx * minibatches :].to_csv(
        train_log_file, header=False, float_format=ffmt
    )
    train_log_file.close()
    del train_log
    validation_log_file.close()
    del validation_log

    full_timeline.save("full_timeline.json")

    trainable = {
        x.name: tf.to_double(x).eval(session=sess).tolist() for x in tf.trainable_variables()
    }

    model_to_json(
        final_nn_file,
        trainable=trainable,
        feature_names=feature_names,
        target_names=target_names,
        train_set=datasets.train,
        scale_factor=scale_factor,
        scale_bias=scale_bias,
        settings=settings,
    )

    print(
        "Best epoch was {:d} with measure '{:s}' of {:f} ".format(
            best_epoch, settings["early_stop_measure"], best_early_measure
        )
    )

    print("Training time was {:.0f} seconds".format(cur_train_time.eval(session=sess)))

    # Finally, check against validation set
    xs, ys = datasets.validation.next_batch(-1, shuffle=False)
    feed_dict = {x: xs, y_ds: ys, is_train: False}
    format = lambda x: to_precision(x, 4, strip_zeros=True)
    rms_val = format(np.sqrt(mse.eval(feed_dict, session=sess)))
    rms_val_descale = format(np.sqrt(mse_descale.eval(feed_dict, session=sess)))
    loss_val = format(loss.eval(feed_dict, session=sess))
    l2_loss_val = format(l2_loss.eval(feed_dict, session=sess))
    print("{:22} {!s}".format("Validation RMS error: ", rms_val))
    print("{:22} {!s}".format("Descaled validation RMS error: ", rms_val_descale))
    print("{:22} {!s}".format("Validation loss: ", loss_val))

    metadata = {
        "epoch": int(epoch_idx),
        "best_epoch": int(best_epoch),
        "rms_validation": rms_val,
        "loss_validation": loss_val,
        "l2_loss_validation": l2_loss_val,
        "rms_validation_descaled": rms_val_descale,
        "walltime [s]": cur_train_time.eval(session=sess),
        "stop_reason": stop_reason,
    }

    try:
        stable_positive_loss_val = format(stable_positive_loss.eval(feed_dict, session=sess))
        metadata["stable_positive_loss_validation"] = stable_positive_loss_val
    except:
        pass
    try:
        import socket

        metadata["hostname"] = socket.gethostname()
    except:
        pass

    # Add metadata dict to nn.json
    with open(final_nn_file) as nn_file:
        data = json.load(nn_file, object_pairs_hook=OrderedDict)

    ordered_dict_prepend(data, "_metadata", metadata)
    data["_parsed_settings"] = settings

    with open(final_nn_file, "w") as nn_file:
        json.dump(data, nn_file, indent=4, separators=(",", ": "))

    sess.close()


def train_NDNN_from_folder(warm_start_nn=None, restore_old_checkpoint=False):
    with open("./settings.json") as file_:
        settings = json.load(file_)
    train(
        settings,
        warm_start_nn=warm_start_nn,
        restore_old_checkpoint=restore_old_checkpoint,
    )


def main(_):
    nn = None
    # from run_model import QuaLiKizNDNN
    # nn = QuaLiKizNDNN.from_json('nn.json')
    train_NDNN_from_folder(warm_start_nn=nn)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
    #                    default=True,
    #                    help='If true, uses fake data for unit testing.')
    # parser.add_argument('--max_steps', type=int, default=100000,
    # parser.add_argument('--max_steps', type=int, default=sys.maxsize,
    #                    help='Number of steps to run trainer.')
    # parser.add_argument('--learning_rate', type=float, default=10.,
    #                    help='Initial learning rate')
    # parser.add_argument('--dropout', type=float, default=0.9,
    #                    help='Keep probability for training dropout.')
    # parser.add_argument('--data_dir', type=str,
    #                    default='train_NN_run/input_data/',
    #                    help='Directory for storing input data')
    # parser.add_argument('--log_dir', type=str, default='train_NN_run/logs/',
    #                    help='Summaries log directory')
    # FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
