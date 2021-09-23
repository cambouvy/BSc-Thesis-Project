import re

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from peewee import AsIs, JOIN, prefetch, SQL
from IPython import embed

from bokeh.layouts import row, column
from bokeh.plotting import figure, show, output_file
from bokeh.transform import linear_cmap
from bokeh.models import (
    ColumnDataSource,
    Range1d,
    LabelSet,
    Label,
    Rect,
    HoverTool,
    Div,
)

from qlknn.NNDB.model import (
    Network,
    PureNetworkParams,
    PostprocessSlice,
    NetworkMetadata,
    TrainMetadata,
    Postprocess,
    db,
    Hyperparameters,
)
from qlknn.plots.statistical_spread import get_base_stats
from qlknn.misc.to_precision import to_precision

# First, get some statistics
target_names = ["efeTEM_GB"]
hyperpars = ["cost_stable_positive_scale", "cost_l2_scale"]
# hyperpars = ['cost_stable_positive_scale', 'cost_stable_positive_offset']
goodness_pars = [
    "rms",
    "no_pop_frac",
    "no_thresh_frac",
    "pop_abs_mis_median",
    "thresh_rel_mis_median",
    "wobble_qlkunstab",
]
try:
    report = get_base_stats(target_names, hyperpars, goodness_pars)
except Network.DoesNotExist:
    report = pd.DataFrame(columns=goodness_pars, index=["mean", "stddev", "stderr"])
query = (
    Network.select(
        Network.id.alias("network_id"),
        PostprocessSlice,
        Postprocess.rms,
        Hyperparameters,
    )
    .join(PostprocessSlice, JOIN.LEFT_OUTER)
    .switch(Network)
    .join(Postprocess, JOIN.LEFT_OUTER)
    .switch(Network)
    .where(Network.target_names == target_names)
    .switch(Network)
    .join(PureNetworkParams)
    .join(Hyperparameters)
    .where(Hyperparameters.cost_stable_positive_offset.cast("numeric") == -5)
    .where(Hyperparameters.cost_stable_positive_function == "block")
)

if query.count() > 0:
    results = list(query.dicts())
    df = pd.DataFrame(results)
    # df['network'] = df['network'].apply(lambda el: 'pure_' + str(el))
    # df['l2_norm'] = df['l2_norm'].apply(np.nanmean)
    df.drop(["id", "network"], inplace=True, axis="columns")
    df.set_index("network_id", inplace=True)
    stats = df
stats = stats.applymap(np.array)
stats = stats.applymap(lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x)
stats.dropna(axis="columns", how="all", inplace=True)
stats.dropna(axis="rows", how="all", inplace=True)

stats = stats.loc[:, hyperpars + goodness_pars]
stats.reset_index(inplace=True)
# stats.set_index(hyperpars, inplace=True)
# stats.sort_index(ascending=False, inplace=True)
# stats = stats.groupby(level=list(range(len(stats.index.levels)))).mean() #Average equal hyperpars
# stats.reset_index(inplace=True)
aggdict = {"network_id": lambda x: tuple(x)}
aggdict.update({name: "mean" for name in goodness_pars})
stats_mean = stats.groupby(hyperpars).agg(aggdict)
aggdict.update({name: "std" for name in goodness_pars})
stats_std = stats.groupby(hyperpars).agg(aggdict)
stats = stats_mean.merge(stats_std, left_index=True, right_index=True, suffixes=("", "_std"))
stats.reset_index(inplace=True)

for name in hyperpars:
    stats[name] = stats[name].apply(str)

for name in goodness_pars:
    fmt = lambda x: "" if np.isnan(x) else to_precision(x, 4)
    fmt_mean = stats[name].apply(fmt)
    stats[name + "_formatted"] = fmt_mean
    fmt = lambda x: "" if np.isnan(x) else to_precision(x, 2)
    fmt_std = stats[name + "_std"].apply(fmt)
    prepend = lambda x: "+- " + x if x != "" else x
    stats[name + "_std_formatted"] = fmt_std.apply(prepend)


x = np.unique(stats[hyperpars[1]].values)
x = sorted(x, key=lambda x: float(x))
y = np.unique(stats[hyperpars[0]].values)
y = sorted(y, key=lambda x: float(x))

source = ColumnDataSource(stats)

plotmode = "bokehz"
hover = HoverTool(
    tooltips=[
        ("network_id", "@network_id"),
        (hyperpars[0], "@" + hyperpars[0]),
        (hyperpars[1], "@" + hyperpars[1]),
    ]
)
plots = []
for statname in goodness_pars:
    fmt = lambda x: "" if np.isnan(x) else to_precision(x, 2)
    title = "{:s} (ref={:s}±{:s})".format(
        statname,
        fmt(report[statname]["mean"]),
        fmt(report[statname]["stddev"] + report[statname]["stderr"]),
    )

    p = figure(title=title, tools="tap", toolbar_location=None, x_range=x, y_range=y)
    p.add_tools(hover)
    color = linear_cmap(statname, "Viridis256", min(stats[statname]), max(stats[statname]))
    p.rect(
        x=hyperpars[1],
        y=hyperpars[0],
        width=1,
        height=1,
        source=source,
        fill_color=color,
        line_color=None,
        nonselection_fill_alpha=0.4,
        nonselection_fill_color=color,
    )
    non_selected = Rect(fill_alpha=0.8)
    label_kwargs = dict(
        x=hyperpars[1],
        y=hyperpars[0],
        level="glyph",
        source=source,
        text_align="center",
        text_color="red",
    )
    labels = LabelSet(text=statname + "_formatted", text_baseline="bottom", **label_kwargs)
    labels_std = LabelSet(text=statname + "_std_formatted", text_baseline="top", **label_kwargs)
    p.add_layout(labels)
    p.add_layout(labels_std)
    p.xaxis.axis_label = hyperpars[1]
    p.yaxis.axis_label = hyperpars[0]
    plots.append(p)

from bokeh.layouts import layout, widgetbox

title = Div(text=",".join(target_names))
l = layout([[title], [plots]])
show(l)
