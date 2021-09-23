import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from IPython import embed

# store = pd.HDFStore('./clipped_nions0_zeffx1_nustar1e-3_sepfluxes.h5')
store = pd.HDFStore("filtered_clipped_nions0_zeffx1_nustar1e-3_sepfluxes_0.1_60.h5")
leqfilter = (store["/gam_GB_leq2max"] != 0).index
# df = df[filterpass]
# merge = store['input'].loc[df[df < 0].index]; merge['efe_GB'] = df[df < 0]; print(merge.sample(200))


def plot_zerohist(dfs, name, ax=None, symmetric=True, plot_zero=True, statnames=None):
    dfsnozero = []
    dfsnozeroinf = []
    zeros = []
    neginf = []
    posinf = []
    nan = []
    colors = []
    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    minminpow = np.inf
    maxmaxpow = -np.inf
    for df, color in zip(dfs, default_colors):
        dfnozero = df.loc[df != 0]
        dfsnozero.append(dfnozero)
        dfnozeroinf = dfnozero.loc[
            (dfnozero != -np.inf) & (dfnozero != np.inf) & ~dfnozero.isnull()
        ]
        dfsnozeroinf.append(dfnozeroinf)

        zeros.append(len(df) - len(dfnozero))
        neginf.append(np.sum(df == -np.inf))
        posinf.append(np.sum(df == np.inf))
        nan.append(np.sum(df == -np.nan))

        minpow = np.ceil(np.log10(np.abs(dfnozeroinf.min())))
        maxpow = np.ceil(np.log10(np.abs(dfnozeroinf.max())))
        minminpow = min(minminpow, minpow)
        maxmaxpow = max(maxmaxpow, maxpow)
        colors.append(color)

    maxbin = [10 ** ii for ii in range(-3, int(maxmaxpow) + 1)]
    if symmetric:
        minbin = -np.flip(np.array(maxbin), 0)
    else:
        minbin = np.flip([-(10 ** ii) for ii in range(-3, int(minminpow) + 1)], 0)

    binbins = np.hstack([minbin, [0], maxbin])
    print(binbins)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    hist, bins, __ = ax.hist(dfsnozeroinf, binbins, log=True, rwidth=0.95, color=colors)
    ax.set_xscale("symlog", linthreshx=10 ** -3)
    ax.set_title(name)
    # ax.set_xlabel(name)
    # ax.set_ylabel('counts')
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)

    for ii, (zero, color) in enumerate(zip(zeros, colors)):
        if plot_zero:
            ax.add_patch(
                patches.Rectangle((-5e-4, 0), 1e-3, zero, color=color)  # (x,y)  # width  # height
            )

    minhistpow = np.floor(np.log10(np.min(hist[0 != hist])))
    maxhistpow = np.floor(np.log10(np.max(hist[0 != hist])))
    maxhistpow = max([maxhistpow, np.ceil(np.log10(np.min(zeros[0])))])
    ax.set_ylim([10 ** minhistpow, 10 ** maxhistpow])

    idx = [(bins[ii], bins[ii + 1]) for ii in range(len(bins) - 1)]
    for ii, id in enumerate(idx):
        if id[0] == 0:
            zeroind = ii
    if statnames == None:
        statnames = range(len(hist))
    stats = pd.DataFrame({ii: array for ii, array in zip(statnames, hist)})
    stats["bins"] = idx
    stats.loc[-1] = zeros + [(0, 0)]
    stats.loc[-2] = neginf + [(-np.inf, -np.inf)]
    stats.loc[-3] = posinf + [(np.inf, np.inf)]
    stats.loc[-4] = nan + [(np.nan, np.nan)]
    stats = stats.set_index("bins")
    # stats = stats.sort()
    print(stats)


len(store.keys())
rows = 4
cols = 3
for ii, name in enumerate(store):
    print(name)
    if name in ["/index", "/input"]:
        continue
    if name in ["/gam_GB_leq2max", "/gam_GB_less2max"]:
        continue
        store[name].hist()

    df = store[name]
    dfs = [df]
    for filter in [
        (store["/gam_GB_leq2max"] != 0),
        (store["/gam_GB_less2max"] != 0),
    ]:
        dfs.append(df.loc[filter])

    row = int(np.floor(ii / cols))
    col = ii % cols
    ax = plt.subplot2grid((rows, cols), (row, col))
    plot_zerohist(dfs, name[1:], ax=ax, statnames=["nofilter", "gam_leq2!=0", "gam_less2!=0"])
    plt.tight_layout()
plt.show()
