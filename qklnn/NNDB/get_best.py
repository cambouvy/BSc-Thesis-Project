from IPython import embed

# import mega_nn
import gc
import numpy as np
import pandas as pd
from itertools import chain
from model import Network, NetworkJSON, NetworkMetadata, Hyperparameters
from peewee import Param
import os
import sys

networks_path = os.path.abspath(os.path.join((os.path.abspath(__file__)), "../../networks"))
sys.path.append(networks_path)
from run_model import QuaLiKizNDNN


target_to_fancy = {
    "efeETG_GB": "Electron ETG Heat Flux",
    "efeITG_GB": "Electron ITG Heat Flux",
    "efeTEM_GB": "Electron TEM Heat Flux",
    "efiITG_GB": "Ion ITG Heat Flux",
    "efiTEM_GB": "Ion TEM Heat Flux",
}
feature_names = ["An", "Ate", "Ati", "Ti_Te", "q", "smag", "x"]
feature_names2 = ["Ati", "Ate", "An", "q", "smag", "x", "Ti_Te"]
query = Network.select(Network.target_names).distinct().tuples()
# df = pd.DataFrame(columns=['target_names', 'id','rms'])
results = pd.DataFrame()
for ii, query_res in enumerate(query):
    target_names = query_res[0]
    subquery = (
        Network.select(Network.id, NetworkMetadata.rms_validation)
        .where(Network.target_names == Param(target_names))
        .where(
            (Network.feature_names == Param(feature_names))
            | (Network.feature_names == Param(feature_names2))
        )
        .join(NetworkMetadata)
        .order_by(NetworkMetadata.rms_validation)
        .tuples()
    )
    df = pd.DataFrame(list(subquery), columns=["id", "rms"])
    index = pd.MultiIndex.from_product([[tuple(target_names)], df.index])
    df.index = index
    # df.loc[ii] = list(chain([target_names], subquery.get()))
    results = pd.concat([results, df])
results["id"] = results["id"].astype("int64")

# for row in results.iterrows():
#    results.set_value(row[0], 'target_names', target_to_fancy[row[1]['target_names']])
# results = results[['target_names', 'l2_norm', 'rms', 'rms_rel']]
# results.columns = ['Training Target', 'L_2 norm', 'RMS error [GB]', 'RMS error [%]']
# print(results.to_latex(index=False))
embed()
