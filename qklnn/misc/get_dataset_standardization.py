import os

import pandas as pd
from IPython import embed

root = "/home/karel/working/qlk_data"
set_name = "training_gen3_7D_nions0_flat_filter8"
store = pd.HDFStore(os.path.join(root, set_name + ".h5.1"))

inp = store["input"]
desc = inp.describe()
stats = desc.loc[("mean", "std"), :]
for outvar in ["efeETG_GB"]:
    var = store["/output/" + outvar]
    desc = var.describe()
    stats.loc[("mean", "std"), desc.name] = (desc["mean"], desc["std"])
stats = stats.T
stats.index.name = "name"
stats.to_csv(set_name + ".csv")
