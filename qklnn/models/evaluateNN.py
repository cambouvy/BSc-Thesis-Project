from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import r2_score
import os
from ffnn import QuaLiKizNDNN
import numpy as np
import pandas as pd
import json
import math
from qlknn.dataset.data_io import save_to_store, load_from_store

""" Compute the MSE/RMSE of all NNs in the grid search"""
if __name__ == "__main__":
    input, data, const = load_from_store("/m100_work/FUAC5_GKNN/camille/results/test_gen5_7D_pedformreg8_filter11.h5.1", dask=False)
    input = input.iloc[:40000,:]
    data = data.iloc[:40000,:]
    mse_nns = {}
    rmse_nns={}

    for cnt in range(1, 26):
        root = os.path.dirname(os.path.realpath(__file__))
        path = "../../networks_structure2/nn" + str(cnt) + ".json"
        nn_path = os.path.join(root, path)
        nn = QuaLiKizNDNN.from_json(nn_path)
        input = input[nn._feature_names]
        fluxes = nn.get_output(input)

        mse = MeanSquaredError()
        mse_nns[cnt] = mse(data["efiITG_GB"], fluxes).numpy()
        rmse_nns[cnt] = math.sqrt(mse_nns[cnt])


    print("MSE: ", mse_nns)
    print("RMSE: ", rmse_nns)
    print("Best NN: ", min(mse_nns, key=mse_nns.get))
