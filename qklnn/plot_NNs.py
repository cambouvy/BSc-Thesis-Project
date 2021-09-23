from qlknn.models.ffnn import QuaLiKizNDNN
import json
import os
from warnings import warn
from collections import OrderedDict
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import shutil

""" Plot all the networks from the grid search using the quickslicer"""

if __name__ == "__main__":
    for cnt in range(1,26):
        # root = os.path.dirname(os.path.realpath(__file__))
        stats_path =  "/m100/home/userexternal/cbouvy00/QLKNN-develop/nn_stats/nn" + str(cnt)
        # stats_path = os.path.join(root, path)
        if os.path.exists(stats_path):
            shutil.rmtree(stats_path)
        os.mkdir(stats_path)
        os.chdir(stats_path)
        src_path = "nn_source" + str(cnt) +".py"
        with open(src_path, 'w') as f:
            text = "from qlknn.models.ffnn import QuaLiKizNDNN\n" \
                   "nns = {} \n" \
                   "nn = QuaLiKizNDNN.from_json(\"/m100/home/userexternal/cbouvy00/QLKNN-develop/networks_structure2/nn" + str(cnt) + ".json\")\n" \
                   "nn.label = \"nn_" + str(cnt) +"\" \n"\
                   +"nns[nn.label] = nn \n" \
                    "slicedim = \"Ati\"\n" \
                    "style = \"mono\""
            f.write(text)

        cli_command = "quickslicer --totstats-to-disk --dump-slice-input --dump-slice-output --summary-to-disk --mode=pretty --fraction=0.00000001  /m100_work/FUAC5_GKNN/camille/results/sane_gen5_7D_pedformreg8_filter11.h5.1 " + src_path
        os.system(cli_command)
        os.chdir("..")
