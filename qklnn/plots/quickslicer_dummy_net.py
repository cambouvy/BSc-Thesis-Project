""" quickslicer Neural Network specification

This file should be importable from the quickslicer and define three things:

nns:      A dictionary-like object containing the neural networks. Each network
          should have some basic attributes, a full description is outside the
          scope, but if its a QuaLiKizNDNN you are usually set.
slicedim: The dimension (string) that should be sliced over. Should exist
          and as input feature of the network in the to-be-sliced 'input' leaf
style:    The style of slicing, usually just corrosponds to the output dimensions,
          but its partially implemented to e.g. slice 1D output from a 3D output network.
          Should be one of 'mono' 'duo' or 'triple'
"""

from collections import OrderedDict
import os

nns = OrderedDict()

# Example pulling a bunch of networks from the NNDB
# from qlknn.NNDB.model import Network
# dbnns = []
# labels = []
# dbnns.append(Network.get_by_id(1723))

# for ii, dbnn in enumerate(dbnns):
#    net = dbnn.to_QuaLiKizNN()
#    if len(labels) == 0:
#        net.label = '_'.join([str(el) for el in [dbnn.__class__.__name__ , dbnn.id]])
#    else:
#        net.label = labels[ii]
#    nns[net.label] = net

# Example using a bunch of on-disk 'Phillip' style late-fusion Neural Networks from disk
# from qlknn.models.kerasmodel import NDHornNet
# network_root = '/home/philipp/Documents/Job/NeuralNets/thirdNets'
# network_names = ['ITG']
#
# for network_name in network_names:
#    # Warning! If you use the name 'nn', Keras won't be able to load the network..
#    net = NDHornNet(os.path.join(network_root, network_name, 'nn.json'))
#    net.label = network_name
#    nns[net.label] = net

# Example setting up QuaLiKizNDNN from tests
from qlknn.models.ffnn import QuaLiKizNDNN, QuaLiKizComboNN

root = os.path.dirname(os.path.realpath(__file__))
net_path = os.path.join(root, "../../tests/gen3_test_files/Network_874_efiITG_GB/nn.json")
net = QuaLiKizNDNN.from_json(net_path)
net.label = "Network_874"
nns[net.label] = net

# Example setting up QuaLiKiz4DNN from disk
# from qlkANNk import QuaLiKiz4DNN
# nns['4D'] = QuaLiKiz4DNN()
# nns['4D'].label = '4D'
# nns['4D']._target_names = ['efeITG_GB', 'efiITG_GB']

# Set the slice dim manually based on target names
slicedim = "Ati"
if len(net._target_names) == 1:
    style = "mono"
elif len(net._target_names) == 2:
    style = "duo"
elif len(net._target_names) == 3:
    style = "triple"
