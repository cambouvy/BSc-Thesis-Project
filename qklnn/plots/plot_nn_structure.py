from math import cos, sin, atan

import numpy as np
import pandas as pd
import matplotlib as mpl

# mpl.use('pdf')
import matplotlib.pyplot as plt
from IPython import embed

from qlknn.models.ffnn import QuaLiKizNDNN

pretty = False


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        plt.gca().add_patch(circle)


class Layer:
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weights):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return (
            self.horizontal_distance_between_neurons
            * (self.number_of_neurons_in_widest_layer - number_of_neurons)
            / 2
        )

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, **kwargs):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = plt.Line2D(line_x_data, line_y_data, **kwargs)
        plt.gca().add_line(line)

    def draw(self, layerType=0, draw_weights=True, drop_connection_frac=1, cutoff=0):
        for this_layer_neuron_index, neuron in enumerate(self.neurons):
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron_index, previous_layer_neuron in enumerate(
                    self.previous_layer.neurons
                ):
                    if draw_weights is True:
                        weight = self.previous_layer.weights[
                            this_layer_neuron_index, previous_layer_neuron_index
                        ]
                        if weight > 0:
                            color = "g"
                        else:
                            color = "r"
                    else:
                        weight = 1
                        color = "b"
                    if np.random.random() < drop_connection_frac:
                        if np.abs(weight) >= cutoff:
                            self.__line_between_two_neurons(
                                neuron, previous_layer_neuron, linewidth=weight, c=color
                            )
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            plt.text(x_text, self.y, "Input Layer", fontsize=12)
        elif layerType == -1:
            plt.text(x_text, self.y, "Output Layer", fontsize=12)
        else:
            plt.text(x_text, self.y, "Hidden Layer " + str(layerType), fontsize=12)


class NeuralNetwork:
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, weights):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, weights)
        self.layers.append(layer)

    def draw(self, draw_weights=True):
        plt.figure()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i, draw_weights=draw_weights)
        plt.axis("scaled")
        plt.axis("off")
        plt.title("Neural Network architecture", fontsize=15)
        plt.show()


class DrawNN:
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def draw(self):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer)
        for l in self.neural_network:
            network.add_layer(l)
        network.draw()


if __name__ == "__main__":
    mode = "debug"

    # store = pd.HDFStore('../gen2_7D_nions0_flat.h5')
    # slicedim, style, nns = nns_from_NNDB()
    # slicedim, style, nns = nns_from_manual()
    # nn = nns['NN model']
    nn = QuaLiKizNDNN.from_json("nn001.json")
    import networkx as nx

    network = NeuralNetwork(96)
    for layer in nn.layers:
        network.add_layer(layer._weights.shape[0], layer._weights.T)
    network.add_layer(layer._weights.shape[1], np.ones(layer._weights.shape[1]))
    network.draw()
