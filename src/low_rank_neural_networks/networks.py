import torch
import torch.nn as nn
from .layers import *


# Define custom neural network architecture from input dictionary
class Network(nn.Module):
    def __init__(self, net_architecture):
        """Constructs a neural network given its network architectu re.
        Args:
            net_architecture: Dictionary of the network architecture.
                              Needs keys 'type' and 'dims'.
                              Low-rank layers need key 'rank'.
        """
        # define Network as child of nn.Module
        super(Network, self).__init__()
        self._layers = torch.nn.Sequential()

        createLayers = LayerFactory()

        # check if layers are consistent
        try:
            for layerPrev, layer in zip(net_architecture[:-1], net_architecture[1:]):
                if layerPrev["dims"][1] != layer["dims"][0]:
                    raise AssertionError("Error: Dimensions of layers not matching.")
        except AssertionError as err:
            print(err)
            exit()

        # define layers
        for i, layer in enumerate(net_architecture):
            self._layers.add_module(name=f"hidden_{i+1}",
                                    module=createLayers(layer))

    def forward(self, x):
        """Returns the output of the neural network.
           The formula implemented is z_k = layer_k(z_{k-1}), where z_0 = x.
        Args:
            x: input image or batch of input images
        Returns: 
            output neural network for given input
        """
        x = x.view(-1, 784)  # Flatten the input image
        for layer in self._layers:
            x = layer(x)
        return x

    def step(self, learningRate):
        """Performs training step on all layers
        Args:
            learningRate: learning rate for training
        Returns: 
            output neural network for given input
        """
        for layer in self._layers:
            layer.step(learningRate)

    def storeParameters(self, location):
        """Stores parameters of all layers
        Args:
            location: folder location where to store params
        """
        for name, layer in self._layers.named_children():
            layer.storeParameters(location, f"_{name}")
