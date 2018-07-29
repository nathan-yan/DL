import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed", category = RuntimeWarning)

import pickle

import matplotlib.pyplot as plt 
import numpy as np

from sklearn.preprocessing import OneHotEncoder 

import torch
import torch.nn as nn

class STN(nn.Module):
    def __init__(self, localization_network, image_dimensions):
        """Initialize a STN module

        localization_network: a network that outputs the appropriate transformation parameters 
        """
        
        super(STN, self).__init__()

        self.image_dimensions = image_dimensions
        target_coordinates = torch.tensor(
                                                [
                                                    [
                                                        [y, x, 1] for x in range (image_dimensions[1])
                                                    ] for y in range (image_dimensions[0])
                                                ], dtype = torch.float32)
        
        target_coordinates = target_coordinates.permute(2, 0, 1)
        self.target_coordinates = target_coordinates.view(1, 3, -1)

        # Define localization network
        self.localization_network = localization_network 
        self.add_module("localization_network", self.localization_network)

    def forward(self, x):
        # transform => batchsize x 6
        transform = self.localization_network.forward(x)
        transform = transform.view(-1, 2, 3)

        # source_locations => batchsize x 2 x prod(image_dimensions)
        source_locations = torch.matmul(transform, self.target_coordinates) 

        x = source_locations[:, 1, :]
        y = source_locations[:, 0, :]

class LocalizationNetwork(nn.Module):
    def __init__(self, layers, activation = nn.ReLU):
        super(LocalizationNetwork, self).__init__()

        self.activation = activation
        self.layers = []

        conv = 0
        linear = 0
        for l in layers:
            if (len(l) > 2):
                m = nn.Conv2d(*l)
                conv += 1

                self.layers.append(m)
                self.add_module('conv_' + str(conv), m)
            else:
                m = nn.Linear(*l)
                linear += 1

                self.layers.append(m)
                self.add_module('linear_' + str(linear), m)

    def forward(self, x):
        for index, l in enumerate(self.layers):
            x = l(x)

            if (index != len(self.layers) - 1):
                x = self.activation(x)
        
        return x

def main():
    image_dimensions = [3, 3]
    target_coordinates = torch.tensor([[[y, x, 1] for x in range (image_dimensions[1])] for y in range (image_dimensions[0])], dtype = torch.float32)
    target_coordinates = target_coordinates.permute(2, 0, 1)
    target_coordinates = target_coordinates.view(3, -1)

 #   target_coordinates = target_coordinates.repeat(2, 1, 1)
    print(target_coordinates.size())
    parameters = torch.tensor([[[
        1, 1, 1
    ],
    [
        1, 1, 1
    ]], [
        [
        2, 2, 2
    ],
    [
        2, 2, 2
    ]
    ]], dtype = torch.float32)
    print(parameters.size())

    print(torch.matmul(parameters, target_coordinates))

    a = STN(3)

    for i in a.parameters():
        print(i)
    print(a.parameters())

if __name__ == "__main__":
    main()