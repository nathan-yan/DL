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
        self.target_coordinates = [[[x, y, 1] for x in range (image_dimensions[1])] for y in range (image_dimensions[0])]

        # Define localization network
        self.localization_network = localization_network 
        self.add_module("localization_network", self.localization_network)

    def forward(self, x):
        transform = self.localization_network.forward(x)

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
    
    #a = STN(3)

    for i in a.parameters():
        print(i)
    print(a.parameters())

if __name__ == "__main__":
    main()