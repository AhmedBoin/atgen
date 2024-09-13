import pickle
import torch
from torch import nn

from typing import List
import random
from inspect import isfunction

from utils import BLUE, BOLD, RESET_COLOR
from layers.activations import ActiSwitch, Pass
from layers.linear import Linear, Flatten, LazyLinear
from layers.conv import Conv2D, MaxPool2D, LazyConv2D
from dna import DNA

from math import fabs as abs

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ATNetwork(nn.Module):
    """
    ATNetwork: A neural network model capable of evolving its structure and weights.
    This class provides methods to dynamically change the network architecture, 
    such as adding or removing layers, evolving weights, and modifying activation functions.
    """
    def __init__(self, *layers, activation=nn.ReLU(), last_activation=None, bias=True, input_size=None):
        """
        Initialize the ATNetwork.

        Args:
            layers (List[int], optional): A list defining the number of neurons in each layer.
            activation (nn.Module, optional): The default activation function to use in all layers except the last. Defaults to nn.ReLU().
            last_activation (nn.Module, optional): The activation function to use in the last layer. Defaults to None (no activation).
            bias (bool, optional): Whether to include bias terms in the layers. Defaults to True.
            backprob_phase (bool, optional): Indicates if back propagation learning-based evolution is enabled. Defaults to True.
        """
        super(ATNetwork, self).__init__()
        if isinstance(layers[0], int):
            self.layers = nn.ModuleList([Linear(layers[idx], layers[idx + 1], bias) for idx in range(len(layers) - 1)])
            self.activation = nn.ModuleList([
                *[ActiSwitch(activation) for _ in range(len(layers) - 2)], 
                Pass() if last_activation is None else last_activation
            ])
        else:
            self.layers = nn.ModuleList()
            self.activation = nn.ModuleList()
            for layer in layers:
                if isinstance(layer, ActiSwitch):
                    if len(self.activation) > 0: # if ActiSwitch is the first layer
                        self.activation.pop(-1)
                    self.activation.append(layer)
                elif isinstance(layer, Pass):
                    pass
                else:
                    self.layers.append(layer)
                    self.activation.append(Pass())

            if input_size is not None:
                self.store_sizes(input_size)

        self.input_size = input_size
        self.default_activation = activation


    def forward(self, x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.activation[i](self.layers[i](x))
        return x
    

    def store_sizes(self, input_size=None):
        input_size = input_size if input_size is not None else self.input_size

        for layer in self.layers:
            if isinstance(layer, Conv2D):
                input_size, channels = layer.store_sizes(input_size)
            elif isinstance(layer, MaxPool2D):
                input_size, channels = layer.store_sizes(input_size, channels)
            elif isinstance(layer, LazyConv2D):
                layer.custom_init(channels)
                input_size, channels = layer.store_sizes(input_size)
            elif isinstance(layer, Flatten):
                input_size, channels = layer.store_sizes(input_size, channels)
            elif isinstance(layer, LazyLinear):
                input_size = layer.custom_init(input_size)


    def genotype(self) -> DNA:
        genome = DNA(self.input_size, self.default_activation)
        for i, (layer, activation) in enumerate(zip(self.layers, self.activation)):
            if isinstance(layer, Linear) or isinstance(layer, LazyLinear):
                genome.append_linear([layer, activation])
            elif isinstance(layer, Conv2D) or isinstance(layer, LazyConv2D):
                genome.append_conv([layer, activation])
            elif isinstance(layer, MaxPool2D):
                genome.append_maxpool([i, layer])
            elif isinstance(layer, Flatten):
                genome.flatten = layer
            
        return genome
    

    @classmethod
    def phenotype(cls, genome: DNA) -> "ATNetwork":
        return cls(*genome.reconstruct(), input_size=genome.input_size)


    def save_network(self, file_name="ATNetwork.pth"):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self, file)


    @classmethod
    def load_network(cls, file_name="ATNetwork.pth") -> "ATNetwork":
        with open(f'{file_name}', 'rb') as file:
            network: ATNetwork = pickle.load(file)
        return network


    @torch.no_grad()
    def summary(self):
        print(f"{BOLD}{BLUE}Model Summary{RESET_COLOR}{BOLD}:")
        print("-" * 100)
        print(f"{'Layer':<11}{'Type':<15}{'Output Shape':<30}{'Parameters':<15}{'Activation':<15}")
        print("-" * 100)

        total_param = 0
        for i, (layer, activation) in enumerate(zip(self.layers, self.activation)):
            total_param += layer.params_count
            layer.print_layer(i)
            activation.print_layer(i)

        print("-" * 100)
        print(f"{BLUE}{'Total Parameters:':<25}{RESET_COLOR}{BOLD}{total_param:,}{RESET_COLOR}")




if __name__ == "__main__":
    # helper loss function
    def loss(x1:torch.Tensor, x2: torch.Tensor):
        val = torch.abs((x2/x1).mean() - 1) * 100
        print(f"loss = {val:.10f}%")


    model = ATNetwork(5, 3, 2, 1)
    class CustomNetwork(ATNetwork):
        def __init__(self):
            # Do not call the parent __init__ with layers; initialize manually, Directly call nn.Module's init
            super(ATNetwork, self).__init__()
            self.layers = nn.ModuleList([
                Linear(5, 3),
                Linear(3, 1)
            ])
            self.activation = nn.ModuleList([
                ActiSwitch(),
                Pass()
            ])

            self.input_size = None
            self.backprob_phase = True
            self.default_activation = nn.ReLU()

    model = CustomNetwork()
    model.summary()
    # summary(model, input_size=(5,))

    x = torch.randn(4, 5)
    y1: torch.Tensor = model(x)
    
    model = model.genotype()
    for _ in range(10):
        model.evolve_linear_network()
    # model.summary()
    # summary(model, input_size=(5,))
    for _ in range(500):
        model.evolve_linear_layer()
    # model.summary()
    # summary(model, input_size=(5,))
    model = ATNetwork.phenotype(model)
        
    y2: torch.Tensor = model(x)
    # model.prune()
    model.summary()
    # summary(model, input_size=(5,))

    print(torch.cat((y1, y2),  dim=1))  
    loss(y1, y2)


    # Sequential implementation
    import torch.nn.functional as F
    model = ATNetwork(
        Linear(10, 5),
        ActiSwitch(F.relu),
        Linear(5, 3),
        Linear(3, 3),
        ActiSwitch(nn.ReLU()),
        Linear(3, 1),
    )
    model.summary()

    model = ATNetwork(
        Conv2D(3, 32, kernel_size=3),
        ActiSwitch(nn.ReLU()),
        Conv2D(32, 32, kernel_size=3),
        ActiSwitch(nn.ReLU()),
        MaxPool2D(),
        Conv2D(32, 64, kernel_size=3),
        ActiSwitch(nn.ReLU()),
        Conv2D(64, 64, kernel_size=3),
        ActiSwitch(nn.ReLU()),
        MaxPool2D(),
        Flatten(),
        Linear(3136, 100),
        ActiSwitch(nn.ReLU()),
        Linear(100, 1),
        input_size=(28, 28)
    )
    model.summary()

    x = torch.randn(64, 3, 28, 28)
    y = model(x)
    print(y.shape)

    genome = model.genotype()
    print(genome)

    model = ATNetwork.phenotype(genome)
    model.summary()

    model = ATNetwork(10, 5, 3, 1)
    genome = model.genotype()
    model.summary()
    print(genome)

    model = ATNetwork.phenotype(genome)
    model.summary()




