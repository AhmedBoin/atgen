# import copy
import math
import random
from typing import List, Tuple

import torch
from torch import nn

from layers import ActiSwitch, Conv2D, Linear, Flatten, MaxPool2D, Pass


class DNA:
    def __init__(self, input_size=None, default_activation=nn.ReLU()):
        self.input_size: Tuple[int, int] = input_size
        self.flatten: Flatten = None
        self.conv: List[Tuple[Conv2D, ActiSwitch]] = []
        self.maxpool: List[Tuple[int, MaxPool2D]] = []
        self.linear: List[Tuple[Linear, ActiSwitch]] = []
        self.default_activation: nn.ReLU = default_activation
    
    def append_conv(self, conv):
        self.conv.append(conv)
    
    def append_linear(self, linear):
        self.linear.append(linear)
    
    def append_maxpool(self, maxpool):
        self.maxpool.append(maxpool)
    
    def __str__(self) -> str:
        return f'DNA Sequence:\n\tInput Size: {self.input_size}{f"\n\tConv: {len(self.conv)}" if self.conv else ""}{f"\n\tMaxPool: {len(self.maxpool)}" if self.maxpool else ""}{f"\n\tLinear: {len(self.linear)}" if self.linear else ""}'
    
    def reconstruct(self):
        layers = []
        if self.conv:
            for layer in self.conv:
                layers.append(layer[0])
                if isinstance(layer[1], ActiSwitch):
                    layers.append(layer[1])
        if self.maxpool:
            for layer in self.maxpool:
                layers.insert(*layer)
        if self.flatten is not None:
            layers.append(self.flatten)
        if self.linear:
            for layer in self.linear:
                layers.append(layer[0])
                if isinstance(layer[1], ActiSwitch):
                    layers.append(layer[1])
        
        return layers
    
    def linear_size(self):
        return [layer[0].out_features for layer in self.linear]
    
    def conv_size(self):
        return [layer[0].out_channels for layer in self.conv]
    
    def rearrange(self):
        if self.maxpool:
            len_conv, len_maxpool = len(self.conv), len(self.maxpool)
            gap = math.ceil(len_conv/len_maxpool)

            for i in range(1, len_maxpool):
                self.maxpool[i-1][0] = i * gap * 2 + (i - 1)
            self.maxpool[-1][0] = len_conv * 2 +len_maxpool
    
    def evolve_linear_network(self, idx: int=None):
        if self.linear:
            idx = random.randint(0, len(self.linear)-1) if idx is None else idx
            self.linear.insert(idx, [
                Linear.init_identity_layer(self.linear[idx][0].in_features, True if self.linear[idx][0].bias is not None else False, self.linear[idx][0].norm_type), 
                ActiSwitch(self.default_activation, True)
            ])
    
    def evolve_conv_network(self, idx: int=None):
        if self.conv:
            idx = random.randint(0, len(self.conv)-1) if idx is None else idx
            self.conv.insert(idx, [
                Conv2D.init_identity_layer(self.conv[idx][0].in_channels, self.conv[idx][0].kernel_size, True if self.conv[idx][0].bias is not None else False, self.conv[idx][0].norm), 
                ActiSwitch(self.default_activation, True)
            ])
    
    def evolve_linear_layer(self, idx: int=None):
        if len(self.linear) > 1: # to avoid changing output layer shape
            idx = random.randint(0, len(self.linear) - 2) if idx is None else idx
            self.linear[idx][0].add_neuron()
            if idx < len(self.linear) - 1: # avoid out of range, this case handle crossover during the process of adding neuron to output layer
                self.linear[idx + 1][0].add_weight()
        elif idx is not None: # handle if only 1 output layer in crossover
            self.linear[idx][0].add_neuron()
    
    def evolve_conv_layer(self, idx: int=None, end_layer=False):
        if self.conv:
            idx = random.randint(0, len(self.conv) - 1) if idx is None else idx
            self.conv[idx][0].add_output_channel()
            if idx < (len(self.conv) - 1): # handle transition from Conv to Linear in crossover
                self.conv[idx + 1][0].add_input_channel()
            else:
                if not end_layer:
                    channels, features = self.conv[idx][0].out_channels, self.linear[0][0].in_features
                    for _ in range(int(features*channels/(channels-1)-features+1)):
                        self.linear[0][0].add_weight()

    @torch.no_grad()
    def evolve_weight(self, mutation_rate, perturbation_rate):
        if self.conv:
            for layer in self.conv:
                for param in layer[0].parameters():
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(param) * perturbation_rate  # Adjust perturbation magnitude
                        param.add_(noise)
                for param in layer[1].parameters():
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(param) * perturbation_rate  # Adjust perturbation magnitude
                        param.add_(noise)
        if self.linear:
            for layer in self.linear:
                for param in layer[0].parameters():
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(param) * perturbation_rate  # Adjust perturbation magnitude
                        param.add_(noise)
                for param in layer[1].parameters():
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(param) * perturbation_rate  # Adjust perturbation magnitude
                        param.add_(noise)

    def evolve_activation(self, activation_dict: List[nn.Module]):
        if self.conv:
            idx = random.randint(0, len(self.conv)-1)
            if isinstance(self.conv[idx][1], ActiSwitch):
                self.conv[idx][1].change_activation(random.choice(activation_dict))
        if len(self.linear) > 1:
            idx = random.randint(0, len(self.linear)-2)
            if isinstance(self.linear[idx][1], ActiSwitch):
                self.linear[idx][1].change_activation(random.choice(activation_dict))

    def prune(self, threshold: float = 0.01):
        """
        Prune neurons with weights below a given threshold.
        
        Args:
            threshold (float): The threshold below which neurons will be pruned.
        """
        for i, layer in enumerate(self.linear[:-1]):
            if layer[0].out_features > 1:
                # Identify neurons to prune
                neurons_to_prune = []
                for neuron_idx in range(layer[0].out_features):
                    neuron_weights = layer[0].weight[neuron_idx].abs()
                    if torch.max(neuron_weights) < threshold:
                        neurons_to_prune.append(neuron_idx)
                
                for neuron_idx in reversed(neurons_to_prune):
                    if layer[0].out_features > 1:
                        layer[0].remove_neuron(neuron_idx)
                        self.linear[i + 1][0].remove_weight(neuron_idx)

        for i, layer in enumerate(self.conv[:-1]):
            if layer[0].out_channels > 1:
                # Identify filters to prune
                filters_to_prune = []
                for neuron_idx in range(layer[0].out_channels):
                    neuron_weights = layer[0].weight[neuron_idx].abs()
                    if torch.max(neuron_weights) < threshold:
                        filters_to_prune.append(neuron_idx)
                
                for neuron_idx in reversed(filters_to_prune):
                    if layer[0].out_channels > 1:
                        layer[0].remove_output_channel(neuron_idx)
                        self.conv[i + 1][0].remove_input_channel(neuron_idx)
    
    

