from torch import nn
from typing import List


class ATGENConfig:
    def __init__(self, default_activation=nn.ReLU(), last_activation=None, bias=True, 
                 crossover_rate = 0.8, weight_mutation_rate=0.01, perturbation_rate=0.9, add_neuron_mutation_rate=0.01, 
                 add_filter_mutation_rate=0.0, linear_mutation_rate=0.0, conv_mutation_rate=0.0, activation_mutation_rate=0.0, 
                 threshold=0.01, activation_dict: List[nn.Module]=None, linear_start=True, input_size=None):
        
        self.default_activation = default_activation
        self.last_activation = last_activation
        self.bias = bias
        self.crossover_rate = crossover_rate
        self.weight_mutation_rate = weight_mutation_rate
        self.perturbation_rate = perturbation_rate
        self.add_neuron_mutation_rate = add_neuron_mutation_rate
        self.add_filter_mutation_rate = add_filter_mutation_rate
        self.linear_mutation_rate = linear_mutation_rate
        self.conv_mutation_rate = conv_mutation_rate
        self.activation_mutation_rate = activation_mutation_rate
        self.threshold = threshold
        self.activation_dict = activation_dict
        self.linear_start = linear_start
        self.input_size = input_size