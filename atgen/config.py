from torch import nn
from typing import List

from utils import activation_functions


class ATGENConfig:
    def __init__(self, default_activation=nn.ReLU(), last_activation=None, bias=True, crossover_rate = 0.8, crossover_decay_rate=1.0, 
                 last_crossover_rate=0.8, mutation_decay_rate=1.0,last_mutation_rate=0.001, weight_mutation_rate=0.1, perturbation_rate=0.9, 
                 add_neuron_mutation_rate=0.01, add_filter_mutation_rate=0.001, linear_mutation_rate=0.0001, conv_mutation_rate=0.0001, 
                 activation_mutation_rate=0.0001, threshold=0.01, activation_dict: List[nn.Module]=activation_functions, 
                 species_metrics=None, linear_start=True, input_size=None):
        
        self.default_activation = default_activation
        self.last_activation = last_activation
        self.bias = bias

        self.crossover_rate = crossover_rate
        self.crossover_decay_rate = crossover_decay_rate
        self.last_crossover_rate = last_crossover_rate

        self.mutation_decay_rate = mutation_decay_rate
        self.last_mutation_rate = last_mutation_rate

        self.weight_mutation_rate = weight_mutation_rate
        self.perturbation_rate = perturbation_rate

        self.add_neuron_mutation_rate = add_neuron_mutation_rate
        self.add_filter_mutation_rate = add_filter_mutation_rate
        self.linear_mutation_rate = linear_mutation_rate
        self.conv_mutation_rate = conv_mutation_rate

        self.activation_mutation_rate = activation_mutation_rate
        self.activation_dict = activation_dict

        self.species_metrics = species_metrics
        self.threshold = threshold
        self.linear_start = linear_start
        self.input_size = input_size


    def crossover_step(self):
        self.crossover_rate = max(self.mutation_decay_rate*self.crossover_rate, self.last_crossover_rate)


    def mutation_step(self):
        if self.weight_mutation_rate > 0:
            self.weight_mutation_rate = max(self.mutation_decay_rate*self.weight_mutation_rate, self.last_mutation_rate)
        if self.add_neuron_mutation_rate > 0:
            self.add_neuron_mutation_rate = max(self.mutation_decay_rate*self.add_neuron_mutation_rate, self.last_mutation_rate)
        if self.add_filter_mutation_rate > 0:
            self.add_filter_mutation_rate = max(self.mutation_decay_rate*self.add_filter_mutation_rate, self.last_mutation_rate)
        if self.linear_mutation_rate > 0:
            self.linear_mutation_rate = max(self.mutation_decay_rate*self.linear_mutation_rate, self.last_mutation_rate)
        if self.conv_mutation_rate > 0:
            self.conv_mutation_rate = max(self.mutation_decay_rate*self.conv_mutation_rate, self.last_mutation_rate)
        if self.activation_mutation_rate > 0:
            self.activation_mutation_rate = max(self.mutation_decay_rate*self.activation_mutation_rate, self.last_mutation_rate)

