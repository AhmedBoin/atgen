from torch import nn
from typing import List

from utils import activation_functions


class ATGENConfig:
    def __init__(self, default_activation=nn.ReLU(), last_activation=None, bias=True, threshold=0.01, activation_dict=activation_functions, 
                 remove=True, crossover_rate = 0.8, crossover_decay_rate=1.0, dropout_population_via_crossover=True, mutate_parents=True,
                 last_crossover_rate=0.8, mutation_decay_rate=1.0,last_mutation_rate=0.001, weight_mutation_rate=0.1, perturbation_rate=0.9, 
                 neuron_mutation_rate=0.01, filter_mutation_rate=0.001, linear_mutation_rate=0.0001, conv_mutation_rate=0.0001, 
                 activation_mutation_rate=0.0001,
                 species_metrics=None, linear_start=True, input_size=None):
        
        self.default_activation = default_activation
        self.last_activation = last_activation
        self.bias = bias

        self.dropout_population_via_crossover = dropout_population_via_crossover
        self.mutate_parents = mutate_parents
        
        self.crossover_rate = crossover_rate
        self.crossover_decay_rate = crossover_decay_rate
        self.last_crossover_rate = last_crossover_rate
        
        self.mutation_decay_rate = mutation_decay_rate
        self.last_mutation_rate = last_mutation_rate
        self.remove = remove
        
        self.weight_mutation_rate = weight_mutation_rate
        self.perturbation_rate = perturbation_rate
        
        self.neuron_mutation_rate = neuron_mutation_rate
        self.filter_mutation_rate = filter_mutation_rate
        self.linear_mutation_rate = linear_mutation_rate
        self.conv_mutation_rate = conv_mutation_rate
        
        self.activation_mutation_rate = activation_mutation_rate
        self.activation_dict: List[nn.Module] = activation_dict
        
        self.species_metrics = species_metrics
        self.threshold = threshold
        self.linear_start = linear_start
        self.input_size = input_size
        

    def crossover_step(self):
        self.crossover_rate = max(self.mutation_decay_rate*self.crossover_rate, self.last_crossover_rate)


    def mutation_step(self):
        if self.weight_mutation_rate > 0:
            self.weight_mutation_rate = max(self.mutation_decay_rate*self.weight_mutation_rate, self.last_mutation_rate)
        if self.neuron_mutation_rate > 0:
            self.neuron_mutation_rate = max(self.mutation_decay_rate*self.neuron_mutation_rate, self.last_mutation_rate)
        if self.filter_mutation_rate > 0:
            self.filter_mutation_rate = max(self.mutation_decay_rate*self.filter_mutation_rate, self.last_mutation_rate)
        if self.linear_mutation_rate > 0:
            self.linear_mutation_rate = max(self.mutation_decay_rate*self.linear_mutation_rate, self.last_mutation_rate)
        if self.conv_mutation_rate > 0:
            self.conv_mutation_rate = max(self.mutation_decay_rate*self.conv_mutation_rate, self.last_mutation_rate)
        if self.activation_mutation_rate > 0:
            self.activation_mutation_rate = max(self.mutation_decay_rate*self.activation_mutation_rate, self.last_mutation_rate)

