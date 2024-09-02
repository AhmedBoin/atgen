import numpy
import torch
from torch import nn

from tqdm import tqdm
import multiprocessing

import copy
import random
from typing import List, Tuple
import pickle

from network import ATNetwork
from memory import ReplayBuffer
from utils import RESET_COLOR, BLUE, GREEN, RED, BOLD, GRAY, activation_functions, print_stats_table


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class ATGEN(nn.Module):
    memory: ReplayBuffer = None
    def __init__(self, population_size: int, layers: List[int], activation=nn.ReLU, last_activation=None, bias=True, 
                 weight_mutation_rate=0.8, perturbation_rate=0.9, layer_mutation_rate=0.2, network_mutation_rate=0.01, 
                 activation_mutation_rate=0.001, threshold=0.01, activation_dict: List[nn.Module]=None, 
                 buffer_size = int(1e5), batch_size: int = 64, backprob_phase=False, experiences_phase=False, device="cpu"):
        """
        Initialize the Genetic Algorithm for evolving neural networks.

        Args:
            population_size (int): The number of networks in the population.
            layers (List[int]): The structure of layers for initial networks.
            activation (function): Activation function for hidden layers.
            last_activation (function): Activation function for the last layer.
            bias (bool): Whether to use bias in layers.
            mutation_rate (float): Probability of mutating a network.
            crossover_rate (float): Probability of performing crossover between two networks.
            threshold (float): Threshold for pruning neurons with low weights.
        """
        self.population_size = population_size
        self.perturbation_rate = perturbation_rate
        self.weight_mutation_rate = weight_mutation_rate
        self.layer_mutation_rate = layer_mutation_rate
        self.network_mutation_rate = network_mutation_rate
        self.activation_mutation_rate = activation_mutation_rate
        self.threshold = threshold
        
        # network parameters
        self.activation_dict = activation_functions if activation_dict is None else activation_dict
        ATGEN.memory = ReplayBuffer(layers[0], buffer_size, batch_size, device)
        self.backprob_phase = backprob_phase
        self.experiences_phase = experiences_phase
        
        # Initialize the population
        self.population = [ATNetwork(*layers, activation=activation, last_activation=last_activation, bias=bias, backprob_phase=backprob_phase).to(device) for _ in range(population_size)]
        self.fitness_scores = [0.0] * population_size
        self.selection_probs = self.fitness_scores
        self.best_fitness = float("-inf")
        self.last_fitness = float("-inf")

    @torch.no_grad()
    def evaluate_fitness(self):
        """
        Evaluate the fitness of each network in the population.
        """
        self.fitness_scores = [0.0] * len(self.population)

        for i, network in enumerate(tqdm(self.population, desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100)):
            self.fitness_scores[i] = self.fitness_fn(network)
        
        # args = [(i, network, self.fitness_fn) for i, network in enumerate(self.population)]
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     for index, fitness_score in tqdm(pool.imap_unordered(evaluate_fitness_single, args), total=len(args), desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100):
        #         self.fitness_scores[index] = fitness_score

    def evaluate_learn(self):
        """
        """
        for genome in tqdm(self.population, desc=f"{BLUE}Generation Refinement{RESET_COLOR}", ncols=100):
            self.backprob_fn(genome)

        # args = [(genome, self.learn_fn) for genome in self.population]

        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     for _ in tqdm(pool.imap_unordered(refine_genome, args), total=len(args), desc=f"{BLUE}RL Generation Refinement{RESET_COLOR}", ncols=100):
        #         pass

    def select_parents(self) -> Tuple[ATNetwork, ATNetwork]:
        """
        Select two parents from the population based on fitness scores using roulette wheel selection.
        
        Returns:
            Tuple[ATNetwork, ATNetwork]: Two selected parent networks.
        """
        parent1 = random.choices(self.population, weights=self.selection_probs, k=1)[0]
        parent2 = random.choices(self.population, weights=self.selection_probs, k=1)[0]
        
        return parent1, parent2

    @torch.no_grad()
    def crossover(self, parent1: ATNetwork, parent2: ATNetwork) -> ATNetwork:
        """
        Perform a more vectorized crossover between two parent networks to produce a child network.
        
        Args:
            parent1 (ATNetwork): The first parent network.
            parent2 (ATNetwork): The second parent network.
        
        Returns:
            ATNetwork: A new child network created from crossover.
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        size1 = child1.genome_type()
        size2 = child2.genome_type()
        
        # Evolve layers to match neuron counts for crossover
        for i, (i1, i2) in enumerate(zip(size1, size2)):
            if i1 < i2:
                for _ in range(i2 - i1):
                    child1.evolve_layer(i)
            elif i2 < i1:
                for _ in range(i1 - i2):
                    child2.evolve_layer(i)

        # Determine final child layer sizes
        size1 = child1.genome_type()
        size2 = child2.genome_type()
        size = [child1.layers[0].in_features] + (size1 if len(size1) > len(size2) else size2)

        # Initialize the child network
        activation = child1.default_activation if random.random() > 0.5 else child2.default_activation
        child = ATNetwork(*size, activation=activation, backprob_phase=child1.backprob_phase).to(device=child1.layers[0].weight.device)

        for layer_idx, layer_size in enumerate(size[1:]):
            if layer_idx < min(len(child1.layers), len(child2.layers)):
                weight1 = child1.layers[layer_idx].weight
                weight2 = child2.layers[layer_idx].weight
                bias1 = child1.layers[layer_idx].bias if child1.layers[layer_idx].bias is not None else None
                bias2 = child2.layers[layer_idx].bias if child2.layers[layer_idx].bias is not None else None

                # Apply random mask to weights and biases for crossover
                mask = torch.rand(layer_size, device=weight1.device) < 0.5
                child_layer_weight = torch.where(mask.unsqueeze(1).expand_as(weight1), weight1, weight2)
                    
                if bias1 is not None and bias2 is not None:
                    child_layer_bias = torch.where(mask, bias1, bias2)
                else:
                    child_layer_bias = bias1 if bias1 is not None else bias2

                child.layers[layer_idx].weight.data.copy_(child_layer_weight)
                if child_layer_bias is not None:
                    child.layers[layer_idx].bias.data.copy_(child_layer_bias)

                # Crossover activation function
                if layer_idx == len(child1.activation) - 1:
                    child.activation[layer_idx] = child2.activation[layer_idx]
                elif layer_idx == len(child2.activation) - 1:
                    child.activation[layer_idx] = child1.activation[layer_idx]
                else:
                    child.activation[layer_idx] = child1.activation[layer_idx] if random.random() < 0.5 else child2.activation[layer_idx]
                
            elif layer_idx < len(child1.layers):
                child.layers[layer_idx] = child1.layers[layer_idx]
                child.activation[layer_idx] = child1.activation[layer_idx]
            elif layer_idx < len(child2.layers):
                child.layers[layer_idx] = child2.layers[layer_idx]
                child.activation[layer_idx] = child2.activation[layer_idx]

        return child

    @torch.no_grad()
    def mutate(self, network: ATNetwork):
        """
        Mutate the given network with a small probability.

        Args:
            network (ATNetwork): The network to mutate.
        """
        
        if random.random() < self.layer_mutation_rate:
            network.evolve_layer()
        if random.random() < self.network_mutation_rate:
            network.evolve_network()
        if random.random() < self.activation_mutation_rate:
            network.evolve_activation(self.activation_dict)

        network.evolve_weight(self.weight_mutation_rate, self.perturbation_rate)

    @torch.no_grad()
    def prune_network(self, network: ATNetwork):
        """
        Prune neurons in the network with weights below the threshold.

        Args:
            network (ATNetwork): The network to prune.
        """
        network.prune(threshold=self.threshold)

                
    def run_generation(self) -> float:
        """
        Run a single generation of the genetic algorithm, including evaluation, selection, crossover, mutation, and pruning.
        
        Args:
            fitness_fn (function): A function to evaluate the fitness of a network.
        """
        
        self.evaluate_fitness()

        # preview training data
        max_fitness = max(self.fitness_scores)
        min_fitness = min(self.fitness_scores)
        mean_fitness = numpy.mean(self.fitness_scores)
        color = GREEN if max_fitness > self.best_fitness else GRAY if max_fitness > self.last_fitness else RED
        print_current_best = f"{BLUE}Best Fitness{RESET_COLOR}: \t {BOLD}{color}{ max_fitness }{RESET_COLOR}"
        self.last_fitness = max_fitness 
        if max_fitness > self.best_fitness: 
            self.best_fitness = max_fitness 
            self.memory.clear()

        # selection of parents
        fits = sorted([fit for fit in self.fitness_scores], reverse=True)[:self.population_size//2]
        min_fit = min(fits)
        fits = [fit-min_fit for fit in fits] # make all positive numbers
        total_fitness = sum(fits)
        self.selection_probs = [score / total_fitness for score in fits] # convert to probabilities
        
        sorted_population = [ind for _, ind in sorted(zip(self.fitness_scores, self.population), key=lambda x: x[0], reverse=True)]
        self.population = sorted_population[:self.population_size//2]  # Select top 50%
        
        new_population: List[ATNetwork] = []
        for _ in tqdm(range(self.population_size//2), desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population.extend(new_population)
        print(print_current_best)
        print(f"{BLUE}{BOLD}Best{RESET_COLOR}", end=" ")
        self.population[0].summary()

        # use best genome to make experiences
        if self.experiences_phase:
            self.experiences_fn(self.population[0]) 

        # smooth the generation networks a little based back propagation
        if self.backprob_phase:
            self.evaluate_learn()

        print_stats_table(self.best_fitness, max_fitness, mean_fitness, min_fitness, self.population_size)
        self.save_population()
        return max_fitness, mean_fitness, min_fitness

    def evolve(self, generation: int=None, fitness: int=None, save_name: str=None, metrics: int=0, plot: bool=False):
        last_fitness = -float("inf")
        if generation is not None:
                for i in range(generation):
                    print(f"\n--- {BLUE}Generation {i+1}{RESET_COLOR} ---")
                    last_fitness = self.run_generation()[metrics]
                    if fitness is not None:
                        if last_fitness >= fitness:
                            print(f"\n--- {BLUE}Fitness reached{RESET_COLOR} ---")
                            break

        elif fitness is not None:
            generation =  0
            while fitness > last_fitness:
                generation += 1
                print(f"\n--- {BLUE}Generation {generation}{RESET_COLOR} ---")
                last_fitness = self.run_generation()[metrics]
            print(f"\n--- {BLUE}Fitness reached{RESET_COLOR} ---")
            
        else:
            raise ValueError("evolve using generation number or fitness value")
        
        if save_name is not None:
            self.population[0].save_network(save_name)

        if plot:
            "plot using matplotlib"
        else:
            return "max, mean, min"
    
    def fitness_fn(self, genome):
        raise NotImplementedError("implement fitness_fn method")
    
    def backprob_fn(self, genome):
        raise NotImplementedError("implement learn_fn method")
    
    def experiences_fn(self, genome):
        raise NotImplementedError("implement store_experiences method")

    def save_population(self, file_name="population.pkl"):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self.population, file)

    def load_population(self, file_name="population.pkl"):
        with open(f'{file_name}', 'rb') as file:
            self.population = pickle.load(file)
    

def evaluate_fitness_single(args):
    """
    Helper function to evaluate the fitness of a single network.
    
    Args:
        args (tuple): A tuple containing the index of the network and the network object itself.
    
    Returns:
        tuple: A tuple containing the index and the fitness score.
    """
    index, network, fitness_fn = args
    fitness_score = fitness_fn(network)
    return (index, fitness_score)

def refine_genome(args):
    genome, learn_fn = args
    learn_fn(genome)
    return genome 


    
if __name__ == "__main__":
    # Create parent networks with specified architectures
    parent1 = ATNetwork(10, 2, 5, 1)
    parent2 = ATNetwork(10, 2, 4, 3, 1)
    
    # Initialize the ATGEN instance
    ga = ATGEN(population_size=10, layers=[8, 1, 4])
    
    # Perform crossover
    child = ga.crossover(parent1, parent2)
    child.summary()

